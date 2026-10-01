"""Federation simulator: CoSE (trust-gated collective self-evolution) and LoRA baselines.

Each node streams one domain at a time (three domains per node, abrupt switches), trains a
LoRA adapter on the frozen GPT-2 read-out block, and exchanges knowledge according to the
selected method.  Honest nodes belong to one of two labelling-policy clusters; cluster B
uses a conflicting taxonomy on two news classes.  Unreliable nodes flip labels, send
noise, or send sign-flipped amplified updates.
"""
import argparse, copy, hashlib, json, math, os, time
import numpy as np, torch
import torch.nn.functional as F
from cose_core import (Backbone, Data, Library, MATS, DIMS, DOMAINS, DOM_LABELS, forward,
                       gates, lora_to_factors, zero_lora, ece)

DEF = dict(K=12, T=36, seg=12, batch=32, p_act=0.8, lr=3e-3, r=8, scale=2.0, steps=(1, 2, 3),
           attackers='flip,noise,scale', conflict=1, eval_every=3, eval_n=200, method='cose',
           mu=1.0, ditto_lam=0.1, ties_k=0.2, n_anchor=40, buf=48,
           # CoSE
           tau_r=0.15, temp=0.05, emit_every=3, emit_eps=0.2, ph_delta=0.05, ph_lambda=3.0,
           rank_c=8, tau_d=0.5, min_emit=3, tau_acc=0.5, tau_gamma=0.02, w_b=0.4, w_g=4.0, w_k=3.0, w_u=1.0, w_rho=3.0,
           mi_ref=0.5, kappa0=0.7, rho_q=0.6, rep_decay=0.95, conf_ref=0.6, kappa_min=0.45, delay=2,
           use_drift=1, use_gate=1, use_spectral=1, use_routing=1, use_rep=1, use_unc=1,
           use_cons=1, use_gain=1, revalidate=1,
           # scenarios
           sleeper=0, sleeper_on=12, audit=27, rollback='repair', joiner=0, join_at=24,
           seed=0, out='')


class Node:
    pass


def remap_label(cfg, node, y, dom):
    y = y.clone()
    if node.policy == 'B' and cfg['conflict']:
        y[y == 3] = 2            # AG: Sci/Tech articles filed under Business
        y[y == 8] = 9            # R8: money-fx filed under interest
    if node.kind == 'flip' or (node.kind == 'sleeper' and node.t_now >= cfg['sleeper_on']):
        labs = DOM_LABELS[dom]
        perm = {l: labs[(i + 1) % len(labs)] for i, l in enumerate(labs)}
        y = torch.tensor([perm[int(v)] for v in y])
    return y


def eval_labels(cfg, policy, y):
    y = y.clone()
    if policy == 'B' and cfg['conflict']:
        y[y == 3] = 2; y[y == 8] = 9
    return y


class Sim:
    def __init__(self, cfg, data, bb):
        self.c = cfg; self.d = data; self.bb = bb
        self.rng = np.random.RandomState(cfg['seed'])
        self.gen = torch.Generator().manual_seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        self.lib_cfg = dict(tau_r=cfg['tau_r'], temp=cfg['temp'], routed=bool(cfg['use_routing']))
        self.build_nodes()
        self.comm = np.zeros(len(self.nodes))
        self.log = {'curve': [], 'events': []}
        self.cap_store = {}      # id -> capsule (relay store, latest versions)
        self.harm = [0, 0]
        self.vote_cache = {}       # accepted / received harmful capsules (receptions by honest nodes)

    # ------------------------------------------------------------------ setup
    def build_nodes(self):
        c, rng = self.c, self.rng
        att = [a for a in c['attackers'].split(',') if a and a != 'none']
        if c['sleeper']:
            att = ['sleeper'] * c['sleeper']
        K = c['K']; nh = K - len(att)
        while True:
            sched = [list(rng.choice(6, 3, replace=False)) for _ in range(K)]
            cnt = np.bincount(sum(sched[:nh], []), minlength=6)
            if cnt.min() >= 3:
                break
        self.nodes = []
        for i in range(K):
            n = Node(); n.id = i; n.kind = 'honest' if i < nh else att[i - nh]
            n.policy = 'A' if (i < nh and i < (nh + 1) // 2) else ('B' if i < nh else 'A')
            n.sched = [DOMAINS[k] for k in sched[i]]; n.tier = c['steps'][i % 3]; n.t_now = 0
            n.start = 0
            self.nodes.append(n)
        if c['joiner']:
            n = Node(); n.id = K; n.kind = 'honest'; n.policy = 'A'; n.tier = 2; n.t_now = 0
            n.sched = [DOMAINS[rng.randint(6)]] * 3; n.start = c['join_at']
            self.nodes.append(n)
        # anchors (public, unlabeled) and node partitions of each domain's training pool
        self.anchor = []
        part = {}
        for dname in DOMAINS:
            idx = np.setdiff1d(self.d.idx[(dname, 'train')], self.d.anchor); rng.shuffle(idx)
            rest = idx
            users = [n for n in self.nodes if dname in n.sched]
            for k, n in enumerate(users):
                part[(n.id, dname)] = rest[k::len(users)]
        self.anchor = self.d.anchor
        self.anchor_dom = self.d.dom[self.anchor]
        self.test = {dn: self.rng.choice(self.d.idx[(dn, 'test')], c['eval_n'], replace=False) for dn in DOMAINS}
        for n in self.nodes:
            n.part = {dn: part.get((n.id, dn), np.array([], int)) for dn in DOMAINS}
            n.ptr = {dn: 0 for dn in DOMAINS}
            n.lora = zero_lora(c['r'], self.gen)
            if c['method'] == 'ffa':
                for m in MATS:
                    n.lora[m][0] = self.shared_A[m] if hasattr(self, 'shared_A') else n.lora[m][0]
            n.opt = torch.optim.Adam([p for m in MATS for p in n.lora[m] if p.requires_grad], lr=c['lr'])
            n.lib = Library(); n.buf = {}; n.rep = {}; n.inbox = []; n.pool = {}
            n.ph = [0.0, 0.0, 0.0, 0]; n.regime = 0; n.version = 0; n.zsum = torch.zeros(self.d.Z.shape[1]); n.zcnt = 0
            n.last_emit = None; n.active_parents = set(); n.snaps = []; n.preq = []; n.seen = set()
            n.hist = {dn: [] for dn in DOMAINS}; n.dense = None; n.zhist = []; n.t_cp = 0; n.shift_run = 0; n.reg_rounds = 0
        if c['method'] == 'ffa':
            g = torch.Generator().manual_seed(1234 + c['seed'])
            self.shared_A = {m: (torch.randn(c['r'], DIMS[m][0], generator=g) / math.sqrt(DIMS[m][0])) for m in MATS}
            for n in self.nodes:
                for m in MATS:
                    n.lora[m][0] = self.shared_A[m].clone()
                n.opt = torch.optim.Adam([n.lora[m][1] for m in MATS], lr=c['lr'])
        if c['method'] == 'ditto':
            for n in self.nodes:
                n.glob = zero_lora(c['r'], self.gen)
                n.gopt = torch.optim.Adam([p for m in MATS for p in n.glob[m]], lr=c['lr'])
        if c['method'] in ('fedavg', 'fedprox', 'ffa', 'median', 'ditto'):
            src = self.nodes[0].glob if c['method'] == 'ditto' else self.nodes[0].lora
            self.server = {m: [t.detach().clone() for t in src[m]] for m in MATS}
            for n in self.nodes:
                self.load_global(n)

    # ------------------------------------------------------------------ helpers
    def batch_idx(self, n, dn, size):
        p = n.part[dn]
        out = []
        for _ in range(size):
            if n.ptr[dn] >= len(p):
                self.rng.shuffle(p); n.ptr[dn] = 0
            out.append(p[n.ptr[dn]]); n.ptr[dn] += 1
        return np.array(out)

    def fwd(self, n, idx, lora=None, lib=True, extra=None, probe=None, use_lora=True):
        d = self.d
        L = (n.lora if lora is None else lora) if use_lora else None
        gw = None
        if self.c['method'] == 'cose' and self.c['use_routing'] and n.zcnt > 0:
            wkey = F.normalize(n.zsum / n.zcnt, dim=0)
            gw = (d.Z[idx] @ wkey >= self.c['tau_r']).float()
        return forward(self.bb, d.H[idx], d.O[idx], d.Z[idx], lora=L,
                       lib=n.lib if lib else None, lib_cfg=self.lib_cfg, extra=extra, probe=probe,
                       scale=self.c['scale'], dense=n.dense, gw=gw)

    def load_global(self, n):
        tgt = n.glob if self.c['method'] == 'ditto' else n.lora
        with torch.no_grad():
            for m in MATS:
                for k in range(2):
                    tgt[m][k].copy_(self.server[m][k])

    def lora_vec(self, lora):
        return {m: (lora[m][0].detach().clone(), lora[m][1].detach().clone()) for m in MATS}

    def submitted(self, n, lora):
        """What a node uploads for factor-space aggregation (attack behaviour applied)."""
        v = self.lora_vec(lora)
        if n.kind == 'noise':
            v = {m: (v[m][0], torch.randn_like(v[m][1]) * (v[m][1].norm() / math.sqrt(v[m][1].numel()) + 1e-3) * 3)
                 for m in MATS}
        elif n.kind == 'scale':
            v = {m: (v[m][0], -5.0 * v[m][1]) for m in MATS}
        return v

    def malicious(self, n):
        return n.kind != 'honest'

    # ------------------------------------------------------------------ local training
    def local_round(self, n, t):
        c = self.c
        dn = n.sched[min((t - n.start) // c['seg'], 2)]
        new_seg = dn not in n.seen
        steps = n.tier
        idx = self.batch_idx(n, dn, c['batch'] * steps)
        y = remap_label(c, n, self.d.y[idx], dn)
        # prequential evaluation on the first mini-batch (before any update this round)
        with torch.no_grad():
            lg = self.fwd(n, idx[:c['batch']], lora=n.glob if False else None)
            if c['method'] == 'ditto':
                lg = self.fwd(n, idx[:c['batch']])
            pl = F.cross_entropy(lg, y[:c['batch']]).item()
            pa = (lg.argmax(1) == y[:c['batch']]).float().mean().item()
        n.preq.append((t, dn, pa, pl, new_seg)); n.cur_dom = dn
        if new_seg:
            n.switch_t = t
        n.seen.add(dn)
        z = self.d.Z[idx].mean(0)
        if c['method'] == 'cose':
            n.batch_key = F.normalize(self.d.Z[idx].mean(0), dim=0)
            self.cose_drift(n, t, pl, dn, new_seg)
            n.zsum += self.d.Z[idx].sum(0); n.zcnt += len(idx)
            n.zhist = [h for h in n.zhist if h[0] >= t - 4]
            b = n.buf.setdefault(dn, [])
            b.extend(list(zip(idx.tolist(), y.tolist())))
            n.buf[dn] = b[-c['buf']:]
        for s in range(steps):
            bi = idx[s * c['batch']:(s + 1) * c['batch']]; by = y[s * c['batch']:(s + 1) * c['batch']]
            if c['method'] == 'ditto':
                lg = forward(self.bb, self.d.H[bi], self.d.O[bi], self.d.Z[bi], lora=n.glob, scale=c['scale'])
                loss = F.cross_entropy(lg, by); n.gopt.zero_grad(); loss.backward(); n.gopt.step()
                lg = self.fwd(n, bi)
                reg = sum(((n.lora[m][k] - n.glob[m][k].detach()) ** 2).sum() for m in MATS for k in range(2))
                loss = F.cross_entropy(lg, by) + c['ditto_lam'] / 2 * reg
            else:
                lg = self.fwd(n, bi)
                loss = F.cross_entropy(lg, by)
                if c['method'] == 'fedprox':
                    loss = loss + c['mu'] / 2 * sum(((n.lora[m][k] - self.server[m][k]) ** 2).sum()
                                                    for m in MATS for k in range(2))
            n.opt.zero_grad(); loss.backward(); n.opt.step()
        if c['method'] == 'cose':
            n.zhist.append((t, self.d.Z[idx].sum(0), len(idx), self.lora_vec(n.lora)))
        return dn

    # ------------------------------------------------------------------ CoSE: drift, capsules
    def cose_drift(self, n, t, loss, dn, new_seg):
        c = self.c
        m_, mean, mn, k = n.ph
        k += 1; mean = mean + (loss - mean) / k
        m_ = m_ + loss - mean - c['ph_delta']
        if k == 1 or m_ <= mn:
            mn = m_; n.t_cp = t
        n.ph = [m_, mean, mn, k]
        shift = False
        if n.zcnt > 0:
            kreg = F.normalize(n.zsum / n.zcnt, dim=0)
            n.shift_run = n.shift_run + 1 if float(n.batch_key @ kreg) < c['tau_d'] else 0
            shift = n.shift_run >= 1
        if c['use_drift'] and ((k > 3 and (m_ - mn) > c['ph_lambda']) or shift):
            n.shift_run = 0
            if shift:
                n.t_cp = t - 1
            self.seal(n, t, n.t_cp)
            n.ph = [0.0, 0.0, 0.0, 0]
            self.log['events'].append(('drift', t, n.id, dn, (t - getattr(n, 'switch_t', -99)) <= 4 and n.switch_t > 0, n.kind))

    def effective_delta(self, n):
        """Working adapter plus the routed share of in-scope library capsules (dense matrices)."""
        c = self.c
        D = {m: (n.lora[m][0].detach().T @ n.lora[m][1].detach().T) * c['scale'] for m in MATS}
        parents = {}
        b = [i for i, _ in n.buf.get(n.cur_dom, [])]
        if n.lib.caps and b:
            keys, P, Q = n.lib.pack()
            g = gates(self.d.Z[np.array(b)], keys, c['tau_r'], c['temp'], bool(c['use_routing'])).mean(0)
            for k, cp in enumerate(n.lib.caps):
                if float(g[k]) > 0.05:
                    for m in MATS:
                        D[m] = D[m] + float(g[k]) * (P[m][k] @ Q[m][k])
                    if not cp['own']:
                        parents[cp['id']] = (float(g[k]), {m: cp['mask'][m].clone() for m in MATS})
        return D, parents

    def capsule_from(self, n, t, sealed):
        c = self.c
        rk = c['rank_c']
        D, par = self.effective_delta(n)
        n.active_parents |= set(par)
        P, Q = {}, {}
        for m in MATS:
            U, S, V = torch.svd_lowrank(D[m], q=rk + 4, niter=3)
            P[m] = (U[:, :rk] * S[:rk]).contiguous(); Q[m] = V[:, :rk].T.contiguous()
        if n.kind == 'noise':
            for m in MATS:
                nrm = 5.0 * max(P[m].norm().item(), 0.5)
                P[m] = torch.randn_like(P[m]); P[m] *= nrm / P[m].norm()
                Q[m] = torch.linalg.qr(torch.randn(Q[m].shape[1], rk))[0].T
        if n.kind == 'scale':
            P = {m: -5.0 * P[m] for m in MATS}
        key = F.normalize(n.zsum / max(n.zcnt, 1), dim=0)
        cid = f'{n.id}.{n.regime}.{n.version}'
        anc = set()
        for p in n.active_parents:
            anc |= {p} | self.cap_store.get(p, {}).get('anc', set())
        cap = dict(id=cid, src=n.id, regime=n.regime, version=n.version, t=t, key=key, P=P, Q=Q,
                   mask={m: torch.ones(rk) for m in MATS}, parents=set(n.active_parents), anc=anc, mix=par,
                   sealed=sealed, own=False, harmful=self.malicious(n) and (n.kind != 'sleeper' or t >= c['sleeper_on']),
                   digest=hashlib.sha256(b''.join(P[m].numpy().tobytes() for m in MATS)).hexdigest()[:16])
        n.version += 1
        return cap

    def broadcast(self, n, cap, t):
        self.cap_store[cap['id']] = cap
        nbytes = 4 * (sum(cap['P'][m].numel() + cap['Q'][m].numel() for m in MATS) + 32) + 256
        self.comm[n.id] += nbytes
        for o in self.nodes:
            if o.id != n.id:
                o.inbox.append((t + int(self.rng.randint(0, self.c['delay'] + 1)), cap))

    def seal(self, n, t, t_cp=None):
        # roll back to the estimated change point: adapter snapshot and key from pre-change rounds
        post = [h for h in n.zhist if t_cp is not None and h[0] > t_cp]
        pre = [h for h in n.zhist if t_cp is None or h[0] <= t_cp]
        for h in post:
            n.zsum = n.zsum - h[1]; n.zcnt -= h[2]
        if pre and post:
            snap = pre[-1][3]
            with torch.no_grad():
                for m in MATS:
                    n.lora[m][0].copy_(snap[m][0]); n.lora[m][1].copy_(snap[m][1])
        if n.zcnt == 0:
            return
        cap = self.capsule_from(n, t, sealed=True)
        own = dict(cap); own['own'] = True
        own['mask'] = {m: torch.ones(self.c['rank_c']) for m in MATS}
        if n.kind == 'honest' or n.kind == 'sleeper':
            n.lib.caps = [x for x in n.lib.caps if not (x['src'] == n.id and x['regime'] == n.regime)]
            n.lib.caps.append(own); n.lib.invalidate()
        self.broadcast(n, cap, t)
        n.lora = zero_lora(self.c['r'], self.gen)
        n.opt = torch.optim.Adam([p for m in MATS for p in n.lora[m]], lr=self.c['lr'])
        n.regime += 1; n.zsum = torch.zeros(self.d.Z.shape[1]); n.zcnt = 0; n.last_emit = None; n.zhist = []; n.reg_rounds = 0
        n.active_parents = set(); n.snaps = []

    def maybe_emit(self, n, t):
        c = self.c
        n.reg_rounds += 1
        if n.reg_rounds < c['min_emit'] or (t - n.start) % c['emit_every'] != c['emit_every'] - 1 or n.zcnt == 0:
            return
        cur = {m: (n.lora[m][0].detach().T @ n.lora[m][1].detach().T) * c['scale'] for m in MATS}
        nrm = math.sqrt(sum(float((cur[m] ** 2).sum()) for m in MATS))
        if n.last_emit is not None:
            dif = math.sqrt(sum(float(((cur[m] - n.last_emit[m]) ** 2).sum()) for m in MATS))
            if dif / (nrm + 1e-8) < c['emit_eps']:
                return
        if nrm < 1e-6:
            return
        n.last_emit = cur
        self.broadcast(n, self.capsule_from(n, t, sealed=False), t)

    # ------------------------------------------------------------------ CoSE: trust gate
    def anchor_scope(self, key):
        s = self.d.Z[self.anchor] @ key
        return self.anchor[(s >= self.c['tau_r']).numpy()]

    def buf_scope(self, n, key):
        items = [it for b in n.buf.values() for it in b]
        if not items:
            return np.array([], int), torch.tensor([])
        idx = np.array([i for i, _ in items]); y = torch.tensor([l for _, l in items])
        s = self.d.Z[idx] @ key
        keep = (s >= self.c['tau_r']).numpy()
        return idx[keep], y[torch.tensor(keep)]

    def cap_gate_vec(self, n, cap, idx):
        """Routing weight the capsule would receive on idx if added to the library."""
        keys = [x['key'] for x in n.lib.caps] + [cap['key']]
        g = gates(self.d.Z[idx], torch.stack(keys), self.c['tau_r'], self.c['temp'], bool(self.c['use_routing']))
        return g[:, -1]

    def predict(self, n, idx, cap=None, mask=None):
        extra = None
        if cap is not None:
            g = self.cap_gate_vec(n, cap, idx)
            P = cap['P'] if mask is None else {m: cap['P'][m] * mask[m][None] for m in MATS}
            extra = [(P, cap['Q'], g)]
        with torch.no_grad():
            return self.fwd(n, idx, extra=extra)

    def standalone(self, cap, idx, mask=None):
        """Prediction of the frozen backbone carrying only this capsule (gate 1 on idx)."""
        P = cap['P'] if mask is None else {m: cap['P'][m] * mask[m][None] for m in MATS}
        P = {m: P[m] * cap['mask'][m][None] for m in MATS}
        g = torch.ones(len(idx))
        with torch.no_grad():
            return forward(self.bb, self.d.H[idx], self.d.O[idx], self.d.Z[idx], extra=[(P, cap['Q'], g)],
                           scale=self.c['scale'])

    def anchor_votes(self, cap):
        if cap['id'] not in self.vote_cache:
            self.vote_cache[cap['id']] = self.standalone(cap, self.anchor).argmax(1)
        return self.vote_cache[cap['id']]

    def gate(self, n, cap, t):
        c = self.c
        sc_mask = (self.d.Z[self.anchor] @ cap['key'] >= c['tau_r']).numpy()
        A = self.anchor[sc_mask]
        if len(A) == 0:
            return False, {}
        info = {}
        pc = self.standalone(cap, A).softmax(1)
        # epistemic uncertainty: component-dropout mutual information of the standalone capsule
        if c['use_unc']:
            ps = []
            g = torch.Generator().manual_seed(int(t * 1000 + n.id))
            for _ in range(6):
                mk = {m: (torch.rand(c['rank_c'], generator=g) < 0.5).float() * 2 for m in MATS}
                ps.append(self.standalone(cap, A, mask=mk).softmax(1))
            ps = torch.stack(ps); pm = ps.mean(0)
            ent = lambda p: -(p * p.clamp(min=1e-9).log()).sum(-1)
            mi = float((ent(pm) - ent(ps).mean(0)).mean())
            info['mi'] = mi
        # behavioural consistency: agreement with the plurality of independent peer capsules
        # covering the same scope, plus the receiver's own confident predictions
        kappa = None
        if c['use_cons']:
            voters, wts = [], []
            latest = {}
            for (src, reg), oc in n.pool.items():
                if src == cap['src'] or src == n.id or float(oc['key'] @ cap['key']) < 0.5:
                    continue
                if src not in latest or oc['t'] > latest[src]['t']:
                    latest[src] = oc
            for src, oc in latest.items():
                rp = n.rep.get(src, [1.0, 1.0])
                voters.append(self.anchor_votes(oc)[sc_mask]); wts.append(rp[0] / (rp[0] + rp[1]))
            p0 = self.predict(n, A).softmax(1)
            own = p0.argmax(1).clone(); own[p0.max(1).values < c['conf_ref']] = -1
            bidx, _ = self.buf_scope(n, cap['key'])
            voters.append(own); wts.append(2.0 if len(bidx) >= 8 else 0.5)
            V = torch.stack(voters); Wt = torch.tensor(wts)[:, None]           # (v, a)
            cnt = torch.stack([((V == k).float() * Wt).sum(0) for k in range(20)])
            top = cnt.max(0); tot = ((V >= 0).float() * Wt).sum(0)
            ok_items = (tot >= 1.5) & (top.values / tot.clamp(min=1e-6) >= 0.6)
            if int(ok_items.sum()) >= 5:
                share = (top.values / tot.clamp(min=1e-6))[ok_items]
                agree = (pc.argmax(1)[ok_items] == top.indices[ok_items]).float()
                kappa = float((agree * share).sum() / share.sum())
        info['kappa'] = kappa
        # compatibility gain on the receiver's labelled buffer
        gain = None
        bidx, by = self.buf_scope(n, cap['key'])
        if c['use_gain'] and len(bidx) >= 8:
            l0 = F.cross_entropy(self.predict(n, bidx), by).item()
            l1 = F.cross_entropy(self.predict(n, bidx, cap), by).item()
            gain = max(-1.0, min(1.0, (l0 - l1) / (l0 + 1e-6)))
        info['gain'] = gain
        rep = n.rep.get(cap['src'], [1.0, 1.0])
        rho = rep[0] / (rep[0] + rep[1])
        sc = c['w_b']
        if gain is not None: sc += c['w_g'] * gain
        if kappa is not None: sc += c['w_k'] * 2 * (kappa - c['kappa0'])
        if c['use_unc']: sc -= c['w_u'] * max(0.0, info['mi'] - c['mi_ref']) / c['mi_ref'] * 4
        if c['use_rep']: sc += c['w_rho'] * (2 * rho - 1)
        T = 1 / (1 + math.exp(-sc)); info['T'] = T
        ok = (T >= c['tau_acc']) if c['use_gate'] else True
        # quarantine: no behavioural or data evidence and an unproven source -> defer
        info['defer'] = bool(c['use_gate'] and c['use_cons'] and ok and kappa is None and gain is None and rho < c['rho_q'])
        if info['defer']:
            return False, info
        # reputation evidence
        if c['use_rep']:
            good = (gain is not None and gain > 0.02) or (kappa is not None and kappa >= c['kappa0'])
            bad = ((gain is not None and gain < -0.02) or (kappa is not None and kappa < c['kappa0'] - 0.15) or
                   (c['use_unc'] and info['mi'] > c['mi_ref'] * 1.2))
            good = good and not bad
            rep = [rep[0] + (1 if good else 0), rep[1] + (1 if bad else 0)]
            n.rep[cap['src']] = rep
        return ok, info

    def spectral_prune(self, n, cap, bidx, by):
        """First-order transfer gain of every rank-1 component; drop contradictory ones."""
        c = self.c
        if len(bidx) < 8:
            return cap['mask'], (0, 0, sum(int(cap['mask'][m].sum()) for m in MATS))
        g = self.cap_gate_vec(n, cap, bidx).detach()
        probe = {m: (torch.zeros(DIMS[m], requires_grad=True), g) for m in MATS}
        lg = self.fwd(n, bidx, probe=probe)
        F.cross_entropy(lg, by).backward()
        gains = {}
        for m in MATS:
            G = probe[m][0].grad                                     # (in,out)
            gains[m] = -((cap['P'][m].T @ G) * cap['Q'][m]).sum(1)  # per-component  -<G, p_k q_k^T>
        tot = sum(float(gains[m].abs().sum()) for m in MATS) + 1e-12
        mask, nc, nn, nk = {}, 0, 0, 0
        for m in MATS:
            gam = gains[m] / tot
            keep = (gam >= -c['tau_gamma']).float() * cap['mask'][m]
            mask[m] = keep
            nc += int(((gam < -c['tau_gamma']) & (cap['mask'][m] > 0)).sum())
            nk += int(((gam > c['tau_gamma']) & (cap['mask'][m] > 0)).sum())
            nn += int(((gam.abs() <= c['tau_gamma']) & (cap['mask'][m] > 0)).sum())
        return mask, (nk, nc, nn)

    def process_inbox(self, n, t):
        c = self.c
        due = [x for x in n.inbox if x[0] <= t]; n.inbox = [x for x in n.inbox if x[0] > t]
        latest = {}
        for _, cap in due:
            k0 = (cap['src'], cap['regime'])
            if k0 not in n.pool or cap['version'] >= n.pool[k0]['version']:
                n.pool[k0] = cap
            self.comm[n.id] += 4 * (sum(cap['P'][m].numel() + cap['Q'][m].numel() for m in MATS) + 32) + 256
            k = (cap['src'], cap['regime'])
            if k not in latest or cap['version'] > latest[k]['version']:
                latest[k] = cap
        honest_rx = n.kind == 'honest'
        for cap in latest.values():
            ok, info = self.gate(n, cap, t)
            harmful = cap['harmful']
            if honest_rx and harmful:
                self.harm[1] += 1
            if not ok and info.get('defer'):
                if honest_rx and harmful:
                    self.harm[1] -= 1
                n.inbox.append((t + 1, cap))
                continue
            if not ok:
                self.log['events'].append(('reject', t, n.id, cap['id'], harmful, info.get('T'), self.nodes[cap['src']].kind, info))
                continue
            rec = dict(cap); rec['mask'] = {m: cap['mask'][m].clone() for m in MATS}
            if c['use_spectral']:
                bidx, by = self.buf_scope(n, cap['key'])
                rec['mask'], cls = self.spectral_prune(n, rec, bidx, by)
                if sum(float(rec['mask'][m].sum()) for m in MATS) == 0:
                    continue
            if honest_rx and harmful:
                self.harm[0] += 1
            # snapshot of the working adapter for provenance-based rollback
            n.snaps.append((t, cap['id'], self.lora_vec(n.lora)))
            n.lib.caps = [x for x in n.lib.caps if not (x['src'] == cap['src'] and x['regime'] == cap['regime'] and not x['own'])]
            n.lib.caps.append(rec); n.lib.invalidate()
            self.log['events'].append(('accept', t, n.id, cap['id'], harmful, info.get('T'), self.nodes[cap['src']].kind, info))

    def revalidate(self, n, t, dn):
        """Re-score foreign capsules active on the current stream; evict contradicting ones."""
        c = self.c
        if not c['revalidate'] or not n.lib.caps:
            return
        b = n.buf.get(dn, [])
        if len(b) < 16:
            return
        idx = np.array([i for i, _ in b]); y = torch.tensor([l for _, l in b])
        keep = []
        changed = False
        for cap in n.lib.caps:
            if cap['own']:
                keep.append(cap); continue
            s = self.d.Z[idx] @ cap['key']
            inn = (s >= c['tau_r']).numpy()
            if inn.sum() < 8:
                keep.append(cap); continue
            others = Library(); others.caps = [x for x in n.lib.caps if x is not cap]
            with torch.no_grad():
                lw = F.cross_entropy(self.fwd(n, idx[inn]), y[torch.tensor(inn)]).item()
                saved = n.lib; n.lib = others
                lo = F.cross_entropy(self.fwd(n, idx[inn]), y[torch.tensor(inn)]).item()
                n.lib = saved
            if lw > lo * 1.02:
                changed = True
                rep = n.rep.get(cap['src'], [1.0, 1.0]); rep[1] += 1; n.rep[cap['src']] = rep
                self.log['events'].append(('evict', t, n.id, cap['id'], cap['harmful']))
                continue
            if c['use_spectral']:
                n.lib.caps = others.caps + [cap]; n.lib.invalidate()
                cap['mask'], _ = self.spectral_prune_existing(n, cap, idx[inn], y[torch.tensor(inn)])
            keep.append(cap)
        n.lib.caps = keep; n.lib.invalidate()

    def spectral_prune_existing(self, n, cap, bidx, by):
        others = [x for x in n.lib.caps if x is not cap]
        saved = n.lib.caps; n.lib.caps = others; n.lib.invalidate()
        try:
            mask, cls = self.spectral_prune(n, cap, bidx, by)
        finally:
            n.lib.caps = saved; n.lib.invalidate()
        return mask, cls

    def track_parents(self, n, dn):
        """Foreign capsules active on the current data become provenance parents."""
        if not n.lib.caps or n.zcnt == 0:
            return
        z = F.normalize(n.zsum / n.zcnt, dim=0)
        for cap in n.lib.caps:
            if not cap['own'] and float(z @ cap['key']) >= self.c['tau_r']:
                n.active_parents.add(cap['id'])

    def contamination(self, cid, bad, memo):
        """Dense contaminated share of capsule cid: its own matrix if revoked, otherwise the
        provenance-weighted contaminated share inherited through recorded mixing weights."""
        if cid in memo:
            return memo[cid]
        cp = self.cap_store.get(cid)
        out = None
        if cp is not None:
            if cid in bad:
                out = {m: cp['P'][m] @ cp['Q'][m] for m in MATS}
            else:
                for pid, (w, pmask) in cp.get('mix', {}).items():
                    sub = self.contamination(pid, bad, memo)
                    if sub is not None:
                        out = {m: (0 if out is None else out[m]) + w * sub[m] for m in MATS}
        memo[cid] = out
        return out

    def rollback(self, t, revoked):
        mode = self.c['rollback']
        bad = {cid for cid, cp in self.cap_store.items() if cp['src'] in revoked and cp['t'] >= self.c['sleeper_on']}
        desc = {cid for cid, cp in self.cap_store.items() if cp['anc'] & bad} - bad
        removed = repaired = 0
        memo = {}
        for n in self.nodes:
            if n.kind != 'honest':
                continue
            keep = []
            for x in n.lib.caps:
                if x['id'] in bad:
                    removed += 1; continue
                if x['id'] in desc and mode == 'lineage':
                    removed += 1; continue
                if x['id'] in desc and mode == 'repair':
                    sub = self.contamination(x['id'], bad, memo)
                    if sub is not None:
                        rk = self.c['rank_c']; y = dict(x); y['P'], y['Q'] = {}, {}
                        for m in MATS:
                            Dm = (x['P'][m] * x['mask'][m][None]) @ x['Q'][m] - sub[m]
                            U, S, V = torch.svd_lowrank(Dm, q=rk + 4, niter=3)
                            y['P'][m] = (U[:, :rk] * S[:rk]).contiguous(); y['Q'][m] = V[:, :rk].T.contiguous()
                        y['mask'] = {m: torch.ones(rk) for m in MATS}; y['anc'] = x['anc'] - bad
                        x = y; repaired += 1
                keep.append(x)
            n.lib.caps = keep; n.lib.invalidate()
            n.active_parents -= bad
            n.inbox = [x for x in n.inbox if x[1]['id'] not in bad]
            n.pool = {k: v for k, v in n.pool.items() if v['id'] not in bad}
            n.rep.update({s: [1.0, 50.0] for s in revoked})
        return dict(removed=removed, repaired=repaired, bad=len(bad), desc=len(desc))

    # ------------------------------------------------------------------ exchange for baselines
    def exchange(self, active, t):
        c = self.c; meth = c['method']
        if meth in ('fedavg', 'fedprox', 'ffa', 'median', 'ditto'):
            subs = [self.submitted(n, n.glob if meth == 'ditto' else n.lora) for n in active]
            if not subs:
                return
            for m in MATS:
                for k in range(2):
                    if meth == 'ffa' and k == 0:
                        continue
                    st = torch.stack([s[m][k] for s in subs])
                    self.server[m][k] = st.median(0).values if meth == 'median' else st.mean(0)
            per = sum(self.server[m][k].numel() for m in MATS for k in range(2) if not (meth == 'ffa' and k == 0))
            for n in active:
                self.comm[n.id] += 2 * 4 * per
                self.load_global(n)
        elif meth == 'gossip':
            vecs = {n.id: self.submitted(n, n.lora) for n in active}
            for n in active:
                if n.kind != 'honest':
                    continue
                nb = [o for o in active if o.id != n.id]
                if not nb:
                    continue
                pick = self.rng.choice(len(nb), min(2, len(nb)), replace=False)
                grp = [vecs[n.id]] + [vecs[nb[p].id] for p in pick]
                with torch.no_grad():
                    for m in MATS:
                        for k in range(2):
                            n.lora[m][k].copy_(torch.stack([g[m][k] for g in grp]).mean(0))
                per = sum(n.lora[m][k].numel() for m in MATS for k in range(2))
                self.comm[n.id] += 2 * len(pick) * 4 * per
        elif meth == 'ties' and t % 3 == 2:
            deltas = {}
            for n in self.nodes:
                v = self.submitted(n, n.lora)
                deltas[n.id] = {m: (v[m][0].T @ v[m][1].T) * c['scale'] for m in MATS}
            per = sum(n.lora[m][k].numel() for m in MATS for k in range(2))
            for n in active:
                if n.kind != 'honest':
                    continue
                peers = [deltas[o.id] for o in self.nodes if o.id != n.id and o.seen]
                if not peers:
                    continue
                n.dense = {}
                for m in MATS:
                    st = torch.stack([p[m] for p in peers])
                    flat = st.abs().flatten(1)
                    k = max(1, int(c['ties_k'] * flat.shape[1]))
                    thr = flat.kthvalue(flat.shape[1] - k + 1, dim=1).values
                    trimmed = st * (st.abs() >= thr[:, None, None]).float()
                    sign = torch.sign(trimmed.sum(0))
                    agree = (torch.sign(trimmed) == sign[None]).float() * (trimmed != 0).float()
                    n.dense[m] = (trimmed * agree).sum(0) / agree.sum(0).clamp(min=1)
                self.comm[n.id] += 4 * per * len(peers) / 3 + 4 * per / 3

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, t, final=False):
        res = {}
        for n in self.nodes:
            if n.kind != 'honest' or (t >= 0 and t < n.start) or (t < 0 and n.start > 0):
                continue
            r = {}
            for dn in DOMAINS:
                idx = self.test[dn]
                with torch.no_grad():
                    lg = self.fwd(n, idx)
                y = eval_labels(self.c, n.policy, self.d.y[idx])
                pr = lg.softmax(1)
                r[dn] = dict(acc=float((lg.argmax(1) == y).float().mean()), pred=lg.argmax(1).numpy(),
                             ece=ece(pr, y), nll=float(F.cross_entropy(lg, y)))
                n.hist[dn].append((t, r[dn]['acc']))
            if final:
                with torch.no_grad():
                    gw = None
                    lgx = forward(self.bb, self.d.Hx, self.d.Ox, self.d.Zx, lora=n.lora if self.c['method'] != 'cose' else None,
                                  lib=n.lib, lib_cfg=self.lib_cfg, scale=self.c['scale'], dense=n.dense)
                    if self.c['method'] == 'cose' and n.zcnt > 0:
                        wkey = F.normalize(n.zsum / n.zcnt, dim=0)
                        gwx = (self.d.Zx @ wkey >= self.c['tau_r']).float()
                        lgx = forward(self.bb, self.d.Hx, self.d.Ox, self.d.Zx, lora=n.lora, lib=n.lib,
                                      lib_cfg=self.lib_cfg, scale=self.c['scale'], gw=gwx)
                r['_ext'] = dict(acc=float((lgx.argmax(1) == self.d.yx).float().mean()))
                pr = []
                for dn in DOMAINS:
                    idx = self.test[dn]
                    with torch.no_grad():
                        pr.append(self.fwd(n, idx).softmax(1))
                pr = torch.cat(pr); yy = torch.cat([eval_labels(self.c, n.policy, self.d.y[self.test[dn]]) for dn in DOMAINS])
                conf, pred = pr.max(1)
                r['_rel'] = [(float(conf[(conf > a) & (conf <= a + 0.1)].sum()), float((pred == yy)[(conf > a) & (conf <= a + 0.1)].float().sum()),
                              int(((conf > a) & (conf <= a + 0.1)).sum())) for a in np.arange(0, 1, 0.1)]
                # policy-conflict classes (AG Sci/Tech, R8 money-fx): accuracy under the node's own policy
                ci = np.concatenate([self.test['ag'][(self.d.y[self.test['ag']] == 3).numpy()],
                                     self.test['r8'][(self.d.y[self.test['r8']] == 8).numpy()]])
                with torch.no_grad():
                    lc = self.fwd(n, ci)
                r['_conf'] = float((lc.argmax(1) == eval_labels(self.c, n.policy, self.d.y[ci])).float().mean())
            res[n.id] = r
        return res

    # ------------------------------------------------------------------ main loop
    def run(self):
        c = self.c; t0 = time.time()
        res = self.evaluate(-1)
        self.log['zero'] = {dn: float(np.mean([res[i][dn]['acc'] for i in res])) for dn in DOMAINS}
        snapshot = None; rb = {}
        for t in range(c['T']):
            for n in self.nodes:
                n.t_now = t
            if c['sleeper'] and t == c['sleeper_on']:
                snapshot = [(copy.deepcopy(n.lib.caps), self.lora_vec(n.lora)) for n in self.nodes]
            active = [n for n in self.nodes if t >= n.start and self.rng.rand() < c['p_act']]
            if c['joiner'] and t == c['join_at']:
                j = self.nodes[-1]
                if j not in active:
                    active.append(j)
                if c['method'] == 'cose':
                    seen = {}
                    for cid, cp in self.cap_store.items():
                        k = (cp['src'], cp['regime'])
                        if k not in seen or cp['version'] > seen[k]['version']:
                            seen[k] = cp
                    j.inbox = [(t, cp) for cp in seen.values()]
                    self.process_inbox(j, t)
                elif c['method'] in ('fedavg', 'fedprox', 'ffa', 'median', 'ditto'):
                    self.load_global(j)
                    if c['method'] == 'ditto':
                        with torch.no_grad():
                            for m in MATS:
                                for k in range(2):
                                    j.lora[m][k].copy_(j.glob[m][k])
                self.log['join0'] = {dn: v['acc'] for dn, v in self.evaluate(t)[j.id].items()}
            for n in active:
                if c['method'] == 'cose' and n.kind in ('honest', 'sleeper'):
                    self.process_inbox(n, t)
                dn = self.local_round(n, t)
                if c['method'] == 'cose':
                    self.track_parents(n, dn)
                    if n.kind == 'honest' or n.kind == 'sleeper':
                        self.revalidate(n, t, dn)
                    self.maybe_emit(n, t)
            if c['method'] != 'cose':
                self.exchange(active, t)
            if c['sleeper'] and t == c['audit']:
                pre = self.evaluate(t)
                revoked = {n.id for n in self.nodes if n.kind == 'sleeper'}
                if c['rollback'] == 'reset' and snapshot is not None:
                    for n, (caps, lv) in zip(self.nodes, snapshot):
                        if n.kind == 'honest':
                            n.lib.caps = caps; n.lib.invalidate()
                            with torch.no_grad():
                                for m in MATS:
                                    n.lora[m][0].copy_(lv[m][0]); n.lora[m][1].copy_(lv[m][1])
                        n.rep.update({s: [1.0, 50.0] for s in revoked})
                    removed = -1
                elif c['rollback'] in ('lineage', 'direct', 'repair'):
                    removed = self.rollback(t, revoked)
                else:
                    removed = 0
                post = self.evaluate(t)
                rb = dict(pre=self.summ(pre), post=self.summ(post), removed=removed)
            if t % c['eval_every'] == c['eval_every'] - 1 or t == c['T'] - 1:
                res = self.evaluate(t, final=(t == c['T'] - 1))
                self.log['curve'].append((t, self.summ(res)['acc']))
        self.final = res; self.rb = rb
        self.log['time'] = time.time() - t0
        return self.metrics()

    def summ(self, res):
        return dict(acc=float(np.mean([np.mean([r[dn]['acc'] for dn in DOMAINS]) for r in res.values()])))

    def metrics(self):
        c = self.c; res = self.final
        hon = [n for n in self.nodes if n.kind == 'honest' and n.start == 0]
        acc, seen, unseen, eces, nlls, fgt, ft, lat = [], [], [], [], [], [], [], []
        dom_acc = {dn: [] for dn in DOMAINS}
        for n in hon:
            r = res[n.id]
            acc.append(np.mean([r[dn]['acc'] for dn in DOMAINS]))
            seen.append(np.mean([r[dn]['acc'] for dn in n.sched]))
            us = [r[dn]['acc'] for dn in DOMAINS if dn not in n.sched]
            unseen.append(np.mean(us))
            eces.append(np.mean([r[dn]['ece'] for dn in DOMAINS])); nlls.append(np.mean([r[dn]['nll'] for dn in DOMAINS]))
            for dn in DOMAINS:
                dom_acc[dn].append(r[dn]['acc'])
            n.final_acc = acc[-1]
            for k, dn in enumerate(n.sched[:2]):
                h2 = [a for (tt, a) in n.hist[dn] if tt >= k * c['seg']]
                if h2:
                    fgt.append(max(h2) - r[dn]['acc'])
            segs = {}
            for (t, dn, pa, pl, new) in n.preq:
                segs.setdefault(dn, []).append((t, pa, new))
            for k, dn in enumerate(n.sched):
                s = segs.get(dn, [])
                if not s:
                    continue
                if k > 0:
                    ft.append(s[0][1] - self.log['zero'][dn])
                accs = [a for _, a, _ in s]
                if len(accs) >= 4:
                    tgt = 0.9 * np.mean(accs[-3:])
                    sm = np.convolve(accs, np.ones(2) / 2, mode='valid')
                    hit = np.where(sm >= tgt)[0]
                    lat.append(int(hit[0]) + 1 if len(hit) else len(accs))
        # semantic disagreement inside each policy cluster
        dis = []
        for pol in 'AB':
            ids = [n.id for n in hon if n.policy == pol]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    dis.append(np.mean([np.mean(res[ids[a]][dn]['pred'] != res[ids[b]][dn]['pred']) for dn in DOMAINS]))
        T = c['T']; nact = len(self.nodes)
        ext = float(np.mean([res[n.id]['_ext']['acc'] for n in hon]))
        rel = np.array([res[n.id]['_rel'] for n in hon]).sum(0).tolist()
        confA = float(np.mean([res[n.id]['_conf'] for n in hon if n.policy == 'A']))
        confB = float(np.mean([res[n.id]['_conf'] for n in hon if n.policy == 'B']))
        out = dict(ext=ext, rel=rel, confA=confA, confB=confB, acc=float(np.mean(acc)), seen=float(np.mean(seen)), unseen=float(np.mean(unseen)),
                   ece=float(np.mean(eces)), nll=float(np.mean(nlls)), forget=float(np.mean(fgt)) if fgt else 0.0,
                   ft=float(np.mean(ft)) if ft else 0.0, latency=float(np.mean(lat)) if lat else 0.0,
                   disagree=float(np.mean(dis)) if dis else 0.0,
                   comm_kb=float(self.comm[[n.id for n in hon]].mean() / T / 1024),
                   dom={dn: float(np.mean(v)) for dn, v in dom_acc.items()},
                   per_node=[float(a) for a in acc], curve=self.log['curve'], zero=self.log['zero'],
                   time=self.log['time'])
        if c['method'] == 'cose':
            ev = self.log['events']
            out['huar'] = self.harm[0] / max(self.harm[1], 1)
            out['n_accept'] = sum(1 for e in ev if e[0] == 'accept')
            out['n_reject'] = sum(1 for e in ev if e[0] == 'reject')
            out['n_evict'] = sum(1 for e in ev if e[0] == 'evict')
            out['n_drift'] = sum(1 for e in ev if e[0] == 'drift')
            out['drift_true'] = sum(1 for e in ev if e[0] == 'drift' and e[4] and e[5] == 'honest')
            out['drift_honest'] = sum(1 for e in ev if e[0] == 'drift' and e[5] == 'honest')
            out['lib_size'] = float(np.mean([len(n.lib.caps) for n in hon]))
            out['T_scores'] = [(e[0], e[4], e[5]) for e in ev if e[0] in ('accept', 'reject')]
            out['T_kind'] = [(e[6], e[0], round(e[5], 4) if e[5] is not None else None,
                              None if e[7].get('kappa') is None else round(e[7]['kappa'], 3),
                              round(e[7].get('mi', 0.0), 4),
                              None if e[7].get('gain') is None else round(e[7]['gain'], 3))
                             for e in ev if e[0] in ('accept', 'reject') and self.nodes[e[2]].kind == 'honest']
            kinds = {}
            for e in ev:
                if e[0] in ('accept', 'reject') and self.nodes[e[2]].kind == 'honest':
                    kk = kinds.setdefault(e[6], [0, 0]); kk[0 if e[0] == 'accept' else 1] += 1
            out['by_kind'] = kinds
        if self.rb:
            out['rollback'] = self.rb
        if 'join0' in self.log:
            out['join0'] = self.log['join0']
            out['join_final'] = {dn: res[self.nodes[-1].id][dn]['acc'] for dn in DOMAINS}
        return out


def run(cfg):
    data = Data(extra=True); bb = Backbone()
    return Sim(cfg, data, bb).run()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument('--' + k, type=type(v) if not isinstance(v, tuple) else str, default=v)
    a = vars(ap.parse_args())
    if isinstance(a['steps'], str):
        a['steps'] = tuple(int(x) for x in a['steps'].strip('()').split(',') if x.strip())
    r = run(a)
    print(json.dumps({k: v for k, v in r.items() if k not in ('per_node', 'curve', 'T_scores')}, indent=1))
    if a['out']:
        json.dump(dict(cfg=a, res=r), open(a['out'], 'w'), default=str)
