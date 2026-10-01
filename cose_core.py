"""Core numerics for CoSE: frozen GPT-2 read-out block with LoRA + routed capsule library.

Only the final transformer block (index 11) of GPT-2 small is adapted.  Its attention
output at the read-out token is cached (query/key/value frozen), so the adapted path is
   a  = o W_o + b_o + dW_o(o)                       (attention output projection)
   h2 = h + a
   f  = gelu(LN2(h2) W_fc + b_fc + dW_fc(.))         (MLP up projection)
   h3 = h2 + f W_pr + b_pr + dW_pr(f)                (MLP down projection)
   y  = LN_f(h3) E^T                                 (verbalizer rows of the tied embedding)
"""
import json, math, os
import numpy as np, torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get('COSE_THREADS', '2')))
MATS = ['o', 'fc', 'pr']
DIMS = {'o': (768, 768), 'fc': (768, 3072), 'pr': (3072, 768)}
VERB = [' world', ' sports', ' business', ' technology', ' earnings', ' acquisition', ' oil', ' trade',
        ' currency', ' interest', ' shipping', ' grain', ' description', ' entity', ' abbreviation',
        ' person', ' location', ' number', ' negative', ' positive']
DOMAINS = ['ag', 'r8', 'trec', 'mr', 'cr', 'tweet']
DOM_LABELS = {'ag': [0, 1, 2, 3], 'r8': list(range(4, 12)), 'trec': list(range(12, 18)),
              'mr': [18, 19], 'cr': [18, 19], 'tweet': [18, 19]}


class Backbone:
    """Frozen parameters of block 11, ln_f and the verbalizer."""

    def __init__(self):
        sd = torch.load(os.path.join(W, 'models/gpt2_small.pt'))
        p = lambda k: sd[k].float()
        self.Wo, self.bo = p('h.11.attn.c_proj.weight'), p('h.11.attn.c_proj.bias')
        self.ln2w, self.ln2b = p('h.11.ln_2.weight'), p('h.11.ln_2.bias')
        self.Wfc, self.bfc = p('h.11.mlp.c_fc.weight'), p('h.11.mlp.c_fc.bias')
        self.Wpr, self.bpr = p('h.11.mlp.c_proj.weight'), p('h.11.mlp.c_proj.bias')
        self.lnfw, self.lnfb = p('ln_f.weight'), p('ln_f.bias')
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer(os.path.join(W, 'models/encoder.json'), os.path.join(W, 'models/vocab.bpe'))
        ids = [tok.encode(v)[0] for v in VERB]
        self.E = p('wte.weight')[ids].contiguous()        # (20, 768)


class Data:
    def __init__(self, extra=False):
        c = os.path.join(W, 'cache')
        self.H = torch.from_numpy(np.load(os.path.join(c, 'H.npy')))
        self.O = torch.from_numpy(np.load(os.path.join(c, 'O.npy')))
        Z = torch.from_numpy(np.load(os.path.join(c, 'Z.npy')))
        meta = json.load(open(os.path.join(c, 'meta.json')))
        self.dom = np.array(meta['domain']); self.split = np.array(meta['split'])
        self.y = torch.tensor(meta['label'])
        self.idx = {(d, s): np.where((self.dom == d) & (self.split == s))[0] for d in set(self.dom) for s in ['train', 'test']}
        # public calibration/anchor set: 40 unlabeled training texts per domain (fixed draw)
        rng = np.random.RandomState(0)
        self.anchor = np.concatenate([rng.choice(self.idx[(d, 'train')], 40, replace=False) for d in DOMAINS])
        # routing keys: anchor-standardised, PCA-whitened (32-d) mean-pooled block-11 inputs
        mu = Z[self.anchor].mean(0); sd = Z[self.anchor].std(0) + 1e-3
        X = (Z[self.anchor] - mu) / sd
        torch.manual_seed(0)
        U, S, V = torch.pca_lowrank(X, q=64, center=False)
        self.key_proj = (mu, sd, V[:, :32] / S[:32])
        self.Z = self.keys(Z)
        if extra:
            ce = os.path.join(W, 'cache_ext')
            self.Hx = torch.from_numpy(np.load(os.path.join(ce, 'H.npy')))
            self.Ox = torch.from_numpy(np.load(os.path.join(ce, 'O.npy')))
            self.Zx = self.keys(torch.from_numpy(np.load(os.path.join(ce, 'Z.npy'))))
            self.yx = torch.tensor(json.load(open(os.path.join(ce, 'meta.json')))['label'])

    def keys(self, Z):
        mu, sd, P = self.key_proj
        return F.normalize(((Z - mu) / sd) @ P, dim=1)


def zero_lora(r, gen):
    st = {}
    for m in MATS:
        i, o = DIMS[m]
        A = torch.randn(r, i, generator=gen) / math.sqrt(i)
        B = torch.zeros(o, r)
        st[m] = [A.requires_grad_(), B.requires_grad_()]
    return st


class Library:
    """Routed capsule components:  P (C,in,R), Q (C,R,out), component mask (C,R), keys (C,768)."""

    def __init__(self):
        self.caps = []          # list of dicts (capsule records incl. id, key, P/Q per matrix, mask)
        self._pack = None

    def invalidate(self):
        self._pack = None

    def pack(self):
        if self._pack is None and self.caps:
            keys = torch.stack([c['key'] for c in self.caps])
            P = {m: torch.stack([c['P'][m] for c in self.caps]) for m in MATS}
            Q = {m: torch.stack([c['Q'][m] * c['mask'][m][:, None] for c in self.caps]) for m in MATS}
            self._pack = (keys, P, Q)
        return self._pack


def gates(z, keys, tau_r, temp, routed=True):
    """Thresholded soft routing of inputs z (B,768) over capsule keys (C,768)."""
    s = z @ keys.T
    if not routed:
        return torch.full_like(s, 1.0 / keys.shape[0])
    w = torch.exp((s - 1.0) / temp) * (s >= tau_r).float()
    return w / w.sum(1, keepdim=True).clamp(min=1e-8) * (w.sum(1, keepdim=True) > 0).float()


def forward(bb, H, O, Z, lora=None, lib=None, lib_cfg=None, extra=None, probe=None, scale=2.0,
            dense=None, gw=None):
    """Return logits (B,20).  extra: list of (P,Q,g) additional routed capsules (for gate tests).
    probe: dict m->(Dm zeros leaf, gvec (B,)) placeholders for first-order gain gradients.
    dense: dict m->full delta matrix added for every input (used by merging baselines)."""
    def delta(m, x):
        out = 0.0
        if lora is not None:
            A, B = lora[m]
            lo = (x @ A.T) @ B.T * scale
            out = out + (lo if gw is None else lo * gw[:, None])
        if dense is not None:
            out = out + x @ dense[m]
        if lib is not None and lib.caps:
            keys, P, Q = lib.pack()
            g = gates(Z, keys, lib_cfg['tau_r'], lib_cfg['temp'], lib_cfg.get('routed', True))
            t = torch.einsum('bi,cir->bcr', x, P[m]) * g[:, :, None]
            out = out + torch.einsum('bcr,cro->bo', t, Q[m])
        if extra is not None:
            for (P1, Q1, g1) in extra:
                out = out + ((x @ P1[m]) * g1[:, None]) @ Q1[m]
        if probe is not None and m in probe:
            Dm, gv = probe[m]
            out = out + (x * gv[:, None]) @ Dm
        return out
    a = O @ bb.Wo + bb.bo + delta('o', O)
    h2 = H + a
    x = F.layer_norm(h2, (768,), bb.ln2w, bb.ln2b)
    u = x @ bb.Wfc + bb.bfc + delta('fc', x)
    f = F.gelu(u, approximate='tanh')
    h3 = h2 + f @ bb.Wpr + bb.bpr + delta('pr', f)
    return F.layer_norm(h3, (768,), bb.lnfw, bb.lnfb) @ bb.E.T


def lora_to_factors(lora, scale=2.0):
    """Exact thin SVD of each LoRA product: returns P=U*S (in,r), Q=V^T (r,out), S."""
    P, Q, S = {}, {}, {}
    for m in MATS:
        A, B = [t.detach() for t in lora[m]]
        q1, r1 = torch.linalg.qr(A.T)            # (in,r)
        q2, r2 = torch.linalg.qr(B)              # (out,r)
        u, s, vt = torch.linalg.svd(r1 @ r2.T * scale)
        P[m] = q1 @ u * s[None]                  # in x r   (columns scaled by singular values)
        Q[m] = vt @ q2.T                         # r x out  (orthonormal rows)
        S[m] = s
    return P, Q, S


def ece(prob, y, bins=15):
    conf, pred = prob.max(1)
    acc = (pred == y).float()
    e = torch.zeros(())
    edges = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        msk = (conf > edges[i]) & (conf <= edges[i + 1])
        if msk.any():
            e += msk.float().mean() * (conf[msk].mean() - acc[msk].mean()).abs()
    return float(e)
