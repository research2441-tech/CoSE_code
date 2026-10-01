"""Aggregate runs into the numbers used by tables, figures and text (results.json)."""
import json, glob, collections, numpy as np
from scipy.stats import wilcoxon

R = collections.defaultdict(dict)
for f in glob.glob('runs/E*.json'):
    name = f.split('/')[-1][:-5]
    base, s = name.rsplit('_s', 1)
    R[base][int(s)] = json.load(open(f))['res']

KEYS = ['acc', 'seen', 'unseen', 'forget', 'ft', 'ece', 'nll', 'disagree', 'latency', 'comm_kb', 'ext', 'confA', 'confB']
rng = np.random.default_rng(0)


def boot(x, B=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float('nan')] * 2
    m = rng.choice(x, (B, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summ(base):
    runs = R.get(base, {})
    out = {'n': len(runs), 'seeds': sorted(runs)}
    for k in KEYS + ['huar', 'n_accept', 'n_reject', 'n_evict', 'n_drift', 'drift_true', 'drift_honest', 'lib_size', 'time']:
        v = [r[k] for r in runs.values() if k in r and r[k] is not None]
        if v:
            out[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, ci=boot(v), vals=[float(x) for x in v])
    if runs:
        out['dom'] = {d: float(np.mean([r['dom'][d] for r in runs.values()])) for d in ['ag', 'r8', 'trec', 'mr', 'cr', 'tweet']}
        cur = [r['curve'] for r in runs.values()]
        L = min(len(c) for c in cur)
        arr = np.array([[a for _, a in c[:L]] for c in cur])
        out['curve'] = dict(t=[t for t, _ in cur[0][:L]], mean=arr.mean(0).tolist(),
                            lo=np.percentile(arr, 2.5, 0).tolist() if len(cur) > 1 else arr.mean(0).tolist(),
                            hi=np.percentile(arr, 97.5, 0).tolist() if len(cur) > 1 else arr.mean(0).tolist())
        rel = np.array([r['rel'] for r in runs.values()]).sum(0)
        out['rel'] = rel.tolist()
        if all('by_kind' in r for r in runs.values()):
            bk = collections.defaultdict(lambda: [0, 0])
            for r in runs.values():
                for k, (a, b) in r['by_kind'].items():
                    bk[k][0] += a; bk[k][1] += b
            out['by_kind'] = dict(bk)
        if all('rollback' in r for r in runs.values()):
            out['rb_pre'] = float(np.mean([r['rollback']['pre']['acc'] for r in runs.values()]))
            out['rb_post'] = float(np.mean([r['rollback']['post']['acc'] for r in runs.values()]))
            out['rb_delta'] = dict(mean=float(np.mean([r['rollback']['post']['acc'] - r['rollback']['pre']['acc'] for r in runs.values()])),
                                   sd=float(np.std([r['rollback']['post']['acc'] - r['rollback']['pre']['acc'] for r in runs.values()], ddof=1)) if len(runs) > 1 else 0)
            rm = [r['rollback']['removed'] for r in runs.values()]
            if isinstance(rm[0], dict):
                out['rb_removed'] = float(np.mean([x['removed'] for x in rm])); out['rb_repaired'] = float(np.mean([x['repaired'] for x in rm]))
                out['rb_bad'] = float(np.mean([x['bad'] for x in rm])); out['rb_desc'] = float(np.mean([x['desc'] for x in rm]))
        if all('join0' in r for r in runs.values()):
            out['join0'] = float(np.mean([np.mean(list(r['join0'].values())) for r in runs.values()]))
            out['join_final'] = float(np.mean([np.mean(list(r['join_final'].values())) for r in runs.values()]))
            out['join0_vals'] = [float(np.mean(list(r['join0'].values()))) for r in runs.values()]
    return out


def paired(a, b, key='acc'):
    A, B = R.get(a, {}), R.get(b, {})
    s = sorted(set(A) & set(B))
    x = np.array([A[i][key] for i in s]); y = np.array([B[i][key] for i in s])
    if len(s) < 5:
        return None
    d = x - y
    try:
        st = wilcoxon(x, y, alternative='two-sided', zero_method='wilcox')
        p = float(st.pvalue)
    except ValueError:
        p = 1.0
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    rpos = ranks[nz > 0].sum(); rneg = ranks[nz < 0].sum(); tot = ranks.sum()
    rbc = float((rpos - rneg) / tot) if tot else 0.0
    return dict(n=len(s), diff=float(d.mean()), ci=boot(d), p=p, rbc=rbc, wins=int((d > 0).sum()))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


if __name__ == '__main__':
    out = {b: summ(b) for b in R}
    M = ['local', 'fedavg', 'fedprox', 'ffa', 'ditto', 'gossip', 'median', 'ties']
    tests = {}
    for key in ['acc', 'unseen', 'forget', 'ece']:
        rows = [(m, paired('E1_cose', f'E1_{m}', key)) for m in M]
        rows = [(m, r) for m, r in rows if r]
        if rows:
            adj = holm([r['p'] for _, r in rows])
            for (m, r), a in zip(rows, adj):
                r['p_holm'] = a
            tests[key] = dict(rows)
    rows = [(m, paired('E1b_cose', f'E1b_{m}', 'acc')) for m in M]
    rows = [(m, r) for m, r in rows if r]
    if rows:
        adj = holm([r['p'] for _, r in rows])
        for (m, r), a in zip(rows, adj):
            r['p_holm'] = a
        tests['clean_acc'] = dict(rows)
    out['_tests'] = tests
    # trust-score distributions by sender kind (main runs)
    ts = collections.defaultdict(list)
    for s, r in R.get('E1_cose', {}).items():
        pass
    json.dump(out, open('results.json', 'w'), indent=1)
    for b in sorted(out):
        if b.startswith('_'):
            continue
        o = out[b]
        print('%-24s n=%2d acc %.3f±%.3f unseen %.3f fgt %.3f ece %.3f' % (b, o['n'], o['acc']['mean'], o['acc']['sd'], o['unseen']['mean'], o['forget']['mean'], o['ece']['mean']))
    for k, v in tests.items():
        print(k, {m: (round(r['diff'], 3), round(r['p_holm'], 4)) for m, r in v.items()})
