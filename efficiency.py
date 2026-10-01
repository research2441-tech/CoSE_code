"""Measured costs of CoSE operations on the experiment host (single thread)."""
import json, time, numpy as np, torch
import torch.nn.functional as F
import cose_sim as cs
from cose_core import Backbone, Data, Library, MATS, DIMS, forward, zero_lora, lora_to_factors

torch.set_num_threads(1)
bb = Backbone(); d = Data()
g = torch.Generator().manual_seed(0)
out = {}


def timeit(fn, rep=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep


idx = np.arange(256)
H, O, Z = d.H[idx], d.O[idx], d.Z[idx]
lora = zero_lora(8, g)
with torch.no_grad():
    for m in MATS:
        lora[m][1].normal_(0, 0.02)
# inference latency vs library size
for C in [0, 10, 20, 40]:
    lib = Library()
    for k in range(C):
        P, Q, _ = lora_to_factors(lora)
        lib.caps.append(dict(key=F.normalize(torch.randn(32), dim=0), P=P, Q=Q, mask={m: torch.ones(8) for m in MATS}))
    cfg = dict(tau_r=0.15, temp=0.05, routed=True)
    with torch.no_grad():
        t = timeit(lambda: forward(bb, H, O, Z, lora=lora, lib=lib if C else None, lib_cfg=cfg))
    out[f'infer_ms_per_sample_C{C}'] = 1000 * t / len(idx)
# local training step (batch 32)
opt = torch.optim.Adam([p for m in MATS for p in lora[m]], lr=3e-3)
b = idx[:32]; y = d.y[b]
def step():
    lg = forward(bb, d.H[b], d.O[b], d.Z[b], lora=lora); loss = F.cross_entropy(lg, y)
    opt.zero_grad(); loss.backward(); opt.step()
out['train_step_ms'] = 1000 * timeit(step)
# capsule encoding (three randomised SVDs of dense effective deltas)
D = {m: (lora[m][0].detach().T @ lora[m][1].detach().T) * 2.0 for m in MATS}
def enc():
    for m in MATS:
        U, S, V = torch.svd_lowrank(D[m], q=12, niter=3)
out['encode_ms'] = 1000 * timeit(enc, 10)
out['capsule_kb_fp32'] = 4 * (sum(8 * (DIMS[m][0] + DIMS[m][1]) for m in MATS) + 32) / 1024
out['lora_params'] = sum(8 * (DIMS[m][0] + DIMS[m][1]) for m in MATS)
# admission of one capsule at a receiver with a 10-capsule library
cfg = dict(cs.DEF); cfg['T'] = 1
sim = cs.Sim(cfg, d, bb)
n = sim.nodes[0]
P, Q, _ = lora_to_factors(lora)
for k in range(10):
    n.lib.caps.append(dict(id=f'x{k}', src=1, regime=k, own=False, key=F.normalize(torch.randn(32), dim=0), P=P, Q=Q,
                           mask={m: torch.ones(8) for m in MATS}))
dn = 'ag'; bi = d.idx[(dn, 'train')][:48]
n.buf[dn] = list(zip(bi.tolist(), d.y[bi].tolist())); n.cur_dom = dn
key = F.normalize(d.Z[bi].mean(0), dim=0)
cap = dict(id='new', src=2, regime=0, version=0, t=0, key=key, P=P, Q=Q, mask={m: torch.ones(8) for m in MATS},
           parents=set(), anc=set(), harmful=False)
out['gate_ms'] = 1000 * timeit(lambda: sim.gate(n, cap, 0), 5)
bidx, by = sim.buf_scope(n, key)
out['spectral_ms'] = 1000 * timeit(lambda: sim.spectral_prune(n, cap, bidx, by), 5)
json.dump(out, open('efficiency.json', 'w'), indent=1)
print(out)
