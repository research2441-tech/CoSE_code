"""Pilot check: LoRA on the read-out block trained centrally on a list of domains
(e.g. all six, or one) and evaluated on every domain.  Usage: python central.py ag,r8 600 3e-3"""
import numpy as np, torch, torch.nn.functional as F, sys
from cose_core import *
bb=Backbone(); d=Data(); g=torch.Generator().manual_seed(0)
doms=sys.argv[1].split(',') if len(sys.argv)>1 else DOMAINS
steps=int(sys.argv[2]) if len(sys.argv)>2 else 600
lr=float(sys.argv[3]) if len(sys.argv)>3 else 2e-3
L=zero_lora(8,g); opt=torch.optim.Adam([p for m in MATS for p in L[m]],lr=lr)
tr=np.concatenate([d.idx[(x,'train')] for x in doms]); rng=np.random.RandomState(0)
for s in range(steps):
    b=rng.choice(tr,16); lg=forward(bb,d.H[b],d.O[b],d.Z[b],lora=L)
    loss=F.cross_entropy(lg,d.y[b]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    for x in DOMAINS:
        i=d.idx[(x,'test')]; lg=forward(bb,d.H[i],d.O[i],d.Z[i],lora=L)
        print(x, round(float((lg.argmax(1)==d.y[i]).float().mean()),3))
