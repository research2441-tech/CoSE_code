"""Run the frozen GPT-2 (124M) through blocks 0-10 once and cache what block 11 needs.

Block 11 is the only adapted block.  Its key/value projections stay frozen, so the
per-token keys and values are cached; the query, output projection and MLP are
LoRA-adapted and only evaluated at the final (read-out) token.
"""
import json, os, sys, time
import numpy as np, torch
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

torch.set_num_threads(2)
W = os.path.dirname(os.path.abspath(__file__))
L = 64
tok = GPT2Tokenizer(os.path.join(W, 'models/encoder.json'), os.path.join(W, 'models/vocab.bpe'))
model = GPT2Model(GPT2Config())
model.load_state_dict(torch.load(os.path.join(W, 'models/gpt2_small.pt')), strict=False)
model.eval()
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
rows = [json.loads(l) for l in open(os.path.join(W, 'data/pool%s.jsonl' % TAG))]
N = len(rows)
suffix = tok.encode('\nLabel:')
out = os.path.join(W, 'cache' + TAG); os.makedirs(out, exist_ok=True)
K = np.lib.format.open_memmap(os.path.join(out, 'K.npy'), 'w+', np.float16, (N, L, 768))
V = np.lib.format.open_memmap(os.path.join(out, 'V.npy'), 'w+', np.float16, (N, L, 768))
H = np.zeros((N, 768), np.float32); Z = np.zeros((N, 768), np.float32); M = np.zeros((N, L), np.bool_)
blk = model.h[11]
t0 = time.time(); bs = 32
for s in range(0, N, bs):
    batch = rows[s:s + bs]
    ids = np.full((len(batch), L), 50256, np.int64); mask = np.zeros((len(batch), L), np.int64)
    for i, r in enumerate(batch):
        t = tok.encode(' ' + r['text'])[:L - len(suffix)] + suffix
        ids[i, L - len(t):] = t; mask[i, L - len(t):] = 1
    ids = torch.tensor(ids); mask = torch.tensor(mask)
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        h = o.hidden_states[11]                       # input to block 11
        qkv = blk.attn.c_attn(blk.ln_1(h))
        _, k, v = qkv.split(768, dim=2)
    K[s:s + len(batch)] = k.numpy().astype(np.float16)
    V[s:s + len(batch)] = v.numpy().astype(np.float16)
    H[s:s + len(batch)] = h[:, -1].numpy()
    mf = mask.unsqueeze(-1).float()
    Z[s:s + len(batch)] = ((h * mf).sum(1) / mf.sum(1)).numpy()
    M[s:s + len(batch)] = mask.numpy().astype(bool)
    if s % 640 == 0:
        print(s, N, round(time.time() - t0, 1), flush=True)
K.flush(); V.flush()
np.save(os.path.join(out, 'H.npy'), H); np.save(os.path.join(out, 'Z.npy'), Z); np.save(os.path.join(out, 'M.npy'), M)
meta = {'domain': [r['domain'] for r in rows], 'split': [r['split'] for r in rows], 'label': [r['label'] for r in rows]}
json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'))
print('done', time.time() - t0)
