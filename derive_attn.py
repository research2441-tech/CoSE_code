"""Collapse cached keys/values into the read-out token's attention output (frozen query)."""
import os, numpy as np, torch
import torch.nn.functional as F
import sys
W = os.path.dirname(os.path.abspath(__file__)); c = os.path.join(W, 'cache' + (sys.argv[1] if len(sys.argv) > 1 else ''))
sd = torch.load(os.path.join(W, 'models/gpt2_small.pt'))
K = np.load(os.path.join(c, 'K.npy'), mmap_mode='r'); V = np.load(os.path.join(c, 'V.npy'), mmap_mode='r')
H = torch.from_numpy(np.load(os.path.join(c, 'H.npy'))); M = torch.from_numpy(np.load(os.path.join(c, 'M.npy')))
x = F.layer_norm(H, (768,), sd['h.11.ln_1.weight'], sd['h.11.ln_1.bias'])
q = x @ sd['h.11.attn.c_attn.weight'][:, :768] + sd['h.11.attn.c_attn.bias'][:768]
N = H.shape[0]; O = np.zeros((N, 768), np.float32)
for s in range(0, N, 512):
    k = torch.from_numpy(np.asarray(K[s:s + 512], dtype=np.float32)).view(-1, 64, 12, 64).transpose(1, 2)
    v = torch.from_numpy(np.asarray(V[s:s + 512], dtype=np.float32)).view(-1, 64, 12, 64).transpose(1, 2)
    qq = q[s:s + 512].view(-1, 12, 1, 64)
    att = (qq @ k.transpose(-1, -2)) / 8.0
    att = att.masked_fill(~M[s:s + 512][:, None, None, :], float('-inf')).softmax(-1)
    O[s:s + 512] = (att @ v).transpose(1, 2).reshape(-1, 768).numpy()
np.save(os.path.join(c, 'O.npy'), O)
print('saved', O.shape)
