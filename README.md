# CoSE: scope-bound evolution capsules for federated continual adaptation of language models

Code for the manuscript *Scope-Bound Evolution Capsules with Evidence-Gated Admission for Collective Continual Adaptation of Federated Language Models at the Edge*.

Every number in the paper's tables and figures is produced by the scripts in this folder. All files are kept in one flat folder; the scripts create `models/`, `data/`, `cache/`, `runs/`, `logs/` and `figs/` beside themselves when they run.

## What the code does

* **Backbone.** GPT-2 small (124M), taken from the public ONNX export of the original checkpoint and converted to a PyTorch state dict (`convert_onnx.py`). Only block 11 (the read-out block) is adapted. LoRA with rank 8 is applied to the attention output projection and to both MLP projections.
* **Domains.** Six public corpora with one 20-way verbalised label space: AG News, Reuters R8, TREC, MR, CR and NLTK short messages. MPQA is held out as an external test set (`prep_data.py`).
* **Caching.** The frozen layers are evaluated once. `cache_features.py` stores the block-11 inputs and keys, and `derive_attn.py` stores the attention output at the read-out token.
* **Simulator** (`cose_sim.py`, numerics in `cose_core.py`). It runs twelve asynchronous nodes for 36 rounds, with abrupt domain switches, two conflicting labelling policies and unreliable senders (label flip, noise, sign-flipped and amplified updates, or sleepers). It implements CoSE and these baselines: Local, FedAvg-LoRA (FedIT), FedProx-LoRA, FFA-LoRA, Ditto, Gossip-LoRA, Median-LoRA and TIES exchange.
* **CoSE components** (each can be switched off for the ablations):
  * scope-routed working adapter
  * Page–Hinkley and key-shift drift sealing
  * self-contained capsules (truncated SVD of the effective delta)
  * trust gate: buffer gain, peer consensus, component-dropout mutual information, Beta reputation and quarantine
  * spectral first-order component pruning
  * re-validation
  * provenance with mixing weights, used for lineage repair after a revocation

## Reproduce

```bash
pip install -r requirements.txt
bash download_data.sh          # weights, tokenizer, corpora from public GitHub repositories
python convert_onnx.py         # -> models/gpt2_small.pt (prints a sanity prediction)
python prep_data.py            # -> data/pool.jsonl, data/pool_ext.jsonl
python cache_features.py       # main pool (about 20 min on 2 CPU cores)
python cache_features.py _ext  # external MPQA pool
python derive_attn.py && python derive_attn.py _ext
bash runmany.sh jobs_all.txt   # every reported run (seeds 10-19), two at a time
python analyze.py              # -> results.json (means, bootstrap CIs, Wilcoxon + Holm)
python fig_results.py && python fig_arch.py
python efficiency.py           # measured operation costs
```

A single run looks like this:

```bash
python cose_sim.py --method cose --seed 10 --out runs/example.json
python cose_sim.py --method median --seed 10 --attackers none
python cose_sim.py --method cose --sleeper 2 --use_gate 0 --attackers flip --rollback repair
```

`cose_sim.py --help` lists every option. The hyperparameter defaults in `DEF` are the values used in the paper. They were fixed on pilot seeds 0 and 1, which are not part of any reported result.

## Files

| File | Purpose |
|---|---|
| `download_data.sh`, `convert_onnx.py`, `prep_data.py` | Data and weight acquisition |
| `cache_features.py`, `derive_attn.py` | Frozen-backbone feature cache |
| `cose_core.py` | Adapted read-out block, LoRA, routed capsule library, ECE |
| `cose_sim.py` | Federation simulator, CoSE and all baselines, metrics |
| `runmany.sh`, `jobs_all.txt` | Job list for every experiment in the paper |
| `analyze.py` | Aggregation and statistics |
| `fig_arch.py`, `fig_results.py` | Figures (grayscale) |
| `efficiency.py` | Timing measurements |
| `central.py` | Pilot check of joint and single-domain central LoRA training (Sections III-C and VI-A) |
| `jobs_tuning.txt` | FedProx $\mu$ selection on the pilot seeds |
| `results.json`, `efficiency.json` | Aggregated results of the reported runs |

## Licence and data

The code is released for research use. The corpora remain under their original licences, and the scripts download them from their public locations.
