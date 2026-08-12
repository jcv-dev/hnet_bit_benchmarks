# 350M Training Process (3 × Vast.ai A100-80GB)

Run each model on a separate Vast.ai machine in parallel. The whole tier
finishes in ~12.5-14 days (bounded by matmulfree) instead of ~23-26 days
sequentially on one GPU.

- Machine 1 — hybrid 350M (~100-125 h)
- Machine 2 — transformer 350M (~140-160 h)
- Machine 3 — matmulfree 350M (~300-340 h)

## Step 0 — Commit + push first (one time, on your machine)

All machines clone from git, so push the uncommitted changes first:

```bash
cd ~/Tesis/hnet_bit_benchmarks
git add -A && git commit -m "350M campaign prep" && git push
```

## Step 1 — Provision (same for all 3)

Vast.ai: **A100-80GB, VM template** (not unprivileged Docker — Triton JIT is
blocked there, 3-10× slower), ≥ 64 GB RAM, ≥ 100 GB disk.

The repo is public, so a plain HTTPS clone works (no GitHub key needed).

## Step 2 — Setup (same for all 3)

```bash
git clone https://github.com/jcv-dev/hnet_bit_benchmarks.git && cd hnet_bit_benchmarks
bash scripts/setup_cloud.sh
mkdir -p ~/.cache/huggingface && echo "hf_..." > ~/.cache/huggingface/token   # optional
```

Note: if `causal-conv1d` fails to install, that's fine — the code falls back
to a native conv (hybrid) or doesn't use it at all (matmulfree, transformer).

## Step 3 — Machine 1: hybrid 350M

```bash
# dataset (bytes only, ~10-20 min)
python scripts/build_dataset.py

# test the new 2-stage topology + GPU probe (~10 min) — BEFORE the full run
bash scripts/probe_350m.sh
#   check: peak_mem_mb 20-30k, stage_0 & stage_1 ratios ~0.1-0.5, loss decreasing

# full run (~100-125 h, ~4-5 days)
tmux new -s hybrid_350M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model hybrid --size 350M --skip_data_build'
```

## Step 4 — Machine 2: transformer 350M (needs BPE tokenization first)

```bash
# dataset incl. BPE (~2.9B tokens, 1-3 h, RAM-heavy — run in tmux)
tmux new -s data_build -d 'cd ~/tesis && source /venv/main/bin/activate && python scripts/build_dataset.py --bpe'
#   wait until done (tmux attach -t data_build), then:

tmux new -s transformer_350M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model transformer --size 350M --skip_data_build'
# (~140-160 h, ~6-7 days.)
```

## Step 5 — Machine 3: matmulfree 350M (no probe needed — flat HGRN, proven at 150M)

```bash
python scripts/build_dataset.py    # bytes only
tmux new -s matmulfree_350M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model matmulfree --size 350M --skip_data_build'
# (~300-340 h, ~12.5-14 days — this is your critical path; start it first.)
```

## Step 6 — Monitoring / resume (all machines)

```bash
tmux attach -t <session>          # Ctrl+B, D to detach
tail -f runs/spanish/<model>_350M.log

# if a run dies:
python train_spanish.py --model <m> --size 350M --skip_data_build \
    --resume_from runs/spanish/<m>_350M/checkpoint_step_<N>.pt
```

## Step 7 — After each run finishes (on each machine)

```bash
python generate_results.py --runs_dir ./runs/spanish --output results_350M.csv   # aggregates that machine's run
python profile_inference.py --export runs/spanish/<m>_350M/model_deploy.pt       # writes *_inference_profile.json
```

## Step 8 — Retrieve results to your local computer

From **your local machine**, pull each machine's run directory and its per-run
CSV into `runs/spanish/`, mirroring the existing `runs/spanish/hybrid_150M/`
layout. One rsync per Vast.ai instance — replace `<IP>` and `<PORT>` with the
values from the Vast.ai instance page (`ssh -p <PORT> root@<IP>`), and adjust
the remote path if you cloned somewhere other than `/workspace` (check with
`pwd` on the machine):

```bash
cd ~/Tesis/hnet_bit_benchmarks

# Machine 1 — hybrid
rsync -av -e "ssh -p <PORT_1>" root@<IP_1>:/workspace/hnet_bit_benchmarks/runs/spanish/hybrid_350M/ runs/spanish/hybrid_350M/
rsync -av -e "ssh -p <PORT_1>" root@<IP_1>:/workspace/hnet_bit_benchmarks/runs/spanish/results_hybrid_350M.csv runs/spanish/

# Machine 2 — transformer
rsync -av -e "ssh -p <PORT_2>" root@<IP_2>:/workspace/hnet_bit_benchmarks/runs/spanish/transformer_350M/ runs/spanish/transformer_350M/
rsync -av -e "ssh -p <PORT_2>" root@<IP_2>:/workspace/hnet_bit_benchmarks/runs/spanish/results_transformer_350M.csv runs/spanish/

# Machine 3 — matmulfree
rsync -av -e "ssh -p <PORT_3>" root@<IP_3>:/workspace/hnet_bit_benchmarks/runs/spanish/matmulfree_350M/ runs/spanish/matmulfree_350M/
rsync -av -e "ssh -p <PORT_3>" root@<IP_3>:/workspace/hnet_bit_benchmarks/runs/spanish/results_matmulfree_350M.csv runs/spanish/
```

This copies everything per model, same as the 150M layout:

| File | Purpose |
|---|---|
| `config.json` | Full training config |
| `training_stats.json` | Params, time, peak memory |
| `training_steps_log.csv` | Step-by-step loss / tok/s / mem |
| `validation_log.csv` | Val BPB every 1k steps |
| `checkpoint_best.pt` / `checkpoint_final.pt` / `checkpoint_milestone_*.pt` | Checkpoints (~2.2 GB each) |
| `model_deploy.pt` | Compact ternary export (auto-generated) |
| `model_deploy_inference_profile.json` | Prefill/decode profile from Step 7 |
| `results_<model>_350M.csv` | Per-run result row (in `runs/spanish/`) |

Verify each machine ran Step 7 before pulling — otherwise `model_deploy.pt`
and the inference profile will be missing (both can be regenerated locally:
`generate_results.py` auto-exports the deploy file, and
`python profile_inference.py --export runs/spanish/<m>_350M/model_deploy.pt`
recreates the profile, as long as the checkpoints were pulled).

### Final aggregation (local, after all 3 machines pulled)

```bash
python generate_results.py --runs_dir ./runs/spanish --output results_350M.csv
```

Merges all runs (150M + 350M) into one CSV and fills in deploy sizes. Done —
the thesis comparison table now has both tiers.
