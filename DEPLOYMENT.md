# Cloud Deployment Guide

This guide covers deploying the Spanish benchmark on a rented cloud GPU.

## 1. Rent a cloud GPU

The models use Triton kernels for fast training. Triton's JIT compiler requires the ability to create executable memory pages at runtime, which is blocked in unprivileged Docker containers (the default on Vast.ai Docker templates). Choose a platform that supports this.

**Provider recommendations:**

| Provider | Triton works? | Typical price (A100 80GB) | Notes |
|---|---|---|---|
| RunPod (PyTorch template) | Always | ~$1.20/hr | Privileged containers by default |
| Vast.ai (VM template) | Always | ~$0.60-1.20/hr | Full virtual machine, no restrictions |
| Vast.ai (Docker template) | Maybe | ~$0.60-1.20/hr | Depends on host. Test before committing. |
| Lambda Labs | Always | ~$1.10/hr | VM instances, not containers |

**Safe choices:** Rent a **VM template** on Vast.ai (use the search filter "VM") or use **RunPod**. Avoid Vast.ai Docker templates if you want to be certain Triton will work.

**Fallback if forced into Docker:** Set `HNETBIT_DISABLE_TRITON=1` to use naive PyTorch loops instead. Training will be 3-10x slower but will produce identical results. The lazy workaround is to skip the `matmulfree` model (which has its own hardcoded Triton) and train only `hybrid`, `hybrid_attn`, and `transformer`.

**Requirements:**
- NVIDIA A100 80GB (or H100, A6000)
- At least 100 GB disk (200 GB recommended for datasets + checkpoints)
- CUDA 12.4+ pre-installed

## 2. Get the code onto the machine

### Option A: Git (recommended)

Push your local repo to a private GitHub/GitLab repository, then clone on the cloud machine:

```bash
# On the cloud machine:
git clone https://github.com/yourusername/tesis.git
cd tesis
```

### Option B: rsync over SSH

From your local machine (replace with the cloud IP):

```bash
rsync -avz --exclude 'venv' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.git' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'runs' --exclude 'data' \
  /path/to/tesis/ user@cloud-ip:/home/user/tesis/
```

### Option C: Vast.ai direct upload

Use Vast.ai's file upload in their web UI, or use `scp`:

```bash
scp -P <port> -r /path/to/tesis/ user@host:/root/tesis/
```

## 3. Run the setup script

On the cloud machine:

```bash
cd tesis

# Make the setup script executable and run it
chmod +x scripts/setup_cloud.sh
bash scripts/setup_cloud.sh
```

This script:
- Creates a Python virtual environment
- Installs PyTorch with CUDA 12.4
- Installs transformers, triton, datasets, einops, etc.
- Attempts to install causal-conv1d (optional, falls back gracefully)
- Verifies CUDA and Triton are functional

## 4. (Optional) Log in to HuggingFace

Required only for the transformer baseline (which uses the gpt2 tokenizer by default, so this is optional).

The `huggingface-cli` shell command may not be available on some templates. Use one of these methods instead:

**Method A: Python login (interactive)**
```bash
python3 -c "from huggingface_hub import login; login()"
```
You will be prompted for a token. Paste from https://huggingface.co/settings/tokens.

**Method B: Write token file directly (non-interactive)**
```bash
mkdir -p ~/.cache/huggingface
echo -n 'YOUR_TOKEN' > ~/.cache/huggingface/token
```

**Method C: If you want to use the Llama tokenizer instead of gpt2:**
```bash
python3 -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"
```

Verify the login worked:
```bash
python3 -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

## 5. Run the smoke test

Verify everything is working with a quick smoke test:

```bash
source venv/bin/activate
bash test_smoke.sh --gpu
```

This trains all model types at tiny size for 15 steps. Expect ~2-5 minutes on an A100.

If the smoke test passes, check the output:
```bash
cat runs/smoke_test/results.csv
```

## 6. Build the dataset

Before running the full benchmark, download and preprocess the Spanish Billion Words dataset:

```bash
# For byte-level models (matmulfree, hybrid, hybrid_attn):
python train_spanish.py --model hybrid --size 150M --max_steps 1 --batch_size 1

# For the transformer (BPE) model:
python train_spanish.py --model transformer --size 150M --max_steps 1 --batch_size 1
```

This downloads ~1.5 GB of text from HuggingFace and tokenizes it. The data caches to `./data/spanish/` and subsequent runs use the cache automatically with `--skip_data_build`.

## 7. Run the benchmark

Each training run takes 20-200 hours. Use `tmux` to keep runs alive across SSH disconnects.

### Using tmux (recommended over nohup)

```bash
# Start a named session for each model:
tmux new -s hybrid_150M

# Inside the tmux session, start training:
cd ~/tesis
source /venv/main/bin/activate
python train_spanish.py --model hybrid --size 150M

# Detach: Ctrl+B, then D
# Reconnect later: tmux attach -t hybrid_150M

# Start multiple runs in parallel (one session per model):
tmux new -s matmulfree_150M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model matmulfree --size 150M'
tmux new -s hybrid_150M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model hybrid --size 150M'
tmux new -s transformer_150M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model transformer --size 150M'
```

If `tmux` is not installed:
```bash
sudo apt update && sudo apt install -y tmux
```

### Without tmux (using nohup)

```bash
nohup python train_spanish.py --model hybrid --size 150M > hybrid_150M.log 2>&1 &

# Monitor output:
tail -f hybrid_150M.log
```

### Monitor progress

```bash
# List running tmux sessions:
tmux ls

# Attach to a running session to see live output:
tmux attach -t hybrid_150M

# List all running processes:
ps aux | grep train_spanish

# TensorBoard:
tensorboard --logdir runs/spanish --bind_all
```

## 8. Run order and schedule

Run all models grouped by size tier. Generate partial results after each tier, then a final aggregate at the end.

### Before starting

Delete any prep-run artifacts (from the dataset build step):

```bash
rm -rf runs/spanish/*
```

### Tier 1: 150M (4 runs)

Run on a single A100. Estimated time: ~120 hours total (sequential) or ~40 hours (parallel).

```bash
rm -rf runs/spanish/*

# Launch all 4 in parallel (one per tmux session):
tmux new -s hybrid_150M      -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model hybrid      --size 150M'
tmux new -s transformer_150M  -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model transformer  --size 150M'
tmux new -s matmulfree_150M  -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model matmulfree   --size 150M'
tmux new -s hybrid_attn_150M -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model hybrid_attn  --size 150M'

# Monitor:
tmux attach -t hybrid_150M
```

After all 4 complete, aggregate results for the 150M tier:

```bash
python generate_results.py --runs_dir ./runs/spanish --output results_150M.csv
```

This writes `results_150M.csv` with only the 150M rows. It is read-only on the per-run data — it never modifies config.json, checkpoints, or logs.

### Tier 2: 350M (3 runs: hybrid, transformer, matmulfree)

No hybrid_attn at 350M (the attention ablation is only run at 150M).

```bash
tmux new -s hybrid_350M      -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model hybrid      --size 350M'
tmux new -s transformer_350M  -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model transformer  --size 350M'
tmux new -s matmulfree_350M  -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model matmulfree   --size 350M'

# After all complete:
python generate_results.py --runs_dir ./runs/spanish --output results_350M.csv
```

### Tier 3: 750M (3 runs: hybrid, transformer, matmulfree)

```bash
tmux new -s hybrid_750M      -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model hybrid      --size 750M'
tmux new -s transformer_750M  -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model transformer  --size 750M'
tmux new -s matmulfree_750M  -d 'cd ~/tesis && source /venv/main/bin/activate && python train_spanish.py --model matmulfree   --size 750M'
```

### Final aggregate

```bash
python generate_results.py --runs_dir ./runs/spanish --output results.csv
```

`generate_results.py` is safe to run multiple times — it only reads per-run data files and writes the CSV specified by `--output`. Each call to a different output file produces independent results.

## 9. Retrieve results

### Option A: rsync results back to local

While training is running or after it completes:

```bash
# From your local machine:
rsync -avz user@cloud-ip:~/tesis/runs/ ./tesis/runs/
rsync -avz user@cloud-ip:~/tesis/results.csv ./tesis/results.csv
```

### Option B: Aggregate on cloud, download CSV only

```bash
# On the cloud:
python generate_results.py --runs_dir ./runs/spanish --output results.csv

# From your local machine:
scp user@cloud-ip:~/tesis/results.csv ./
```


## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce batch size: `--batch_size 2` or `--grad_accum 4` |
| `Triton kernel compilation fails` (Docker) | Set `HNETBIT_DISABLE_TRITON=1` to force CPU fallback. This makes training slower (the model falls back to naive PyTorch loops) but allows the benchmark to run in unprivileged containers. Example: `HNETBIT_DISABLE_TRITON=1 python train_spanish.py --model hybrid --size 150M` |
| `Triton not available` | Check CUDA version. Triton requires CUDA 11.8+ |
| `Dataset download fails` | The dataset is from HuggingFace. Ensure internet access and try again |
| `huggingface_hub.errors.GatedRepoError` | You need a HuggingFace token. Run `python3 -c "from huggingface_hub import login; login()"` or create `~/.cache/huggingface/token` manually with your token from https://huggingface.co/settings/tokens |
| `Training is slow` | Ensure `--no_bf16` is NOT set. BF16 is enabled by default and doubles throughput on A100 |
| `SSH disconnects during training` | Use `tmux` (recommended): start training inside `tmux new -s run_name`, then detach with Ctrl+B then D. Reconnect with `tmux attach -t run_name`. Alternatively use `nohup python ... > log.txt 2>&1 &` but you cannot reattach to see live output. |
