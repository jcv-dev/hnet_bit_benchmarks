# Cloud Deployment Guide

This guide covers deploying the Spanish benchmark on a rented cloud GPU.

## 1. Rent a cloud GPU

**Requirements:**
- NVIDIA A100 80GB (or H100, A6000)
- At least 100 GB disk (200 GB recommended for datasets + checkpoints)
- CUDA 12.4+ pre-installed
- SSH access

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

Required only for the transformer baseline (which uses the gpt2 tokenizer by default, so this is optional):

```bash
huggingface-cli login
```

If you want to use the Llama tokenizer instead of gpt2:
```bash
huggingface-cli login --token YOUR_HF_TOKEN
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

### Individual runs

```bash
# Lightweight baseline (starts producing results quickly):
nohup python train_spanish.py --model matmulfree --size 150M > matmulfree_150M.log 2>&1 &
nohup python train_spanish.py --model hybrid --size 150M > hybrid_150M.log 2>&1 &
nohup python train_spanish.py --model transformer --size 150M > transformer_150M.log 2>&1 &

# Ablation:
nohup python train_spanish.py --model hybrid_attn --size 150M > hybrid_attn_150M.log 2>&1 &
```

### Monitor progress

```bash
# Check loss every 10 steps (log output):
tail -f matmulfree_150M.log

# TensorBoard:
tensorboard --logdir runs/spanish --bind_all

# Check what's running:
jobs -l
```

## 8. Retrieve results

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
| `Triton not available` | Check CUDA version. Triton requires CUDA 11.8+ |
| `Dataset download fails` | The dataset is from HuggingFace. Ensure internet access and try again |
| `huggingface_hub.errors.GatedRepoError` | You need to login for the Llama tokenizer: `huggingface-cli login`. Or use the default gpt2 tokenizer |
| `Training is slow` | Ensure `--no_bf16` is NOT set. BF16 is enabled by default and doubles throughput on A100 |
| `SSH disconnects during training` | Use `nohup` or `tmux` / `screen` to keep the process alive |
