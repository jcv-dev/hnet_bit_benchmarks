#!/usr/bin/env bash
# TESTING-ONLY script for the 350M tier.
#
# 1. Validates the 2-stage (3-group) HNetBit topology on CPU — the 350M
#    hybrid is the first 2-stage run in this repo (150M/tiny are 1-stage).
# 2. Runs a 20-step GPU probe of the real hybrid 350M config in a scratch
#    run dir (never touches runs/spanish/hybrid_350M/).
#
# The full pipeline runs are launched manually with the direct
# train_spanish.py commands (see AGENTS.md / DEPLOYMENT.md), same as the
# 150M tier.
#
# Usage:
#   bash scripts/probe_350m.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Activate the template venv if present (Vast.ai / RunPod convention)
if [ -d /venv/main ]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

# ---------------------------------------------------------------------------
# [1/2] 2-stage topology validation (CPU, ~1 minute)
# ---------------------------------------------------------------------------
echo "=== [1/2] Validating 2-stage (3-group) HNetBit topology ==="
python3 "$ROOT/scripts/validate_2stage.py"

# ---------------------------------------------------------------------------
# [2/2] GPU probe: hybrid 350M, 20 optimizer steps in a scratch run dir
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/2] GPU probe: hybrid 350M (20 steps, scratch dir) ==="
PROBE_DIR="$ROOT/runs/spanish/probe"
rm -rf "$PROBE_DIR"
python3 "$ROOT/train_spanish.py" \
    --model hybrid --size 350M \
    --max_steps 20 \
    --skip_data_build \
    --output_dir "$PROBE_DIR"

echo ""
echo "--- Probe results (runs/spanish/probe/hybrid_350M/training_steps_log.csv) ---"
python3 -c "
import csv
rows = list(csv.DictReader(open('$PROBE_DIR/hybrid_350M/training_steps_log.csv')))
last = rows[-1]
print(f\"  peak_mem_mb       : {last['peak_mem_mb']} MB\")
print(f\"  tok_per_sec       : {last['tok_per_sec']} bytes/s\")
for k, v in last.items():
    if 'compression' in k:
        print(f'  {k:<20}: {v}')
print('  loss trajectory: ' + ' -> '.join(f\"{r['loss'][:6]}\" for r in rows))
"

echo ""
echo "--- GO/NO-GO checks ---"
echo "  peak_mem_mb should be roughly 20,000-30,000 MB (not > 65,000)"
echo "  both stage_0 and stage_1 compression ratios should be in ~0.1-0.5"
echo "  loss should be decreasing"
echo "  if NO-GO: fix before launching; the full runs take 4-14 days each"
echo ""
echo "Probe complete. Launch the full runs manually, e.g.:"
echo "  tmux new -s hybrid_350M -d 'python train_spanish.py --model hybrid --size 350M --skip_data_build'"
