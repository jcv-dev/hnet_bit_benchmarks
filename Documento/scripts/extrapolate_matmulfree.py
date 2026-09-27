#!/usr/bin/env python3
"""Extrapola la porción temprana de la curva de validación de matmulfree_150M.

La validación periódica real de matmulfree_150M solo sobrevivió desde el paso
166,000 (21.76B bytes). Se reconstruye la porción temprana usando la FORMA de la
curva completa de validación de matmulfree_350M (misma arquitectura y esquema de
bytes por paso), desplazada por una constante que se ancla en el primer punto
real de matmulfree_150M para garantizar continuidad en la unión.

Escribe validation_log_extrapolated.csv (mismo formato que validation_log.csv).
"""

import csv
import math
import numpy as np
from pathlib import Path

RUNS = Path("/home/moravak/Tesis/runs (1)/runs/spanish")
SRC = RUNS / "matmulfree_350M" / "validation_log.csv"
DST_RUN = RUNS / "matmulfree_150M"
DST = DST_RUN / "validation_log_extrapolated.csv"

LN2 = math.log(2)

with open(SRC) as f:
    src_rows = list(csv.DictReader(f))
src_steps = np.array([int(r["step"]) for r in src_rows])
src_bpb = np.array([float(r["val_bpb"]) for r in src_rows])

with open(DST_RUN / "validation_log.csv") as f:
    real_rows = list(csv.DictReader(f))
real_first_step = int(real_rows[0]["step"])
real_first_bpb = float(real_rows[0]["val_bpb"])
print(f"Primer punto real: step {real_first_step}, bpb {real_first_bpb:.4f}")

# Ancla: valor de la curva 350M en el mismo paso (mismos bytes por paso)
i_anchor = int(np.argmin(np.abs(src_steps - real_first_step)))
if abs(src_steps[i_anchor] - real_first_step) > 100:
    raise RuntimeError("No hay evaluación 350M cerca del punto de ancla")
delta = real_first_bpb - float(src_bpb[i_anchor])
print(f"Ancla 350M: step {src_steps[i_anchor]}, bpb {src_bpb[i_anchor]:.4f} -> delta {delta:.4f}")

# Isotonic regression no creciente sobre la porción sintética (PAVA)
def pava_non_increasing(y):
    y = list(y)
    blocks = [[v] for v in y]
    while True:
        merged = False
        for i in range(len(blocks) - 1):
            if np.mean(blocks[i]) < np.mean(blocks[i + 1]):
                blocks[i] = blocks[i] + blocks[i + 1]
                del blocks[i + 1]
                merged = True
                break
        if not merged:
            break
    out = []
    for b in blocks:
        out.extend([np.mean(b)] * len(b))
    return np.array(out)

rows_out = []
n = 0
for r in src_rows:
    step = int(r["step"])
    if step >= real_first_step:
        break
    bpb = float(r["val_bpb"]) + delta
    rows_out.append((step, r["bytes_seen"], bpb))
    n += 1

raw = np.array([b for _, _, b in rows_out])
fit = pava_non_increasing(raw)
# Desplazamiento constante para continuidad exacta con el primer punto real
fit = fit + (real_first_bpb - fit[-1])

with open(DST, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "bytes_seen", "val_loss", "val_bpb"])
    for (step, bytes_seen, _), bpb in zip(rows_out, fit):
        w.writerow([step, bytes_seen, round(bpb * LN2, 6), round(bpb, 6)])

# Verificación: continuidad y monotonicidad de la curva fusionada
merged = list(zip([s for s, _, _ in rows_out], fit))
merged += [(int(r["step"]), float(r["val_bpb"])) for r in real_rows]
merged.sort()
bpbs = [b for _, b in merged]
viol = sum(1 for a, b in zip(bpbs, bpbs[1:]) if b > a)
print(f"Escritos {n} puntos sintéticos; total curva {len(merged)}; "
      f"violaciones de monotonicidad: {viol}")
print(f"Último sintético: step {merged[n-1][0]}, bpb {merged[n-1][1]:.4f}")
print(f"Primer real      : step {merged[n][0]},   bpb {merged[n][1]:.4f}")
print(f"Inicio sintético: step {merged[0][0]}, bpb {merged[0][1]:.4f}")