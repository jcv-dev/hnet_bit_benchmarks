#!/usr/bin/env python3
"""Genera las figuras de la tesis a partir de los logs de entrenamiento.

Figuras:
  learning_curves.png       BPB de validación vs bytes procesados (6 ejecuciones)
  scaling_projection.png    Proyección de BPB vs parámetros hasta 800M
  compression_evolution.png Evolución de las razones de compresión del chunking
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUNS = Path("/home/moravak/Tesis/runs (1)/runs/spanish")
OUT = Path("/home/moravak/Tesis/Documento/Recursos")

MODELS = {
    ("hybrid", "150M"): dict(label="HNetBit 150M", color="#1f77b4", marker="o"),
    ("hybrid", "350M"): dict(label="HNetBit 350M", color="#aec7e8", marker="o"),
    ("matmulfree", "150M"): dict(label="MatMul-Free 150M", color="#ff7f0e", marker="s"),
    ("matmulfree", "350M"): dict(label="MatMul-Free 350M", color="#ffbb78", marker="s"),
    ("transformer", "150M"): dict(label="Transformer 150M", color="#2ca02c", marker="^"),
    ("transformer", "350M"): dict(label="Transformer 350M", color="#98df8a", marker="^"),
}

MILESTONES = [6.25e9, 12.5e9, 18.75e9, 25.0e9]
MAX_PTS = 20


def read_val_log(model: str, size: str):
    """Lee el log de validación, fusionando el reconstruido y el extrapolado si existen."""
    run_dir = RUNS / f"{model}_{size}"
    rows = {}
    for fname in ("validation_log.csv", "validation_log_reconstructed.csv",
                  "validation_log_extrapolated.csv"):
        path = run_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                if step not in rows:
                    rows[step] = (float(row["bytes_seen"]) / 1e9,
                                  float(row["val_bpb"]))
    steps = sorted(rows)
    xs = np.array([rows[s][0] for s in steps])
    ys = np.array([rows[s][1] for s in steps])
    return xs, ys


def read_train_log(model: str, size: str):
    path = RUNS / f"{model}_{size}" / "training_steps_log.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    return cols, rows


def downsample(x, y, max_pts):
    """Puntos uniformemente espaciados en escala log + último punto."""
    if len(x) <= max_pts:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, max_pts - 1, dtype=int))
    idx = np.append(idx, len(x) - 1)
    return x[idx], y[idx]


def milestone_points(xs, ys):
    """Punto más cercano a cada hito de bytes."""
    px, py = [], []
    for m in MILESTONES:
        i = int(np.argmin(np.abs(xs - m / 1e9)))
        px.append(xs[i])
        py.append(ys[i])
    return np.array(px), np.array(py)


def make_learning_curves():
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    for (model, size), style in MODELS.items():
        xs, ys = read_val_log(model, size)
        xd, yd = downsample(xs, ys, MAX_PTS)
        ax.plot(xd, yd, color=style["color"], linewidth=1.6, label=style["label"])
        mx, my = milestone_points(xs, ys)
        ax.plot(mx, my, color=style["color"], marker=style["marker"],
                markersize=5, linestyle="None", markeredgecolor="white",
                markeredgewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.7e9 / 1e9, 3.2e10 / 1e9)
    ax.set_ylim(1.30, 2.35)
    ax.set_xlabel("Bytes de entrenamiento procesados (miles de millones)")
    ax.set_ylabel("BPB de validación")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("learning_curves.png OK")


def make_scaling_projection():
    final_bpb = {
        "Transformer": ((150, 1.4837), (350, 1.3861), "#2ca02c"),
        "MatMul-Free": ((150, 1.4743), (350, 1.3698), "#ff7f0e"),
        "HNetBit":     ((150, 1.6221), (350, 1.4373), "#1f77b4"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=220)
    for name, ((n1, y1), (n2, y2), color) in final_bpb.items():
        n = np.array([n1, n2], dtype=float)
        y = np.array([y1, y2])
        slope = np.polyfit(np.log(n), np.log(y), 1)[0]
        n_ext = np.logspace(np.log10(n1), np.log10(800), 50)
        y_ext = y2 * (n_ext / n2) ** slope
        ax.plot(n_ext, y_ext, color=color, linestyle="--", linewidth=1.2,
                alpha=0.85, label=name)
        ax.plot([n1, n2], [y1, y2], color=color, marker="o", markersize=5.5,
                linestyle="None", zorder=3)
        y800 = y2 * (800 / n2) ** slope
        ax.plot([800], [y800], color=color, marker="*", markersize=11,
                linestyle="None", zorder=3)
        # Etiquetas de valor sin superposiciones: offsets distintos por modelo
        if name == "Transformer":
            offs = [(0, 9), (0, 9), (0, 9)]
        elif name == "MatMul-Free":
            offs = [(0, -12), (0, -12), (0, -12)]
        else:  # HNetBit
            offs = [(0, 9), (0, 9), (0, 30)]
        for (nn, yy), (dx, dy) in zip([(n1, y1), (n2, y2), (800, y800)], offs):
            ax.annotate(f"{yy:.2f}", (nn, yy), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlim(120, 1000)
    ax.set_ylim(1.20, 1.70)
    ax.set_xticks([150, 350, 800])
    ax.set_xticklabels([])  # sin etiquetas en los ticks (evita superposición)
    ax.set_xlabel("Parámetros del modelo", labelpad=2)
    ax.set_ylabel("BPB final")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    # Etiquetas de tamaño dentro del área de trazado, en la parte inferior
    for nn, label in [(150, "150M"), (350, "350M"), (800, "800M")]:
        ax.annotate(label, (nn, 1.205), ha="center", va="bottom",
                    fontsize=8.5, color="#444444")
    fig.tight_layout()
    fig.savefig(OUT / "scaling_projection.png", bbox_inches="tight")
    plt.close(fig)
    print("scaling_projection.png OK")


def smooth_centered(y, win):
    """Media móvil centrada con normalización correcta en los bordes."""
    kernel = np.ones(win)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / den


def make_compression_evolution():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), dpi=200, sharey=False)
    for ax, (model, size) in zip(axes, [("hybrid", "150M"), ("hybrid", "350M")]):
        cols, rows = read_train_log(model, size)
        xs = np.array([float(r["bytes_seen"]) / 1e9 for r in rows])
        stage_cols = [c for c in cols if c.startswith("stage_") and c.endswith("_compression_ratio")]
        for c in sorted(stage_cols):
            ys = np.array([float(r[c]) for r in rows])
            ys = smooth_centered(ys, 100)
            xd, yd = downsample(xs, ys, 100)
            ax.plot(xd, yd, linewidth=1.1, alpha=0.9,
                    label=c.replace("_compression_ratio", "").replace("_", " ").capitalize())
        if "overall_compression_ratio" in cols and len(stage_cols) > 1:
            ys = np.array([float(r["overall_compression_ratio"]) for r in rows])
            ys = smooth_centered(ys, 100)
            xd, yd = downsample(xs, ys, 100)
            ax.plot(xd, yd, linewidth=1.6, color="black", label="Global")
        ax.set_xlabel("Bytes procesados (miles de millones)")
        ax.set_ylabel("Fracción de límites (razón de compresión)")
        ax.set_title(f"HNetBit {size}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "compression_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("compression_evolution.png OK")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_learning_curves()
    make_scaling_projection()
    make_compression_evolution()
