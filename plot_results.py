"""
plot_results.py

Reads completed run data from experiments.db and saves publication-quality
charts to outputs/. Run after agent.py completes — no re-training needed.

Generates:
  outputs/f1_progress.png    — val_f1 across iterations (LinkedIn money shot)
  outputs/loss_curves.png    — per-run stacked subplots: loss (top) + val_f1
                               (bottom), fully separated for legibility
"""

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no GUI — saves to file cleanly, no threading issues
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent
DB_PATH    = ROOT / "experiments.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────

BG      = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f0883e"
MUTED   = "#8b949e"
TEXT    = "#c9d1d9"
GRID    = "#21262d"
GOAL_F1 = 0.90

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   MUTED,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

# ── load data ─────────────────────────────────────────────────────────────────

def load_runs() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE status = 'done' ORDER BY iteration ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── chart 1: F1 progress ──────────────────────────────────────────────────────

def plot_f1_progress(runs: list[dict]) -> None:
    iterations  = [r["iteration"] for r in runs]
    f1_scores   = [r["best_f1"]   for r in runs]

    running_best = 0.0
    best_so_far  = []
    for f in f1_scores:
        running_best = max(running_best, f)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.axhline(GOAL_F1, color=GREEN, linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Goal  {GOAL_F1:.2f}")

    colors = [GREEN if f >= GOAL_F1 else ACCENT for f in f1_scores]
    bars   = ax.bar(iterations, f1_scores, color=colors, alpha=0.55,
                    width=0.5, zorder=2)

    ax.plot(iterations, best_so_far, color=GREEN, linewidth=2.2,
            marker="o", markersize=7, zorder=3, label="Best so far")

    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, color=TEXT,
        )

    ax.set_xlim(0.3, max(iterations) + 0.7)
    ax.set_ylim(min(f1_scores) - 0.05, min(1.02, max(f1_scores) + 0.06))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("val_f1",    fontsize=11)
    ax.set_title(
        "Self-Improving ML Agent — F1 Across Iterations",
        fontsize=13, pad=14, color=TEXT,
    )
    ax.grid(axis="y", zorder=0)
    ax.legend(framealpha=0.2, edgecolor=MUTED)

    for it, f in zip(iterations, f1_scores):
        if f >= GOAL_F1:
            ax.annotate(
                f"  Goal hit\n  run {it}",
                xy=(it, f), xytext=(it + 0.15, f + 0.01),
                fontsize=9, color=GREEN,
            )
            break

    out = OUTPUT_DIR / "f1_progress.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── chart 2: per-run stacked subplots ────────────────────────────────────────
#
# Each run gets a column of 2 stacked subplots:
#   top    — train_loss (dashed) + val_loss (solid)   [loss scale, left axis]
#   bottom — val_f1 with goal line                    [0–1 scale, left axis]
#
# Previously this was a single subplot with twinx() dual axes, which caused
# val_f1 and loss lines to visually overlap and share cramped tick space.
# Stacked subplots give each metric its own full vertical range and axis.

def plot_loss_curves(runs: list[dict]) -> None:
    n    = len(runs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    # each run = 2 subplot rows (loss on top, f1 on bottom) + a shared title row
    fig = plt.figure(figsize=(cols * 5, rows * 7))
    fig.patch.set_facecolor(BG)

    # outer grid: one column-group per run
    outer = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.35)

    for idx, run in enumerate(runs):
        history = json.loads(run["history"] or "[]")
        if not history:
            continue

        epochs     = [e["epoch"]      for e in history]
        train_loss = [e["train_loss"] for e in history]
        val_loss   = [e["val_loss"]   for e in history]
        val_f1     = [e["val_f1"]     for e in history]

        # inner grid: 2 rows (loss / f1) within this run's cell
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[idx // cols, idx % cols],
            hspace=0.08,          # tight gap between loss and f1 panels
            height_ratios=[1, 1],
        )

        ax_loss = fig.add_subplot(inner[0])
        ax_f1   = fig.add_subplot(inner[1], sharex=ax_loss)

        # ── loss panel ────────────────────────────────────────────────────
        ax_loss.set_facecolor(PANEL)
        ax_loss.plot(epochs, train_loss, color=ACCENT,  linewidth=1.6,
                     linestyle="--", alpha=0.8, label="train loss")
        ax_loss.plot(epochs, val_loss,   color=ORANGE,  linewidth=1.8,
                     label="val loss")
        ax_loss.set_ylabel("Loss", fontsize=9, color=MUTED)
        ax_loss.tick_params(colors=MUTED, labelsize=8)
        ax_loss.grid(zorder=0)
        ax_loss.legend(fontsize=7, framealpha=0.2, edgecolor=MUTED,
                       loc="upper right")

        # run title lives on the loss panel
        ax_loss.set_title(
            f"Run {run['iteration']}  ·  best_f1 = {run['best_f1']:.4f}",
            fontsize=10, pad=8, color=TEXT,
        )
        # hide x-tick labels on top panel — shared axis shows them on bottom
        plt.setp(ax_loss.get_xticklabels(), visible=False)

        for spine in ax_loss.spines.values():
            spine.set_edgecolor(MUTED)

        # ── val_f1 panel ──────────────────────────────────────────────────
        ax_f1.set_facecolor(PANEL)
        ax_f1.plot(epochs, val_f1, color=GREEN, linewidth=2.0,
                   marker="o", markersize=4, label="val_f1")
        ax_f1.axhline(GOAL_F1, color=GREEN, linewidth=0.9,
                      linestyle=":", alpha=0.6, label=f"Goal {GOAL_F1}")
        ax_f1.set_ylabel("val_f1", fontsize=9, color=GREEN)
        ax_f1.set_xlabel("Epoch",  fontsize=9, color=MUTED)
        ax_f1.tick_params(colors=MUTED, labelsize=8)
        ax_f1.tick_params(axis="y", colors=GREEN)
        ax_f1.set_ylim(
            max(0, min(val_f1) - 0.05),
            min(1.02, max(val_f1) + 0.05),
        )
        ax_f1.grid(zorder=0)
        ax_f1.legend(fontsize=7, framealpha=0.2, edgecolor=MUTED,
                     loc="lower right")
        ax_f1.xaxis.set_major_locator(MaxNLocator(integer=True))

        for spine in ax_f1.spines.values():
            spine.set_edgecolor(MUTED)

    fig.suptitle(
        "Self-Improving ML Agent — Per-Run Training Curves",
        fontsize=13, y=1.01, color=TEXT,
    )

    out = OUTPUT_DIR / "loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runs = load_runs()
    if not runs:
        print("No completed runs found in experiments.db.")
    else:
        print(f"Found {len(runs)} completed run(s).\n")
        plot_f1_progress(runs)
        plot_loss_curves(runs)
        print("\nDone. Open outputs/ to see the charts.")