
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI needed — saves to file cleanly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent
DB_PATH    = ROOT / "experiments.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────

BG       = "#0d1117"
PANEL    = "#161b22"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#f0883e"
RED      = "#ff7b72"
MUTED    = "#8b949e"
GOAL_F1  = 0.90

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    MUTED,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
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
    iterations = [r["iteration"] for r in runs]
    f1_scores  = [r["best_f1"]   for r in runs]
    best_so_far = []
    running_best = 0.0
    for f in f1_scores:
        running_best = max(running_best, f)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # goal line
    ax.axhline(GOAL_F1, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Goal  {GOAL_F1:.2f}")

    # per-run F1 bars
    colors = [GREEN if f >= GOAL_F1 else ACCENT for f in f1_scores]
    bars = ax.bar(iterations, f1_scores, color=colors, alpha=0.55,
                  width=0.5, zorder=2)

    # running best line
    ax.plot(iterations, best_so_far, color=GREEN, linewidth=2.2,
            marker="o", markersize=7, zorder=3, label="Best so far")

    # value labels on bars
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=9, color="#c9d1d9",
        )

    ax.set_xlim(0.3, max(iterations) + 0.7)
    ax.set_ylim(min(f1_scores) - 0.05, min(1.02, max(f1_scores) + 0.06))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("val_f1", fontsize=11)
    ax.set_title(
        "Self-Improving ML Agent — F1 Across Iterations",
        fontsize=13, pad=14, color="#c9d1d9",
    )
    ax.grid(axis="y", zorder=0)
    ax.legend(framealpha=0.2, edgecolor=MUTED)

    # star on goal-crossing run
    for i, (it, f) in enumerate(zip(iterations, f1_scores)):
        if f >= GOAL_F1:
            ax.annotate(
                f"  ✓ Goal hit\n  run {it}",
                xy=(it, f), xytext=(it + 0.15, f + 0.01),
                fontsize=9, color=GREEN,
            )
            break

    out = OUTPUT_DIR / "f1_progress.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── chart 2: per-run loss curves ──────────────────────────────────────────────

def plot_loss_curves(runs: list[dict]) -> None:
    n = len(runs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 5, rows * 4))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

    for idx, run in enumerate(runs):
        history = json.loads(run["history"] or "[]")
        if not history:
            continue

        epochs     = [e["epoch"]      for e in history]
        train_loss = [e["train_loss"] for e in history]
        val_loss   = [e["val_loss"]   for e in history]
        val_f1     = [e["val_f1"]     for e in history]

        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.set_facecolor(PANEL)

        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL)

        ax.plot(epochs, train_loss, color=ACCENT,  linewidth=1.6, label="train loss")
        ax.plot(epochs, val_loss,   color=ORANGE,  linewidth=1.6, label="val loss",
                linestyle="--")
        ax2.plot(epochs, val_f1,   color=GREEN,   linewidth=2.0, label="val_f1",
                 marker="o", markersize=4)

        ax.set_title(
            f"Run {run['iteration']}  ·  best_f1={run['best_f1']:.4f}",
            fontsize=10, pad=8,
        )
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss",  fontsize=9, color=MUTED)
        ax2.set_ylabel("val_f1", fontsize=9, color=GREEN)
        ax2.tick_params(axis="y", colors=GREEN)
        ax2.axhline(GOAL_F1, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.5)

        ax.grid(axis="both", zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # combined legend
        lines  = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, framealpha=0.2, edgecolor=MUTED,
                  loc="upper right")

    fig.suptitle(
        "Self-Improving ML Agent — Per-Run Training Curves",
        fontsize=13, y=1.01, color="#c9d1d9",
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