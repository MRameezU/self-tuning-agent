
import base64
import io
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from jinja2 import Environment, BaseLoader

from config import DB_PATH, OUTPUTS_DIR, GOAL_F1, MODEL_NAME, DATASET_NAME

logger = logging.getLogger(__name__)


# ── DB helper (local, doesn't import from memory.py to keep report standalone) ─

@contextmanager
def _db(path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Chart generation ──────────────────────────────────────────────────────────

def _f1_progression_chart(runs: list[sqlite3.Row]) -> str:
    """
    Bar chart of best_f1 per iteration with the goal line overlaid.
    Returns a base64-encoded PNG data URI.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive — safe for headless / subprocess
    import matplotlib.pyplot as plt

    iterations = [r["iteration"] for r in runs]
    f1_scores  = [r["best_f1"]   for r in runs]
    colours    = ["#4CE87A" if f >= GOAL_F1 else "#4C9BE8" for f in f1_scores]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.bar(iterations, f1_scores, color=colours, width=0.6, zorder=3)

    # value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{score:.4f}",
            ha="center", va="bottom",
            color="#e0e0e0", fontsize=8,
        )

    ax.axhline(GOAL_F1, color="#E8834C", linestyle="--", linewidth=1.5,
               zorder=4, label=f"Goal F1 = {GOAL_F1}")

    ax.set_xlabel("Iteration", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("Best Val F1", color="#e0e0e0", fontsize=9)
    ax.set_title("F1 Score Progression Across Agent Iterations",
                 color="#e0e0e0", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(iterations)
    ax.tick_params(colors="#e0e0e0", labelsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#16213e", labelcolor="#e0e0e0", fontsize=8)
    ax.grid(axis="y", color="#2d2d5e", linewidth=0.6, linestyle="--", alpha=0.7, zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d5e")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _epoch_curve_chart(history: list[dict], run_id: str) -> str:
    """
    Small dual-axis chart: loss (left) + val_f1 (right) per epoch.
    Returns a base64-encoded PNG data URI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return ""

    epochs     = [e["epoch"]      for e in history]
    train_loss = [e["train_loss"] for e in history]
    val_loss   = [e["val_loss"]   for e in history]
    val_f1     = [e["val_f1"]     for e in history]

    fig, ax1 = plt.subplots(figsize=(7, 3), facecolor="#1a1a2e")
    ax1.set_facecolor("#16213e")

    ax1.plot(epochs, train_loss, "--", color="#4C9BE8", linewidth=1.2,
             alpha=0.7, label="train loss")
    ax1.plot(epochs, val_loss,   "-",  color="#4C9BE8", linewidth=1.6,
             label="val loss")
    ax1.set_xlabel("Epoch", color="#e0e0e0", fontsize=8)
    ax1.set_ylabel("Loss",  color="#4C9BE8", fontsize=8)
    ax1.tick_params(axis="y", colors="#4C9BE8", labelsize=7)
    ax1.tick_params(axis="x", colors="#e0e0e0", labelsize=7)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, "-", color="#4CE87A", linewidth=1.8,
             marker="o", markersize=3, label="val F1")
    ax2.set_ylabel("Val F1", color="#4CE87A", fontsize=8)
    ax2.tick_params(axis="y", colors="#4CE87A", labelsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(GOAL_F1, color="#E8834C", linestyle=":", linewidth=1,
                alpha=0.8)

    for spine in ax1.spines.values():
        spine.set_edgecolor("#2d2d5e")
    ax1.grid(color="#2d2d5e", linewidth=0.5, linestyle="--", alpha=0.5)

    lines = [l for l in ax1.get_lines() + ax2.get_lines()
             if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=7, facecolor="#16213e",
               labelcolor="#e0e0e0", loc="lower right")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Jinja2 template ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Agent Report — {{ dataset_name }}</title>
<style>
  :root {
    --bg:       #1a1a2e;
    --surface:  #16213e;
    --surface2: #0f3460;
    --accent:   #4C9BE8;
    --green:    #4CE87A;
    --orange:   #E8834C;
    --red:      #E84C4C;
    --text:     #e0e0e0;
    --muted:    #888;
    --border:   #2d2d5e;
    --radius:   8px;
    --font:     'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 14px; line-height: 1.6; padding: 32px; }
  a { color: var(--accent); }

  /* ── header ── */
  .header { background: var(--surface); border: 1px solid var(--border);
             border-radius: var(--radius); padding: 24px 32px; margin-bottom: 32px; }
  .header h1 { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
  .header h1 span { color: var(--accent); }
  .meta-grid { display: flex; gap: 32px; flex-wrap: wrap; margin-top: 16px; }
  .meta-item { display: flex; flex-direction: column; gap: 2px; }
  .meta-label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                letter-spacing: 0.05em; }
  .meta-value { font-size: 18px; font-weight: 600; }
  .meta-value.goal-met  { color: var(--green); }
  .meta-value.goal-miss { color: var(--orange); }

  /* ── section title ── */
  h2 { font-size: 16px; font-weight: 600; color: var(--muted);
       text-transform: uppercase; letter-spacing: 0.08em;
       margin: 40px 0 16px; border-bottom: 1px solid var(--border);
       padding-bottom: 8px; }

  /* ── chart ── */
  .chart-wrap { background: var(--surface); border: 1px solid var(--border);
                border-radius: var(--radius); padding: 16px;
                margin-bottom: 32px; text-align: center; }
  .chart-wrap img { max-width: 100%; border-radius: 4px; }

  /* ── run cards ── */
  .run-card { background: var(--surface); border: 1px solid var(--border);
              border-radius: var(--radius); margin-bottom: 24px;
              overflow: hidden; }
  .run-header { display: flex; align-items: center; gap: 16px;
                padding: 14px 20px; border-bottom: 1px solid var(--border);
                background: var(--surface2); }
  .run-iter { font-size: 20px; font-weight: 700; color: var(--accent); min-width: 40px; }
  .run-f1   { font-size: 20px; font-weight: 700; }
  .run-f1.met  { color: var(--green); }
  .run-f1.miss { color: var(--text); }
  .run-status { font-size: 12px; padding: 2px 10px; border-radius: 20px;
                font-weight: 600; }
  .status-done    { background: #1a3a2a; color: var(--green); }
  .status-oom     { background: #3a2a1a; color: var(--orange); }
  .status-crashed { background: #3a1a1a; color: var(--red); }
  .run-id { color: var(--muted); font-size: 12px; font-family: monospace; margin-left: auto; }
  .run-body { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  .run-section { padding: 16px 20px; border-right: 1px solid var(--border); }
  .run-section:last-child { border-right: none; }
  .run-section h3 { font-size: 11px; color: var(--muted); text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 8px; }
  .run-section p  { font-size: 13px; color: var(--text); line-height: 1.5; }
  .verdict-correct { color: var(--green); font-weight: 600; }
  .verdict-wrong   { color: var(--red);   font-weight: 600; }
  .verdict-pending { color: var(--muted); }

  /* ── config table ── */
  .config-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .config-table td { padding: 5px 10px; border-bottom: 1px solid var(--border); }
  .config-table td:first-child { color: var(--muted); width: 42%; }
  .config-table td:last-child  { font-family: monospace; color: #b0d4ff; }

  /* ── curve chart ── */
  .curve-wrap { padding: 8px 20px 16px; border-top: 1px solid var(--border); }
  .curve-wrap img { max-width: 100%; border-radius: 4px; }

  /* ── footer ── */
  .footer { margin-top: 48px; text-align: center; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────────── -->
<div class="header">
  <h1>Self-Improving ML Agent &nbsp;·&nbsp; <span>{{ dataset_name }}</span></h1>
  <div class="meta-grid">
    <div class="meta-item">
      <span class="meta-label">Best Val F1</span>
      <span class="meta-value {% if best_f1 >= goal_f1 %}goal-met{% else %}goal-miss{% endif %}">
        {{ "%.4f"|format(best_f1) }}
      </span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Goal</span>
      <span class="meta-value">≥ {{ goal_f1 }}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Total Runs</span>
      <span class="meta-value">{{ runs|length }}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Model</span>
      <span class="meta-value">{{ model_name }}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Goal Met</span>
      <span class="meta-value {% if best_f1 >= goal_f1 %}goal-met{% else %}goal-miss{% endif %}">
        {% if best_f1 >= goal_f1 %}✓ Yes{% else %}✗ Not yet{% endif %}
      </span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Generated</span>
      <span class="meta-value" style="font-size:14px">{{ generated_at }}</span>
    </div>
  </div>
</div>

<!-- ── F1 Progression Chart ───────────────────────────────────────────── -->
<h2>F1 Progression</h2>
<div class="chart-wrap">
  <img src="{{ progression_chart }}" alt="F1 progression chart">
</div>

<!-- ── Per-run cards ─────────────────────────────────────────────────── -->
<h2>Run Details</h2>
{% for run in runs %}
<div class="run-card">

  <div class="run-header">
    <span class="run-iter">#{{ run.iteration }}</span>
    <span class="run-f1 {% if run.best_f1 >= goal_f1 %}met{% else %}miss{% endif %}">
      F1 {{ "%.4f"|format(run.best_f1) }}
    </span>
    <span class="run-status status-{{ run.status }}">{{ run.status }}</span>
    <span class="run-id">{{ run.run_id }}</span>
  </div>

  <div class="run-body">
    <div class="run-section">
      <h3>Rationale</h3>
      <p>{{ run.rationale }}</p>
    </div>
    <div class="run-section">
      <h3>
        Hypothesis &nbsp;
        {% if run.hypothesis_correct == 1 %}
          <span class="verdict-correct">✓ Correct</span>
        {% elif run.hypothesis_correct == 0 %}
          <span class="verdict-wrong">✗ Off</span>
        {% else %}
          <span class="verdict-pending">— not assessed</span>
        {% endif %}
      </h3>
      <p>{{ run.hypothesis }}</p>
    </div>
  </div>

  <div style="padding: 0 20px 16px; border-top: 1px solid var(--border);">
    <h3 style="font-size:11px; color:var(--muted); text-transform:uppercase;
               letter-spacing:.08em; margin: 14px 0 8px;">Config</h3>
    <table class="config-table">
      <tr><td>Optimizer</td>     <td>{{ run.optimizer }}  lr={{ run.learning_rate }}</td></tr>
      <tr><td>Batch / Epochs</td><td>{{ run.batch_size }} / {{ run.epochs }}</td></tr>
      <tr><td>Scheduler</td>     <td>{{ run.scheduler }}</td></tr>
      <tr><td>Augmentations</td> <td>{{ run.augmentations_list|join(', ') or 'none' }}</td></tr>
      <tr><td>Class weights</td> <td>{{ run.class_weights }}</td></tr>
      <tr><td>Freeze backbone</td><td>{{ run.freeze_backbone }}  unfreeze_at={{ run.unfreeze_after_epoch }}</td></tr>
      <tr><td>Architecture</td>  <td>{{ run.architecture }}</td></tr>
    </table>
  </div>

  {% if run.curve_chart %}
  <div class="curve-wrap">
    <img src="{{ run.curve_chart }}" alt="epoch curve run {{ run.iteration }}">
  </div>
  {% endif %}

</div>
{% endfor %}

<div class="footer">
  Generated by self-tuning-agent &nbsp;·&nbsp; {{ generated_at }}
</div>
</body>
</html>
"""


# ── ReportGenerator ───────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Reads all runs from SQLite, renders an HTML report, and writes it to
    outputs/report.html.

    Usage:
        ReportGenerator().generate()   # uses defaults from config.py
        ReportGenerator().generate(db_path=Path("alt.db"), output_dir=Path("."))
    """

    def generate(
        self,
        db_path:    Path = DB_PATH,
        output_dir: Path = OUTPUTS_DIR,
    ) -> Path:
        """
        Build and write the report. Returns the path to the HTML file.
        Raises RuntimeError if the database is empty or missing.
        """
        runs = self._load_runs(db_path)
        if not runs:
            raise RuntimeError(
                f"No runs found in {db_path}. "
                "Run the agent at least once before generating a report."
            )

        best_f1 = max(r["best_f1"] for r in runs)

        logger.info("Generating F1 progression chart...")
        progression_chart = _f1_progression_chart(runs)

        logger.info("Generating per-run epoch curves...")
        run_dicts = []
        for r in runs:
            history = json.loads(r["history"] or "[]")
            curve   = _epoch_curve_chart(history, r["run_id"]) if history else ""
            rd = dict(r)
            rd["augmentations_list"] = json.loads(r["augmentations"] or "[]")
            rd["curve_chart"]        = curve
            rd["class_weights"]      = bool(r["class_weights"])
            rd["freeze_backbone"]    = bool(r["freeze_backbone"])
            run_dicts.append(rd)

        env      = Environment(loader=BaseLoader())
        template = env.from_string(_HTML_TEMPLATE)

        html = template.render(
            dataset_name       = DATASET_NAME,
            model_name         = MODEL_NAME,
            goal_f1            = GOAL_F1,
            best_f1            = best_f1,
            runs               = run_dicts,
            progression_chart  = progression_chart,
            generated_at       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "report.html"
        out_path.write_text(html, encoding="utf-8")

        logger.info("Report written to %s", out_path)
        return out_path

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_runs(db_path: Path) -> list[sqlite3.Row]:
        if not db_path.exists():
            raise RuntimeError(f"Database not found: {db_path}")
        with _db(db_path) as conn:
            return conn.execute(
                "SELECT * FROM runs ORDER BY iteration ASC"
            ).fetchall()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    try:
        gen  = ReportGenerator()
        path = gen.generate()
        print(f"\nReport written → {path}")
        print("Open it in a browser to review all runs.")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
