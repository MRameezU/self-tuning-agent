"""
memory.py

Two jobs:
  1. Persist every run to SQLite so the agent has durable memory across
     restarts — proposals, results, epoch history, hypothesis verdicts.
  2. Build the formatted context string the LLM reads at the start of
     each iteration. This is the agent's working memory.

MLflow gets a parallel write of every run for the dashboard — it's purely
for human inspection, the agent itself only reads from SQLite.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import mlflow

from config import DB_PATH, CONTEXT_WINDOW, GOAL_F1, MLFLOW_EXPERIMENT_NAME
from llm_core import ExperimentProposal
from trainer import RunResult

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    iteration           INTEGER NOT NULL,
    created_at          TEXT    NOT NULL,

    -- proposal fields (flattened for easy querying)
    rationale           TEXT,
    architecture        TEXT,
    optimizer           TEXT,
    learning_rate       REAL,
    batch_size          INTEGER,
    epochs              INTEGER,
    scheduler           TEXT,
    augmentations       TEXT,   -- JSON array
    class_weights       INTEGER,
    freeze_backbone     INTEGER,
    unfreeze_after_epoch INTEGER,
    hypothesis          TEXT,

    -- result fields
    best_f1             REAL,
    status              TEXT,
    history             TEXT,   -- JSON array of epoch dicts
    hypothesis_correct  INTEGER -- 1 | 0 | NULL (assessed after run)
);
"""


# ── DB connection ─────────────────────────────────────────────────────────────

@contextmanager
def _db(path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Memory ────────────────────────────────────────────────────────────────────

class Memory:
    """
    Agent's persistent store. One instance lives for the lifetime of a run
    of agent.py and gets passed into the agent loop.

    Usage:
        mem = Memory()
        mem.save(proposal, result)
        context = mem.build_context()   # feed this to LLMCore.propose()
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()
        self._init_mlflow()

    # ── public ────────────────────────────────────────────────────────────────

    def save(self, proposal: ExperimentProposal, result: RunResult) -> None:
        """
        Persist a completed run. Writes to SQLite first, then MLflow.
        MLflow failure is logged but never raises — the agent loop shouldn't
        die because the dashboard had a hiccup.
        """
        iteration = self._next_iteration()
        self._save_sqlite(proposal, result, iteration)

        try:
            self._save_mlflow(proposal, result, iteration)
        except Exception as e:
            logger.warning("MLflow logging failed (non-fatal): %s", e)

    def build_context(self) -> str:
        """
        Format the last N runs into a string the LLM can reason over.
        Each run shows: what was tried, what happened, whether the hypothesis
        was correct. This is the agent's memory — quality here directly
        affects proposal quality.
        """
        runs = self._fetch_recent_runs(CONTEXT_WINDOW)

        if not runs:
            return (
                "No previous runs. This is the first experiment.\n"
                "Start with a sensible baseline — frozen backbone, AdamW, "
                "lr=1e-3, no class weights, minimal augmentation."
            )

        best_f1_overall = max(r["best_f1"] for r in runs)
        lines = [
            f"Goal: val_f1 >= {GOAL_F1}",
            f"Best val_f1 so far: {best_f1_overall:.4f}",
            f"Runs in context: {len(runs)}\n",
        ]

        for r in runs:
            augs = json.loads(r["augmentations"] or "[]")
            verdict = _hypothesis_verdict(r["hypothesis_correct"])

            lines += [
                f"--- Run {r['iteration']} (id={r['run_id']}) ---",
                f"  Status:    {r['status']}",
                f"  best_f1:   {r['best_f1']:.4f}",
                f"  Optimizer: {r['optimizer']}  lr={r['learning_rate']}  "
                f"batch={r['batch_size']}  epochs={r['epochs']}",
                f"  Scheduler: {r['scheduler']}",
                f"  Augments:  {', '.join(augs) if augs else 'none'}",
                f"  Weights:   class_weights={bool(r['class_weights'])}  "
                f"freeze_backbone={bool(r['freeze_backbone'])}  "
                f"unfreeze_at={r['unfreeze_after_epoch']}",
                f"  Rationale: {r['rationale']}",
                f"  Hypothesis:{r['hypothesis']}",
                f"  Verdict:   {verdict}",
            ]

            # show the epoch curve so the LLM can spot trends
            history = json.loads(r["history"] or "[]")
            if history:
                curve = "  Curve:     " + " → ".join(
                    f"e{e['epoch']}={e['val_f1']:.3f}" for e in history
                )
                lines.append(curve)

            lines.append("")  # blank line between runs

        return "\n".join(lines)

    def best_f1(self) -> float:
        """Best val_f1 across all completed runs. Returns 0.0 if no runs yet."""
        with _db(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(best_f1) FROM runs WHERE status = 'done'"
            ).fetchone()
            return float(row[0]) if row[0] is not None else 0.0

    def run_count(self) -> int:
        """Total number of runs recorded (any status)."""
        with _db(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            return int(row[0])

    def mark_hypothesis(self, run_id: str, correct: bool) -> None:
        """
        Record whether the agent's hypothesis for a run turned out to be true.
        Called by agent.py after comparing the predicted vs actual F1.
        """
        with _db(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET hypothesis_correct = ? WHERE run_id = ?",
                (int(correct), run_id),
            )

    # ── private ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with _db(self.db_path) as conn:
            conn.executescript(_SCHEMA)
        logger.debug("SQLite ready at %s", self.db_path)

    def _init_mlflow(self) -> None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.debug("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)

    def _next_iteration(self) -> int:
        with _db(self.db_path) as conn:
            row = conn.execute("SELECT MAX(iteration) FROM runs").fetchone()
            return (row[0] or 0) + 1

    def _save_sqlite(
        self,
        proposal: ExperimentProposal,
        result:   RunResult,
        iteration: int,
    ) -> None:
        p = asdict(proposal)
        with _db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, iteration, created_at,
                    rationale, architecture, optimizer, learning_rate,
                    batch_size, epochs, scheduler, augmentations,
                    class_weights, freeze_backbone, unfreeze_after_epoch,
                    hypothesis, best_f1, status, history
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    result.run_id,
                    iteration,
                    datetime.utcnow().isoformat(),
                    p["rationale"],
                    p["architecture"],
                    p["optimizer"],
                    p["learning_rate"],
                    p["batch_size"],
                    p["epochs"],
                    p["scheduler"],
                    json.dumps(p["augmentations"]),
                    int(p["class_weights"]),
                    int(p["freeze_backbone"]),
                    p["unfreeze_after_epoch"],
                    p["hypothesis"],
                    result.best_f1,
                    result.status,
                    json.dumps(result.history),
                ),
            )
        logger.debug("Saved run %s (iter %d) to SQLite", result.run_id, iteration)

    def _save_mlflow(
        self,
        proposal:  ExperimentProposal,
        result:    RunResult,
        iteration: int,
    ) -> None:
        p = asdict(proposal)
        with mlflow.start_run(run_name=f"iter_{iteration:02d}_{result.run_id}"):
            # params — everything the agent chose
            mlflow.log_params({
                "iteration":           iteration,
                "optimizer":           p["optimizer"],
                "learning_rate":       p["learning_rate"],
                "batch_size":          p["batch_size"],
                "epochs":              p["epochs"],
                "scheduler":           p["scheduler"],
                "augmentations":       ",".join(p["augmentations"]),
                "class_weights":       p["class_weights"],
                "freeze_backbone":     p["freeze_backbone"],
                "unfreeze_after_epoch":p["unfreeze_after_epoch"],
            })

            # per-epoch metrics — gives MLflow the full loss curve
            for ep in result.history:
                mlflow.log_metrics(
                    {
                        "train_loss": ep["train_loss"],
                        "val_loss":   ep["val_loss"],
                        "val_f1":     ep["val_f1"],
                    },
                    step=ep["epoch"],
                )

            # summary
            mlflow.log_metrics({
                "best_f1": result.best_f1,
            })

            mlflow.set_tags({
                "status":     result.status,
                "run_id":     result.run_id,
                "rationale":  p["rationale"][:250],  # MLflow tag limit
                "hypothesis": p["hypothesis"][:250],
            })

    def _fetch_recent_runs(self, n: int) -> list[sqlite3.Row]:
        with _db(self.db_path) as conn:
            return conn.execute(
                """
                SELECT * FROM runs
                ORDER BY iteration DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hypothesis_verdict(correct: Optional[int]) -> str:
    if correct is None:
        return "not assessed"
    return "✓ correct" if correct else "✗ incorrect"


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # use a temp DB so the smoke test doesn't pollute experiments.db
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    print(f"Testing Memory with temp DB: {test_db}\n")

    mem = Memory(db_path=test_db)

    # simulate two runs
    from llm_core import ExperimentProposal
    from trainer import RunResult

    proposal_1 = ExperimentProposal(
        rationale="Baseline — want to see raw imbalance effect",
        architecture="EfficientNetV2-S, dropout=0.2",
        optimizer="AdamW",
        learning_rate=1e-3,
        batch_size=32,
        epochs=10,
        scheduler="CosineAnnealing",
        augmentations=["RandomHorizontalFlip"],
        class_weights=False,
        freeze_backbone=True,
        unfreeze_after_epoch=0,
        hypothesis="Expect ~0.78 F1. Normal recall will be poor due to 3:1 imbalance.",
    )
    result_1 = RunResult(
        run_id="aabbccdd",
        proposal=proposal_1,
        history=[
            {"epoch": 1, "train_loss": 0.42, "val_loss": 0.55, "val_f1": 0.74},
            {"epoch": 2, "train_loss": 0.35, "val_loss": 0.50, "val_f1": 0.78},
        ],
        best_f1=0.78,
        status="done",
    )

    mem.save(proposal_1, result_1)
    mem.mark_hypothesis("aabbccdd", correct=True)

    proposal_2 = ExperimentProposal(
        rationale="Run 1 confirmed imbalance — adding class weights and more augmentation",
        architecture="EfficientNetV2-S, dropout=0.3",
        optimizer="AdamW",
        learning_rate=3e-4,
        batch_size=32,
        epochs=15,
        scheduler="CosineAnnealing",
        augmentations=["RandomHorizontalFlip", "RandomRotation", "ColorJitter"],
        class_weights=True,
        freeze_backbone=True,
        unfreeze_after_epoch=0,
        hypothesis="Expect ~0.83 F1. Class weights should fix Normal recall.",
    )
    result_2 = RunResult(
        run_id="11223344",
        proposal=proposal_2,
        history=[
            {"epoch": 1, "train_loss": 0.31, "val_loss": 0.44, "val_f1": 0.80},
            {"epoch": 2, "train_loss": 0.27, "val_loss": 0.41, "val_f1": 0.83},
        ],
        best_f1=0.83,
        status="done",
    )

    mem.save(proposal_2, result_2)

    print("\n" + "=" * 60)
    print("build_context() output:\n")
    print(mem.build_context())
    print("=" * 60)
    print(f"\nrun_count(): {mem.run_count()}")
    print(f"best_f1():   {mem.best_f1()}")
    print("\nPhase 3 smoke test passed.")

    test_db.unlink()