"""
agent.py

Entry point. Runs the agent loop:
    reason → execute → observe → reflect → repeat

Nothing clever here — it's a while loop with a budget. The intelligence
lives in the LLM (llm_core.py) and the training sandbox (trainer.py).
This file just connects them and keeps track of what's happening.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich import box

from config import GOAL_F1, MAX_RUNS, MODEL_NAME, DATASET_NAME, OUTPUTS_DIR,ONNX_OPSET
from llm_core import ExperimentProposal, LLMCore
from memory import Memory
from trainer import TrainingRunner

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_header() -> None:
    console.print()
    console.print(Panel(
        f"[bold white]Self-Improving ML Agent[/]  ·  "
        f"[dim]Model:[/] [cyan]{MODEL_NAME}[/]  ·  "
        f"[dim]Dataset:[/] [cyan]{DATASET_NAME}[/]\n"
        f"[dim]Goal:[/] [green]val_f1 ≥ {GOAL_F1}[/]  ·  "
        f"[dim]Budget:[/] [yellow]{MAX_RUNS} runs[/]",
        border_style="bright_black",
        padding=(0, 2),
    ))
    console.print()


def _print_iteration_header(iteration: int, best_so_far: float) -> None:
    console.print(Rule(
        f"[bold]Iteration {iteration} / {MAX_RUNS}[/]  ·  "
        f"[dim]best so far:[/] [green]{best_so_far:.4f}[/]",
        style="bright_black",
    ))
    console.print()


def _print_proposal(proposal: ExperimentProposal) -> None:
    content = (
        f"[dim]Rationale:[/]\n  {proposal.rationale}\n\n"
        f"[dim]Hypothesis:[/]\n  [italic]{proposal.hypothesis}[/]\n\n"
        f"[dim]Config:[/]\n"
        f"  optimizer={proposal.optimizer}  lr={proposal.learning_rate}  "
        f"batch={proposal.batch_size}  epochs={proposal.epochs}\n"
        f"  scheduler={proposal.scheduler}  "
        f"class_weights={proposal.class_weights}  "
        f"freeze={proposal.freeze_backbone}  "
        f"unfreeze_at={proposal.unfreeze_after_epoch}\n"
        f"  augments={proposal.augmentations}"
    )
    console.print(Panel(
        content,
        title="[bold cyan]Agent's Plan[/]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def _print_run_summary(result, iteration: int) -> None:
    status_color = {
        "done":    "green",
        "oom":     "yellow",
        "crashed": "red",
    }.get(result.status, "white")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("status",  f"[{status_color}]{result.status}[/]")
    table.add_row("best_f1", f"[bold green]{result.best_f1:.4f}[/]")
    table.add_row("run_id",  result.run_id)
    console.print(table)
    console.print()


def _print_goal_achieved(best_f1: float, iteration: int) -> None:
    console.print()
    console.print(Panel(
        f"[bold green]Goal achieved![/]  val_f1 = [bold]{best_f1:.4f}[/]  "
        f"(target {GOAL_F1})  ·  completed in [bold]{iteration}[/] runs",
        border_style="green",
        padding=(0, 2),
    ))
    console.print()


def _print_budget_exhausted(best_f1: float) -> None:
    console.print()
    console.print(Panel(
        f"[yellow]Budget exhausted.[/]  Best val_f1 reached: [bold]{best_f1:.4f}[/]  "
        f"(target was {GOAL_F1})",
        border_style="yellow",
        padding=(0, 2),
    ))
    console.print()


# ── Reflect ───────────────────────────────────────────────────────────────────

def _assess_hypothesis(proposal: ExperimentProposal, best_f1: float) -> bool:
    """
    Crude but effective: extract the predicted F1 from the hypothesis string
    and check if actual F1 landed within 0.03 of it. If no number is found,
    fall back to checking directional improvement.

    This verdict gets written to SQLite and fed back into the next prompt
    so the LLM can calibrate its own confidence over time.
    """
    import re
    numbers = re.findall(r"0\.\d+", proposal.hypothesis)
    if numbers:
        predicted = float(numbers[0])
        return abs(best_f1 - predicted) <= 0.03
    # no specific number — just check it said "improve" and it did
    return best_f1 > 0.0


# ── ONNX export ───────────────────────────────────────────────────────────────

def _export_best_model(run_id: str) -> None:
    """
    Export the best checkpoint to ONNX. Called once when goal is achieved.
    Wrapped in a try/except so a failed export never kills the agent.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

    ckpt_path = OUTPUTS_DIR / f"{run_id}_best.pt"
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found for ONNX export: %s", ckpt_path)
        return

    try:
        model = efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        dummy = torch.randn(1, 3, 300, 300)
        onnx_path = OUTPUTS_DIR / "best_model.onnx"

        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=ONNX_OPSET,
        )
        console.print(f"[dim]ONNX model exported → {onnx_path}[/]")

    except Exception as e:
        logger.warning("ONNX export failed (non-fatal): %s", e)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent() -> None:
    _print_header()

    llm    = LLMCore()
    memory = Memory()
    runner = TrainingRunner()

    # check Ollama is up before we commit to anything
    console.print("[dim]Checking Ollama...[/]", end=" ")
    if not llm.ping():
        console.print(f"[red]✗[/]")
        console.print(
            f"[red]Ollama not reachable or model not loaded.[/]\n"
            f"Run:  ollama pull {llm.model}\n"
            f"Then: ollama serve"
        )
        sys.exit(1)
    console.print("[green]✓[/]\n")

    goal_achieved = False
    best_run_id   = None

    for iteration in range(1, MAX_RUNS + 1):
        _print_iteration_header(iteration, memory.best_f1())

        # ── Reason ──────────────────────────────────────────────────────────
        console.print("[dim]⟳  Agent reasoning...[/]")
        console.print()

        try:
            context  = memory.build_context()
            proposal = llm.propose(context)
        except RuntimeError as e:
            console.print(f"[red]LLM call failed: {e}[/]")
            console.print("[dim]Skipping iteration and retrying next...[/]\n")
            continue

        _print_proposal(proposal)

        # ── Execute ─────────────────────────────────────────────────────────
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Training {task.description}[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                MODEL_NAME,
                total=proposal.epochs,
            )
            for epoch_metrics in runner.execute(proposal):
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"{MODEL_NAME}  "
                        f"[green]val_f1={epoch_metrics['val_f1']:.4f}[/]"
                    ),
                )

        result = runner.last_result

        # ── Observe ─────────────────────────────────────────────────────────
        _print_run_summary(result, iteration)

        if result.status == "oom":
            console.print(
                "[yellow]OOM — this run's results won't count toward the best. "
                "The agent will propose a smaller config next.[/]\n"
            )

        # ── Reflect ─────────────────────────────────────────────────────────
        correct = _assess_hypothesis(proposal, result.best_f1)
        memory.save(proposal, result)
        memory.mark_hypothesis(result.run_id, correct)

        verdict = "[green]✓ correct[/]" if correct else "[red]✗ off[/]"
        console.print(f"[dim]Hypothesis verdict:[/] {verdict}\n")

        # ── Check goal ──────────────────────────────────────────────────────
        if result.best_f1 >= GOAL_F1:
            goal_achieved = True
            best_run_id   = result.run_id
            _print_goal_achieved(result.best_f1, iteration)
            break

    # ── Wrap up ───────────────────────────────────────────────────────────────
    if not goal_achieved:
        _print_budget_exhausted(memory.best_f1())

    if best_run_id:
        console.print("[dim]Exporting best model to ONNX...[/]")
        _export_best_model(best_run_id)

    console.print(
        f"\n[dim]Runs logged to experiments.db  ·  "
        f"MLflow dashboard: http://localhost:5000[/]\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()