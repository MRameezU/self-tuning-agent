# quick_test.py — run this once to confirm trainer works
import json
from trainer import TrainingRunner
from llm_core import ExperimentProposal

proposal = ExperimentProposal(
    rationale="smoke test",
    architecture="EfficientNetV2-S, dropout=0.2",
    optimizer="AdamW",
    learning_rate=1e-3,
    batch_size=32,
    epochs=2,           # just 2 epochs to validate the loop
scheduler="CosineAnnealing",
    augmentations=["RandomHorizontalFlip"],
    class_weights=True,
    freeze_backbone=True,
    unfreeze_after_epoch=0,
    hypothesis="smoke test only",
)

runner = TrainingRunner()
for metrics in runner.execute(proposal):
    print(metrics)

print(runner.last_result)