
import json
import sys
import uuid
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ── RunResult ─────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Everything the agent needs to know about a completed training run."""
    run_id:   str
    proposal: "ExperimentProposal"        # imported lazily to avoid torch in agent
    history:  list[dict]                  # [{epoch, train_loss, val_loss, val_f1}, ...]
    best_f1:  float
    status:   str                         # "done" | "oom" | "crashed"


# ── TrainingRunner (used by agent.py) ─────────────────────────────────────────

class TrainingRunner:
    """
    Launches trainer.py as a child process, streams epoch metrics back to
    the caller as dicts, and returns a RunResult when the run finishes.

    Usage:
        runner = TrainingRunner()
        for epoch_metrics in runner.execute(proposal):
            print(epoch_metrics)   # {epoch, train_loss, val_loss, val_f1}
        result = runner.last_result
    """

    def __init__(self) -> None:
        self.last_result: Optional[RunResult] = None

    def execute(
        self, proposal: "ExperimentProposal"
    ) -> Generator[dict, None, None]:
        """
        Yields one dict per completed epoch. After the generator is exhausted,
        self.last_result is populated with the full RunResult.
        """
        import subprocess
        from llm_core import ExperimentProposal

        run_id   = uuid.uuid4().hex[:8]
        proposal_json = json.dumps(asdict(proposal))

        proc = subprocess.Popen(
            [sys.executable, __file__, proposal_json, run_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        history: list[dict] = []
        status  = "crashed"

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Non-JSON stdout from trainer: %s", line)
                continue

            msg_type = msg.get("type")

            if msg_type == "epoch":
                history.append(msg["data"])
                yield msg["data"]

            elif msg_type == "status":
                status = msg["value"]

            elif msg_type == "error":
                logger.error("Trainer error: %s", msg.get("message"))

        proc.wait()

        stderr_output = proc.stderr.read().strip()
        if stderr_output:
            logger.debug("Trainer stderr:\n%s", stderr_output)

        # subprocess can exit non-zero without emitting a status line
        if proc.returncode == 2:
            status = "oom"
        elif proc.returncode != 0 and status not in ("done", "oom"):
            status = "crashed"

        best_f1 = max((e["val_f1"] for e in history), default=0.0)

        self.last_result = RunResult(
            run_id=run_id,
            proposal=proposal,
            history=history,
            best_f1=best_f1,
            status=status,
        )


# ── Training subprocess (executed directly) ───────────────────────────────────

def _emit(msg: dict) -> None:
    """Write a JSON line to stdout and flush immediately so the parent reads it."""
    print(json.dumps(msg), flush=True)


def _emit_epoch(epoch: int, train_loss: float, val_loss: float, val_f1: float) -> None:
    _emit({"type": "epoch", "data": {
        "epoch": epoch,
        "train_loss": round(train_loss, 5),
        "val_loss":   round(val_loss,   5),
        "val_f1":     round(val_f1,     5),
    }})


def _emit_status(value: str) -> None:
    _emit({"type": "status", "value": value})


def _emit_error(message: str) -> None:
    _emit({"type": "error", "message": message})


def _build_transforms(augmentation_names: list[str], training: bool):
    """
    Builds an Albumentations pipeline for the given list of augmentation names.
    Always ends with normalisation to ImageNet stats (EfficientNetV2-S was
    pretrained on ImageNet, so this is non-negotiable).

    CLAHE is particularly effective for X-rays — it enhances local contrast in
    the lung fields and makes features more visible to the backbone.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    aug_map = {
        "RandomHorizontalFlip": A.HorizontalFlip(p=0.5),
        "RandomRotation":       A.Rotate(limit=15, p=0.5),
        "ColorJitter":          A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
        "RandomAffine":         A.Affine(translate_percent=0.05, scale=(0.9, 1.1), p=0.4),
        "GaussianBlur":         A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        "CLAHE":                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    }

    transforms = []

    if training:
        for name in augmentation_names:
            if name in aug_map:
                transforms.append(aug_map[name])
            else:
                logger.warning("Unknown augmentation '%s' — skipping", name)

    transforms += [
        A.Resize(300, 300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def _build_loaders(data_path: Path, proposal: dict, num_workers: int = 4):
    """
    Builds train and val DataLoaders from the chest_xray folder structure.
    Chest X-rays are grayscale — they're converted to 3-channel RGB here
    so they match the 3-channel input EfficientNetV2-S expects.
    """
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class XRayDataset(Dataset):
        def __init__(self, root: Path, transform) -> None:
            self.samples:   list[tuple[Path, int]] = []
            self.transform = transform

            for label_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
                class_dir = root / class_name
                if not class_dir.exists():
                    raise FileNotFoundError(
                        f"Expected class directory not found: {class_dir}\n"
                        f"Check DATA_PATH in your .env — should point to "
                        f"the folder containing train/ and val/."
                    )
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.samples.append((img_path, label_idx))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            augmented = self.transform(image=img_np)
            return augmented["image"], label

    train_transform = _build_transforms(proposal["augmentations"], training=True)
    val_transform   = _build_transforms([], training=False)

    train_dataset = XRayDataset(data_path / "train", train_transform)
    val_dataset   = XRayDataset(data_path / "test",   val_transform)

    # class weights for imbalanced dataset — ~3:1 pneumonia to normal
    # inverse frequency weighting: normal class gets 3x the penalty
    if proposal["class_weights"]:
        counts = [0, 0]
        for _, label in train_dataset.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [total / (2 * c) for c in counts]
        sample_weights = [weights[label] for _, label in train_dataset.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=proposal["batch_size"],
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=proposal["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=proposal["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def _build_model(proposal: dict, device):
    """
    Loads EfficientNetV2-S with ImageNet weights, replaces the classifier head,
    and applies dropout. Backbone freezing is handled here — unfreeze logic
    lives in the training loop.
    """
    import torch.nn as nn
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    if proposal["freeze_backbone"]:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=proposal.get("dropout", 0.2)),
        nn.Linear(in_features, 2),
    )

    return model.to(device)


def _build_optimizer(model, proposal: dict, backbone_unfrozen: bool = False):
    """
    Build the optimizer with differential learning rates when the backbone
    is unfrozen.

    When backbone_unfrozen=True the backbone gets lr/10. This prevents the
    catastrophic val loss spike that occurs when ~20M newly-unfrozen parameters
    are updated at the full head learning rate — observed empirically as a
    val_loss jump from ~0.27 to ~1.30 in a single epoch.

    When the backbone is still frozen, only classifier params have
    requires_grad=True, so a flat LR is fine.
    """
    import torch.optim as optim

    lr   = proposal["learning_rate"]
    name = proposal["optimizer"]

    if backbone_unfrozen:
        # differential LR — backbone gets 10x lower rate than the head
        param_groups = [
            {"params": model.features.parameters(),   "lr": lr / 10},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    else:
        # backbone is frozen — only optimise params that require grad
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if name == "AdamW":
        return optim.AdamW(param_groups, weight_decay=1e-4)
    elif name == "SGD":
        return optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(optimizer, proposal: dict, steps_per_epoch: int):
    import torch.optim.lr_scheduler as sched

    name   = proposal["scheduler"]
    epochs = proposal["epochs"]

    if name == "CosineAnnealing":
        return sched.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "StepLR":
        return sched.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    elif name == "OneCycleLR":
        return sched.OneCycleLR(
            optimizer,
            max_lr=proposal["learning_rate"] * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def _train_epoch(model, loader, criterion, optimizer, scheduler, device, is_onecycle: bool):
    import torch

    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if is_onecycle:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def _val_epoch(model, loader, criterion, device):
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    total_loss  = 0.0
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    val_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, float(val_f1)


def run_training(proposal_dict: dict, run_id: str) -> None:
    """
    Full training loop. Executed inside the subprocess.
    Streams JSON lines to stdout. Exits with:
        0 — completed normally
        1 — unexpected crash
        2 — OOM
    """
    import torch
    import torch.nn as nn

    from config import DATA_PATH, OUTPUTS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_loader, val_loader = _build_loaders(DATA_PATH, proposal_dict)
        model     = _build_model(proposal_dict, device)
        optimizer = _build_optimizer(model, proposal_dict, backbone_unfrozen=False)
        scheduler = _build_scheduler(optimizer, proposal_dict, len(train_loader))
        criterion = nn.CrossEntropyLoss()

        epochs          = proposal_dict["epochs"]
        unfreeze_at     = proposal_dict["unfreeze_after_epoch"]
        is_onecycle     = proposal_dict["scheduler"] == "OneCycleLR"
        backbone_frozen = proposal_dict["freeze_backbone"]
        best_f1         = 0.0
        best_ckpt       = OUTPUTS_DIR / f"{run_id}_best.pt"

        for epoch in range(1, epochs + 1):

            # progressive unfreezing — rebuild optimizer with differential LR
            # so the backbone gets lr/10 instead of the full head rate.
            # this prevents the val_loss spike caused by updating ~20M freshly
            # unfrozen params at the same rate as the small classifier head.
            if backbone_frozen and unfreeze_at > 0 and epoch == unfreeze_at:
                for param in model.features.parameters():
                    param.requires_grad = True
                backbone_frozen = False
                optimizer = _build_optimizer(
                    model, proposal_dict, backbone_unfrozen=True
                )
                scheduler = _build_scheduler(
                    optimizer, proposal_dict, len(train_loader)
                )
                logger.info(
                    "Epoch %d: backbone unfrozen with differential LR "
                    "(backbone=%.2e, head=%.2e)",
                    epoch,
                    proposal_dict["learning_rate"] / 10,
                    proposal_dict["learning_rate"],
                )

            train_loss           = _train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, is_onecycle
            )
            val_loss, val_f1     = _val_epoch(model, val_loader, criterion, device)

            if not is_onecycle:
                scheduler.step()

            _emit_epoch(epoch, train_loss, val_loss, val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_ckpt)

        _emit_status("done")
        sys.exit(0)

    except torch.cuda.OutOfMemoryError:
        _emit_error("CUDA out of memory — reduce batch_size or unfreeze fewer layers")
        _emit_status("oom")
        sys.exit(2)

    except Exception as exc:
        _emit_error(str(exc))
        _emit_status("crashed")
        sys.exit(1)


# ── Entry point when subprocess'd ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python trainer.py '<proposal_json>' <run_id>")
        sys.exit(1)

    proposal_raw = json.loads(sys.argv[1])
    run_id_arg   = sys.argv[2]

    run_training(proposal_raw, run_id_arg)