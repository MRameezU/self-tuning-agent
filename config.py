

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent
DATA_PATH   = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "chest_xray"))
OUTPUTS_DIR = ROOT_DIR / "outputs"
DB_PATH     = ROOT_DIR / "experiments.db"

OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Agent settings ────────────────────────────────────────────────────────────

GOAL_F1  = float(os.getenv("GOAL_F1", 0.90))
MAX_RUNS = int(os.getenv("MAX_RUNS", 12))

# how many past runs we feed to the LLM as context
# 8 is enough — any more and you're just padding the prompt
CONTEXT_WINDOW = 8

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")

# keep it low — we want consistent JSON, not creative writing
OLLAMA_TEMPERATURE = 0.3

# 32b at Q4_K_M can be slow on first token. 120s is generous but avoids
# false timeouts mid-reasoning on a loaded machine.
OLLAMA_TIMEOUT = 120

# ── Training defaults ─────────────────────────────────────────────────────────

# these are just fallbacks — the agent overrides them every run
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS     = 10
DEFAULT_LR         = 1e-3

# ── Hyperparameter search space ───────────────────────────────────────────────
# The agent reasons over this — it doesn't enumerate it blindly.
# Adding something here makes it available to the LLM context automatically.

SEARCH_SPACE: dict[str, list] = {
    "learning_rate":        [1e-4, 3e-4, 5e-4, 1e-3],
    "batch_size":           [32, 64],
    "epochs":               [10, 15, 20],
    "dropout":              [0.2, 0.3, 0.4, 0.5],
    "optimizer":            ["AdamW", "SGD"],
    "scheduler":            ["CosineAnnealing", "StepLR", "OneCycleLR"],
    "class_weights":        [True, False],
    "augmentations":        [
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "RandomAffine",
        "GaussianBlur",
        "CLAHE",             # critical for X-ray contrast
    ],
    "freeze_backbone":      [True, False],
    "unfreeze_after_epoch": [0, 3, 5],
}

# ── MLflow ────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_NAME = "chest-xray-agent"

# ── Misc ──────────────────────────────────────────────────────────────────────

# printed in the Rich header on startup
MODEL_NAME   = "EfficientNetV2-S"
DATASET_NAME = "Chest X-Ray Pneumonia"
NUM_CLASSES  = 2