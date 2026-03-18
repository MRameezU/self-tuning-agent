# self-tuning-agent

> Give it a dataset and a goal. It figures out the rest.

An autonomous ML experiment agent that fine-tunes an image classification model, reasons about its results in natural language, and iterates until it hits a target F1 score — without any human intervention between runs.

Unlike AutoML or Optuna, this agent doesn't blindly search. It **reads its own results, writes a diagnosis, forms a hypothesis, and proposes the next experiment** based on that reasoning. Every decision is logged with the *why*, not just the *what*.

---

## Demo

```
$ python agent.py

╭─────────────────────────────────────────────────────╮
│  Self-Improving ML Agent  │  Goal: val_f1 ≥ 0.90    │
│  Model: EfficientNetV2-S  │  Budget: 12 runs         │
╰─────────────────────────────────────────────────────╯

━━━━━━━━━━━━━━━━━━━━  Iteration 1 / 12  ━━━━━━━━━━━━━━━━━━━━

⟳ Agent reasoning...

╭───  Agent's Plan for Run 1 ───────────────────────────────────────────────╮
│ Starting with a baseline fine-tune to establish a performance floor.         │
│ No class weighting yet — I want to see the raw imbalance effect first.       │
│                                                                               │
│ Hypothesis: expect val_f1 ~0.78. Normal class recall will likely be poor     │
│ due to 3:1 class imbalance. This will inform weighting in the next run.      │
│                                                                               │
│ Architecture:  EfficientNetV2-S, dropout=0.2, backbone frozen               │
│ Optimizer:     AdamW  lr=0.001                                               │
│ Augmentations: RandomHorizontalFlip, RandomRotation                          │
╰──────────────────────────────────────────────────────────────────────────────╯

Training EfficientNetV2-S... ━━━━━━━━━━━━━━━━  12/15  val_f1=0.7823

...

✓ Goal achieved in 6 runs!  Best val_f1: 0.9134
```

---

## What Makes This Different

| | AutoML / Optuna | This agent |
|---|---|---|
| Search strategy | Blind / random | Reasoned, hypothesis-driven |
| Explainability | None | Full natural-language trace |
| Diagnoses failures | No | Yes — reads metrics, explains what went wrong |
| Hypothesis tracking | No | Logs prediction vs. outcome each run |
| Demo-able | Not really | Yes — the terminal output *is* the demo |

---

## Architecture

```
agent.py          ← orchestrator — the entry point
llm_core.py       ← Ollama wrapper, ExperimentProposal parsing
trainer.py        ← subprocess sandbox, PyTorch training template
memory.py         ← SQLite agent memory + MLflow logging
visualizer.py     ← live matplotlib loss plot (separate thread)
report.py         ← Jinja2 HTML/Markdown report generation
config.py         ← constants, search space, paths
```

**Core loop:**
```python
for iteration in range(max_runs):
    context  = memory.build_context()   # read history → format for LLM
    proposal = llm.propose(context)     # LLM reasons → ExperimentProposal
    results  = runner.execute(proposal) # subprocess trains → streams metrics
    memory.save(proposal, results)      # SQLite + MLflow
    if results.best_f1 >= goal: break
```

That's it. No LangGraph, no AutoGen, no CrewAI — just a while loop.

---

## Tech Stack

**LLM layer:** Ollama (local) · Qwen2.5-Coder-32B or DeepSeek-R1:8B · Pydantic

**Training layer:** PyTorch · EfficientNetV2-S · Albumentations (CLAHE) · scikit-learn

**Tracking layer:** MLflow · SQLite · ONNX export

**Terminal UI:** Rich · Matplotlib (live plot) · Jinja2 (report)

---

## Requirements

- Python 3.10+
- CUDA-capable GPU with 8GB+ VRAM (tested on RTX 4070 Super 12GB)
- [Ollama](https://ollama.ai) installed and running

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/self-tuning-agent.git
cd self-tuning-agent
pip install -r requirements.txt
```

### 2. Pull the reasoning model

```bash
# Primary — best reasoning quality
ollama pull qwen2.5-coder:32b

# Alternative — visible <think> chain-of-thought, great for demos
ollama pull deepseek-r1:8b
```

### 3. Download the dataset

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

Expected structure:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env — set OLLAMA_HOST, DATA_PATH, GOAL_F1
```

### 5. Run

```bash
# Start MLflow UI (optional, opens at localhost:5000)
mlflow ui &

# Launch the agent
python agent.py
```

---

## Configuration

```env
# .env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:32b
DATA_PATH=./data/chest_xray
GOAL_F1=0.90
MAX_RUNS=12
```

---

## Hyperparameter Search Space

The agent reasons over this space — it doesn't enumerate it blindly:

```python
SEARCH_SPACE = {
    "learning_rate":        [1e-4, 3e-4, 5e-4, 1e-3],
    "batch_size":           [32, 64],
    "epochs":               [10, 15, 20],
    "dropout":              [0.2, 0.3, 0.4, 0.5],
    "optimizer":            ["AdamW", "SGD"],
    "scheduler":            ["CosineAnnealing", "StepLR", "OneCycleLR"],
    "class_weights":        [True, False],
    "augmentations":        ["RandomHorizontalFlip", "RandomRotation",
                             "ColorJitter", "RandomAffine", "GaussianBlur", "CLAHE"],
    "freeze_backbone":      [True, False],
    "unfreeze_after_epoch": [0, 3, 5],
}
```

---

## Expected Improvement Arc

| Run | Agent action | Expected F1 |
|-----|-------------|-------------|
| 1 | Baseline fine-tune, no class weights | ~0.78 |
| 2 | Adds class weights after diagnosing Normal recall | ~0.83 |
| 3 | Aggressive augmentation + higher dropout | ~0.85 |
| 4 | Unfreezes backbone, switches to OneCycleLR | ~0.87 |
| 5–7 | LR refinement, CLAHE augmentation | ~0.90+ |

---

## Output

After a completed run you get:

- **MLflow dashboard** — all runs with metrics, params, and comparison view
- **experiments.db** — full reasoning trace queryable via SQL
- **outputs/best_model.onnx** — production-ready model export
- **report.html** — per-run narrative: what the agent tried, why, and what happened

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built by a lead ML engineer as a demonstration of agentic reasoning applied to real model optimization. The goal was not to build another AutoML tool — it was to build something that thinks out loud.
