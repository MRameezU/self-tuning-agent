# Ollama on CPU · PyTorch Training on GPU
### Running Ollama and PyTorch on the same machine without VRAM conflicts

**Who this guide is for:** You installed Ollama via `snap` on Ubuntu, you have a
CUDA-capable GPU, and you want to run PyTorch training jobs that use the full GPU
while Ollama handles LLM inference on CPU.

If you installed Ollama via the official `curl` installer, Ollama does not run as
a system service and this conflict likely does not apply to you.

---

## The Problem

Snap installs Ollama as a background service that partially offloads model layers
onto the GPU by default. With a 32B model this consumes 6GB+ of VRAM before
PyTorch starts, causing OOM crashes mid-training-run.

The fix: disable the snap service and launch Ollama manually with GPU offloading
disabled so PyTorch gets the full VRAM budget.

---

## Hardware this was tested on

- CPU: Intel Core i9-14900K (32 threads)
- RAM: 64GB
- GPU: RTX 4070 Super (12GB VRAM)
- OS: Ubuntu 24.04
- Ollama: installed via snap

The approach works on any Ubuntu machine where Ollama was installed via snap.
The specific VRAM numbers will vary by GPU and model size.

---

## One-time setup

### Step 1 — Set your paths

Before running any of the commands below, set these two variables in your terminal.
Every command in this guide uses them.

```bash
# Path to the Ollama binary inside the snap package.
# 'current' is a stable symlink snap maintains automatically —
# it always points to the active revision regardless of version number.
export OLLAMA_BIN="/snap/ollama/current/bin/ollama"

# Absolute path to your project directory.
export PROJECT_DIR="$HOME/self-tuning-agent"
```

To make these permanent, add both lines to your `~/.bashrc` and run `source ~/.bashrc`.

---

### Step 2 — Disable the snap Ollama service

Snap runs Ollama as a background service automatically. Disable it so you control
the environment variables yourself.

```bash
sudo snap stop ollama
sudo snap disable ollama
```

Verify it's fully stopped:

```bash
pgrep -a ollama
# should return nothing
```

---

### Step 3 — Launch Ollama manually on CPU only

```bash
CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 $OLLAMA_BIN serve > /tmp/ollama.log 2>&1 &
disown %1
```

Two environment variables are required together:
- `CUDA_VISIBLE_DEVICES=""` — hides all GPUs from the process at the OS level
- `OLLAMA_NUM_GPU=0` — Ollama-specific flag that disables its own GPU offloading

`disown` detaches the process from your terminal so it survives if the session closes.

Wait 5 seconds, then verify:

```bash
sleep 5
$OLLAMA_BIN run qwen2.5-coder:32b "say hello"
$OLLAMA_BIN ps
# PROCESSOR column must show: 100% CPU
```

Also confirm zero Ollama presence on the GPU:

```bash
nvidia-smi | grep ollama
# should return nothing
```

---

### Step 4 — Add the Ollama alias

The snap CLI wrapper (`/snap/bin/ollama`) is unavailable when the snap service
is disabled. Add an alias so you can keep using the `ollama` command:

```bash
alias ollama="$OLLAMA_BIN"
```

To make it permanent:

```bash
echo "alias ollama=\"$OLLAMA_BIN\"" >> ~/.bashrc
source ~/.bashrc
```

---

### Step 5 — Activate your virtual environment

```bash
cd $PROJECT_DIR
source .venv/bin/activate
```

Verify the correct Python is active:

```bash
which python
# should show: <your project dir>/.venv/bin/python
```

---

### Step 6 — Launch the agent

```bash
cd $PROJECT_DIR
source .venv/bin/activate

# clean slate (only needed between full experiment runs)
rm -f experiments.db
rm -rf mlruns/ outputs/

# optional: MLflow UI at localhost:5000
mlflow ui --port 5000 &

# launch agent in background, tail the log
nohup python agent.py > agent_run.log 2>&1 &
tail -f agent_run.log
```

---

## Every session after reboot — full startup sequence

```bash
# 1. Start Ollama on CPU
CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 /snap/ollama/current/bin/ollama serve > /tmp/ollama.log 2>&1 &
disown %1

# 2. Set alias
alias ollama='/snap/ollama/current/bin/ollama'

# 3. Verify (recommended)
sleep 5
ollama run qwen2.5-coder:32b "say hello"
ollama ps   # must show 100% CPU

# 4. Activate venv
cd $PROJECT_DIR
source .venv/bin/activate

# 5. Launch agent
nohup python agent.py > agent_run.log 2>&1 &
tail -f agent_run.log
```

---

## Convenience script: start_agent.sh

Save this in your home directory. Edit the two variables at the top, then run it
after every reboot instead of typing the sequence manually.

```bash
cat > ~/start_agent.sh << 'EOF'
#!/bin/bash
set -e

# ── Configure these for your machine ──────────────────────────────────────────
OLLAMA_BIN="/snap/ollama/current/bin/ollama"
PROJECT_DIR="$HOME/self-tuning-agent"   # update to your actual project path
# ──────────────────────────────────────────────────────────────────────────────

VENV="$PROJECT_DIR/.venv/bin/activate"

echo "=== Starting Ollama on CPU ==="
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 $OLLAMA_BIN serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
disown $OLLAMA_PID
echo "Ollama started (PID $OLLAMA_PID)"

sleep 5

echo "=== Verifying CPU-only mode ==="
$OLLAMA_BIN ps

echo "=== Activating virtual environment ==="
cd $PROJECT_DIR
source $VENV
echo "venv active: $(which python)"

echo "=== Launching agent ==="
nohup python agent.py > agent_run.log 2>&1 &
echo "Agent started. Tailing log..."
tail -f agent_run.log
EOF

chmod +x ~/start_agent.sh
```

Run it with:

```bash
~/start_agent.sh
```

---

## Verifying the split is working

With both processes running, open a second terminal:

```bash
nvidia-smi
```

You should see:
- PyTorch training process consuming VRAM (amount depends on your model and batch size)
- No Ollama process in the GPU list

```bash
/snap/ollama/current/bin/ollama ps
# qwen2.5-coder:32b   ...   100% CPU
```

---

## Model reference

| Model | Disk size | RAM needed | Notes |
|-------|-----------|------------|-------|
| qwen2.5-coder:32b | ~20GB | ~22GB | Recommended — fast, reliable JSON output, no think blocks |
| deepseek-r1:8b | ~5GB | ~8GB | Smaller footprint, but produces `<think>` blocks that require extra parsing and frequently drops required JSON fields — use for demos only, not unattended runs |

---

## .env reference

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:32b
DATA_PATH=./data/chest_xray
GOAL_F1=0.90
MAX_RUNS=12
```

---

## config.py timeout reference

```python
OLLAMA_TIMEOUT = 120  # seconds — sufficient for qwen2.5-coder:32b on a modern CPU
```

If you see timeouts, check `/tmp/ollama.log` — Ollama has likely crashed rather than
being genuinely slow.
