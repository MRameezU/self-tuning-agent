# Ollama on CPU · PyTorch Training on GPU
### Setup guide for self-tuning-agent on Ubuntu (RTX 4070 Super + snap Ollama)

The core problem: Ollama (snap install) partially offloads model layers onto the GPU
by default. With qwen2.5-coder:32b this consumes ~6GB VRAM before PyTorch starts,
causing OOM crashes at epoch 4 of every training run.

The fix: disable snap Ollama's service and launch it manually with GPU disabled.

---

## Hardware this was tested on

- CPU: Intel Core i9-14900K (32 threads)
- RAM: 64GB
- GPU: RTX 4070 Super (12GB VRAM)
- OS: Ubuntu 24.04
- Ollama: installed via snap

---

## Step 1 — Disable the snap Ollama service

Snap runs Ollama as a background service automatically. Disable it so you can
control the environment variables yourself.

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

## Step 2 — Launch Ollama manually on CPU only

```bash
CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 /snap/ollama/112/bin/ollama serve > /tmp/ollama.log 2>&1 &
```

Two env vars are required together:
- `CUDA_VISIBLE_DEVICES=""` — hides all GPUs from the process
- `OLLAMA_NUM_GPU=0` — Ollama-specific flag, disables GPU offloading

Disown the process so it survives terminal close:

```bash
disown %1
```

Wait 5 seconds, then verify:

```bash
sleep 5
/snap/ollama/112/bin/ollama run qwen2.5-coder:32b "say hello"
/snap/ollama/112/bin/ollama ps
# PROCESSOR column must show: 100% CPU
```

Also confirm zero Ollama presence on GPU:

```bash
nvidia-smi | grep ollama
# should return nothing
```

---

## Step 3 — Add the ollama alias (per session)

The snap CLI wrapper (`/snap/bin/ollama`) is unavailable when the snap service
is disabled. Add an alias at the start of any session where you need the CLI:

```bash
alias ollama='/snap/ollama/112/bin/ollama'
```

To make it permanent, add it to `~/.bashrc`:

```bash
echo "alias ollama='/snap/ollama/112/bin/ollama'" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4 — Activate the virtual environment

Always activate the `.venv` before running anything Python-related:

```bash
cd /home/narsun/PycharmProjects/self-tuning-agent
source .venv/bin/activate
# prompt will change to: (.venv) narsun@...
```

Verify it's using the right Python:

```bash
which python
# should show: /home/narsun/PycharmProjects/self-tuning-agent/.venv/bin/python
```

---

## Step 5 — Launch the agent

```bash
cd /home/narsun/PycharmProjects/self-tuning-agent
source .venv/bin/activate

# clean slate (only needed between full experiment runs)
rm -f experiments.db
rm -rf mlruns/ outputs/

# optional: MLflow UI
mlflow ui --port 5000 &

# launch agent
nohup python agent.py > agent_run.log 2>&1 &
tail -f agent_run.log
```

---

## Every time you reboot — full startup sequence

```bash
# 1. Start Ollama on CPU
CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 /snap/ollama/112/bin/ollama serve > /tmp/ollama.log 2>&1 &
disown %1

# 2. Set alias
alias ollama='/snap/ollama/112/bin/ollama'

# 3. Verify (optional but recommended)
sleep 5
ollama run qwen2.5-coder:32b "say hello"
ollama ps   # must show 100% CPU

# 4. Activate venv
cd /home/narsun/PycharmProjects/self-tuning-agent
source .venv/bin/activate

# 5. Launch agent
nohup python agent.py > agent_run.log 2>&1 &
tail -f agent_run.log
```

Or save this as a shell script — see convenience script below.

---

## Convenience script: start_agent.sh

Save this in the project root and run it after every reboot.

```bash
cat > ~/start_agent.sh << 'EOF'
#!/bin/bash
set -e

OLLAMA_BIN="/snap/ollama/112/bin/ollama"
PROJECT_DIR="/home/narsun/PycharmProjects/self-tuning-agent"
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
- PyTorch training process consuming 3–5GB VRAM
- No Ollama process in the GPU list

```bash
/snap/ollama/112/bin/ollama ps
# qwen2.5-coder:32b   ...   100% CPU
```

---

## Model reference

| Model | Size on disk | RAM needed | Use case |
|-------|-------------|------------|----------|
| qwen2.5-coder:32b | ~20GB | ~22GB | Primary — fast, reliable JSON, no think blocks |
| deepseek-r1:8b | ~5GB | ~8GB | Demo only — visible reasoning but unreliable JSON output |

qwen2.5-coder:32b is the correct model for production agent runs.
deepseek-r1:8b produces `<think>` blocks that require extra parsing logic
and frequently drops required fields from the JSON schema.

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
OLLAMA_TIMEOUT = 120  # seconds — sufficient for qwen2.5-coder:32b on i9-14900K
```

Do not raise this above 120 for qwen2.5-coder:32b. If you see timeouts with
this model, Ollama has likely crashed — check /tmp/ollama.log.