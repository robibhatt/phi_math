#!/bin/bash
#SBATCH --job-name=exp_script
#SBATCH --partition=v100-galvani
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=40:00:00
# (we keep default --export=ALL so your outer conda stays active)

set -euo pipefail

###############################################################################
# Determine repo root robustly in sbatch:
# SLURM runs a *copied* script from /var/... so BASH_SOURCE points there.
# Use SLURM_SUBMIT_DIR (where you ran sbatch from) as the anchor.
###############################################################################
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# If submit dir is inside a git repo, normalize to the git top-level
if git -C "$REPO_ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-toplevel)"
fi

cd "$REPO_ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run_name>" >&2
  exit 1
fi

RUN_NAME="$1"

###############################################################################
# Create directories for Slurm log files BEFORE anything prints
###############################################################################
RESULTS_ROOT="$REPO_ROOT/experiment_3_results"
mkdir -p "$RESULTS_ROOT/${RUN_NAME}/logs"

###############################################################################
# --- Diagnostics ---
###############################################################################
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_ROOT=$REPO_ROOT"
pwd
ls -ld . || true
ls -ld "$RESULTS_ROOT" || true
scontrol show job "$SLURM_JOB_ID" || true
nvidia-smi || true

###############################################################################
# --- Ensure writable TMP for this job ---
# Use a *short* path to avoid libzmq IPC length limits (<107 chars)
# Example: /tmp/pth-$USER-$RUN_NAME-$SLURM_JOB_ID
###############################################################################
SHORT_TMP_ROOT="/tmp/pth-${USER:-unknown}"
SHORT_TMPDIR="${SHORT_TMP_ROOT}/${RUN_NAME}-${SLURM_JOB_ID:-local}"

mkdir -p "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR"
export XDG_RUNTIME_DIR="$TMPDIR"
export CUDA_CACHE_PATH="$TMPDIR/cuda_cache"
mkdir -p "$CUDA_CACHE_PATH"

###############################################################################
# --- Force caches to our TMP (avoid stale /scratch_local paths) ---
###############################################################################
unset TORCHINDUCTOR_CACHE_DIR TORCHINDUCTOR_REMOTE_CACHE PYTORCH_TUNING_CACHE_DIR \
      TRITON_CACHE_DIR XDG_CACHE_HOME

export XDG_CACHE_HOME="$TMPDIR/.cache"; mkdir -p "$XDG_CACHE_HOME"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torch/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_TORCH_COMPILE_CACHE_DIR="$XDG_CACHE_HOME/vllm/torch_compile_cache"
export TORCHINDUCTOR_USE_REMOTE_CACHE=0

export WANDB_ENTITY=post-train-hallucinations
export WANDB_PROJECT=direct-fine-tune
export WANDB_MODE=online   # or offline if cluster blocks WAN

###############################################################################
# --- Run your experiment ---
###############################################################################

# Phase 1: setup + data + vLLM eval (single process)
python -m experiment_3_train_big_guy.experiment_setup "$RUN_NAME"

# Phase 2: training (match your SBATCH: 1 GPU)
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
PORT=$((10000 + RANDOM % 50000))
echo "Launching training with NUM_GPUS=${NUM_GPUS}"

srun accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${PORT}" \
  -m experiment_3_train_big_guy.experiment_train_accel "$RUN_NAME"

# Phase 3: results
python -m experiment_3_train_big_guy.experiment_finish "$RUN_NAME"
