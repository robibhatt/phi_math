#!/bin/bash
#SBATCH --job-name=test_gpu_cpu
#SBATCH --output=logs/test_gpu_cpu.out
#SBATCH --error=logs/test_gpu_cpu.err
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=01:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

###############################################################################
# --- Ensure writable TMP for this job ---
###############################################################################
if [[ -n "${SLURM_TMPDIR:-}" && -w "$SLURM_TMPDIR" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
else
  export TMPDIR="$PWD/tmp"
  mkdir -p "$TMPDIR"
fi
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

python run_vllm_queries_yaml.py