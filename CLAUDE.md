# Project Context for Claude

## Environment Setup

Always activate the conda environment before running anything:
```bash
source /mnt/lustre/work/luxburg/luj210/miniconda3/etc/profile.d/conda.sh && conda activate phi_math
```

## GPU Access on SLURM Cluster

This project runs on an HPC cluster managed by SLURM. Claude cannot launch interactive nodes (no `--pty` support), but CAN run GPU commands via `srun`:

```bash
srun --partition=v100-galvani --gres=gpu:v100:4 --mem=80G --time=00:10:00 python script.py
```

### Development workflow:
1. Edit code
2. Test with short `srun` jobs (small batch, few iterations) to verify it works
3. Debug from error output, fix, repeat
4. For full training runs, use `sbatch` to submit batch jobs

### Useful commands:
- Quick GPU test: `srun --partition=v100-galvani --gres=gpu:v100:1 --mem=20G --time=00:05:00 python script.py`
- Check queue: `squeue -u $USER`
- Check cluster: `sinfo -p v100-galvani`

### Constraints:
- Bash tool timeout is ~10 minutes, so keep test runs short
- For longer training, write a batch script and submit with `sbatch`

## Quick GPU Verification Tests

### Single-GPU Test (quick verification)
```bash
srun --partition=v100-galvani --gres=gpu:v100:1 --mem=20G --time=00:05:00 \
    python -m scripts.run_train --config configs/train/gsm8k_lora_quick.yaml
```

### Multi-GPU FSDP Test (distributed training verification)
```bash
srun --partition=v100-galvani --gres=gpu:v100:4 --mem=80G --time=00:10:00 \
    torchrun --nproc_per_node=4 -m scripts.run_train \
    --config configs/train/gsm8k_lora_fsdp_quick.yaml
```

**Note:** FSDP with mixed precision (fp16/bf16) has compatibility issues with PEFT LoRA. Use `mixed_precision: "no"` in FSDP configs for reliable training.
