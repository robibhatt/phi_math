# Training Scripts

This directory contains scripts for fine-tuning language models with LoRA.

## Quick Start

### Single-GPU Training

```bash
python -m scripts.run_train --config configs/train/gsm8k_lora.yaml
```

### Multi-GPU Training (FSDP)

```bash
# Using the convenience script (auto-detects available GPUs)
./scripts/run_train_fsdp.sh --config configs/train/gsm8k_lora_fsdp.yaml

# Specify number of GPUs
./scripts/run_train_fsdp.sh --config configs/train/gsm8k_lora_fsdp.yaml --num_gpus 4

# Or use torchrun directly
torchrun --nproc_per_node=4 -m scripts.run_train --config configs/train/gsm8k_lora_fsdp.yaml
```

## Configuration Files

Two example configs are provided in `configs/train/`:

| Config | Description |
|--------|-------------|
| `gsm8k_lora.yaml` | Single-GPU LoRA fine-tuning |
| `gsm8k_lora_fsdp.yaml` | Multi-GPU FSDP training |

## Config Structure

### Basic Settings

```yaml
task_name: "gsm8k"           # Task to train on
results_root: "results/train" # Where to save outputs
seed: 42                      # Random seed
base_model: "microsoft/phi-1_5"  # Base model to fine-tune
trainer: "hf"                 # Trainer type (hf = HuggingFace)
```

### LoRA Settings

```yaml
lora:
  r: 8                        # LoRA rank
  lora_alpha: 16              # LoRA alpha (scaling factor)
  lora_dropout: 0.05          # Dropout probability
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"                # Bias training: "none", "all", or "lora_only"
  task_type: "CAUSAL_LM"      # Task type for PEFT
```

### Training Hyperparameters

```yaml
hyperparams:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 0.0002
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_scheduler_type: "cosine"
  logging_steps: 10
  save_steps: 500
  max_seq_length: 512
  mixed_precision: "fp16"     # "fp16" for V100s, "bf16" for A100+, "no" for fp32
```

### Weights & Biases

```yaml
wandb:
  project: "phi-math-finetune"
  entity: "your-entity"       # Your W&B username or team
  run_name: "gsm8k-lora-run"
  tags: ["lora", "gsm8k"]
  enabled: true               # Set to false to disable logging
```

### FSDP Configuration (Multi-GPU)

Add this section to enable FSDP:

```yaml
fsdp:
  enabled: true

  # Sharding strategy
  # - FULL_SHARD: Maximum memory savings (shard params, grads, optimizer states)
  # - SHARD_GRAD_OP: Shard grads and optimizer only (faster, more memory)
  # - NO_SHARD: Like DDP (no sharding)
  # - HYBRID_SHARD: Shard within node, replicate across nodes
  sharding_strategy: "FULL_SHARD"

  # CPU offloading (slower but saves GPU memory)
  cpu_offload: false

  # Auto-wrapping policy
  auto_wrap_policy: "TRANSFORMER_BASED_WRAP"

  # Transformer layer class to wrap (model-specific)
  # Common values: PhiDecoderLayer, LlamaDecoderLayer, GPT2Block
  transformer_layer_cls_to_wrap: "PhiDecoderLayer"

  # State dict type for checkpointing
  # SHARDED_STATE_DICT is recommended for large models
  state_dict_type: "SHARDED_STATE_DICT"

  # Activation checkpointing (trade compute for memory)
  activation_checkpointing: false
```

## Outputs

After training completes, outputs are saved to `results/train/<task>_<timestamp>/`:

```
results/train/gsm8k_20240101_120000/
├── adapter/           # LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoints/       # Training checkpoints
└── metrics.json       # Final training metrics
```

## FSDP Tips

1. **Sharding Strategy**: Start with `FULL_SHARD` for maximum memory efficiency. Use `SHARD_GRAD_OP` if you have enough memory and want faster training.

2. **Mixed Precision**: Use `bf16` on A100/H100 GPUs, `fp16` on V100s.

3. **CPU Offloading**: Enable `cpu_offload: true` if you're running out of GPU memory, but expect slower training.

4. **Activation Checkpointing**: Enable for very large models that don't fit in memory even with FSDP.

5. **Master Port Conflicts**: If running multiple jobs, use different ports:
   ```bash
   ./scripts/run_train_fsdp.sh --config <config> --master_port 29501
   ```

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `cpu_offload: true` in FSDP config
- Enable `activation_checkpointing: true`

### NCCL Timeout
- Increase timeout: `export NCCL_TIMEOUT=1800`
- Check network connectivity between nodes

### W&B Entity Not Found
- Ensure the entity exists on wandb.ai
- Set `entity: ""` to use your default account
