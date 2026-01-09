"""Training runner for orchestrating the training pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from phi_synth_math.core.config import TrainingConfig
from phi_synth_math.core.registry import make_trainer


def _make_training_run_dir(results_root: Path, task_name: str) -> Path:
    """Create timestamped run directory for training.

    Args:
        results_root: Root directory for results.
        task_name: Name of the task.

    Returns:
        Path to the created run directory.

    Directory format: {results_root}/{task_name}_{YYYYMMDD}_{HHMMSS}
    """
    results_root = Path(results_root).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{task_name}_{timestamp}"
    run_dir = results_root / run_dir_name

    # Handle rare case of duplicate timestamps
    counter = 1
    original_name = run_dir_name
    while run_dir.exists():
        run_dir_name = f"{original_name}_{counter}"
        run_dir = results_root / run_dir_name
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_config_snapshot(run_dir: Path, config: TrainingConfig) -> None:
    """Save training config as YAML snapshot.

    Args:
        run_dir: Run directory to save config to.
        config: Training configuration to save.
    """
    # Convert frozen dataclass to dict for YAML serialization
    config_dict = {
        "task_name": config.task_name,
        "results_root": config.results_root,
        "seed": config.seed,
        "base_model": config.base_model,
        "trainer": config.trainer,
        "lora": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "target_modules": list(config.lora.target_modules),
            "bias": config.lora.bias,
            "task_type": config.lora.task_type,
        },
        "hyperparams": {
            "num_train_epochs": config.hyperparams.num_train_epochs,
            "per_device_train_batch_size": config.hyperparams.per_device_train_batch_size,
            "gradient_accumulation_steps": config.hyperparams.gradient_accumulation_steps,
            "learning_rate": config.hyperparams.learning_rate,
            "warmup_ratio": config.hyperparams.warmup_ratio,
            "weight_decay": config.hyperparams.weight_decay,
            "max_grad_norm": config.hyperparams.max_grad_norm,
            "lr_scheduler_type": config.hyperparams.lr_scheduler_type,
            "logging_steps": config.hyperparams.logging_steps,
            "save_steps": config.hyperparams.save_steps,
            "max_seq_length": config.hyperparams.max_seq_length,
            "mixed_precision": config.hyperparams.mixed_precision,
        },
        "wandb": {
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "run_name": config.wandb.run_name,
            "tags": list(config.wandb.tags),
            "enabled": config.wandb.enabled,
        },
        "train_dataset": {
            "name": config.train_dataset.name,
            "split": config.train_dataset.split,
        },
    }

    # Add FSDP config if present
    if config.fsdp is not None:
        config_dict["fsdp"] = {
            "enabled": config.fsdp.enabled,
            "sharding_strategy": config.fsdp.sharding_strategy,
            "cpu_offload": config.fsdp.cpu_offload,
            "auto_wrap_policy": config.fsdp.auto_wrap_policy,
            "transformer_layer_cls_to_wrap": config.fsdp.transformer_layer_cls_to_wrap,
            "min_num_params": config.fsdp.min_num_params,
            "state_dict_type": config.fsdp.state_dict_type,
            "backward_prefetch": config.fsdp.backward_prefetch,
            "forward_prefetch": config.fsdp.forward_prefetch,
            "sync_module_states": config.fsdp.sync_module_states,
            "use_orig_params": config.fsdp.use_orig_params,
            "cpu_ram_efficient_loading": config.fsdp.cpu_ram_efficient_loading,
            "limit_all_gathers": config.fsdp.limit_all_gathers,
            "activation_checkpointing": config.fsdp.activation_checkpointing,
        }

    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _save_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Save training metrics to JSON file.

    Args:
        run_dir: Run directory to save metrics to.
        metrics: Training metrics dictionary.
    """
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


class TrainingRunner:
    """Orchestrates the training pipeline.

    Handles:
    - Creating timestamped run directories
    - Saving config snapshots
    - Running training with specified trainer
    - Saving adapters and metrics
    """

    def run(self, config: TrainingConfig) -> Path:
        """Run the training pipeline.

        Args:
            config: Training configuration.

        Returns:
            Path to the run directory containing all outputs.
        """
        # Create run directory
        run_dir = _make_training_run_dir(
            Path(config.results_root), config.task_name
        )

        # Save config snapshot
        _save_config_snapshot(run_dir, config)

        # Create and run trainer
        trainer = make_trainer(config.trainer, config)
        metrics = trainer.train()

        # Save adapter
        adapter_dir = run_dir / "adapter"
        trainer.save_adapter(adapter_dir)

        # Save metrics
        _save_metrics(run_dir, metrics)

        return run_dir
