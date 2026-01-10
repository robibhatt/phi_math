"""Training runner for orchestrating the training pipeline."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch.distributed as dist
import yaml

from phi_synth_math.core.config import TrainingConfig
from phi_synth_math.core.registry import make_trainer


def _get_rank() -> int:
    """Get the rank of the current process in distributed training.

    Checks environment variables set by torchrun/accelerate before falling
    back to torch.distributed (which may not be initialized yet).

    Returns:
        Rank of current process (0 if not in distributed mode).
    """
    # Check torchrun/accelerate environment variables first
    # These are set before dist.init_process_group() is called
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)

    # Fall back to torch.distributed if initialized
    if dist.is_initialized():
        return dist.get_rank()

    return 0


def _get_world_size() -> int:
    """Get the world size (number of processes) in distributed training.

    Returns:
        World size (1 if not in distributed mode).
    """
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)

    if dist.is_initialized():
        return dist.get_world_size()

    return 1


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training.

    Returns:
        True if this is rank 0 or not in distributed mode.
    """
    return _get_rank() == 0


def _is_distributed() -> bool:
    """Check if we're running in distributed mode.

    Returns:
        True if running with multiple processes.
    """
    return _get_world_size() > 1


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

        Note:
            In distributed training, only rank 0 creates directories and saves
            config/metrics. Other ranks read the run_dir from a marker file.
        """
        results_root = Path(config.results_root)
        results_root.mkdir(parents=True, exist_ok=True)

        # In distributed mode, use a marker file to coordinate run_dir
        marker_file = results_root / ".current_run_dir"

        if _is_main_process():
            # Rank 0 creates the directory and writes marker
            run_dir = _make_training_run_dir(results_root, config.task_name)
            marker_file.write_text(str(run_dir))
            # Save config snapshot (only on rank 0)
            _save_config_snapshot(run_dir, config)
        else:
            # Other ranks wait for marker file and read run_dir from it
            import time
            for _ in range(60):  # Wait up to 60 seconds
                if marker_file.exists():
                    run_dir = Path(marker_file.read_text().strip())
                    if run_dir.exists():
                        break
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"Rank {_get_rank()} timed out waiting for run_dir marker"
                )

        # Create and run trainer
        trainer = make_trainer(config.trainer, config)
        metrics = trainer.train(run_dir=run_dir)

        # Save adapter and metrics (only on rank 0)
        if _is_main_process():
            adapter_dir = run_dir / "adapter"
            trainer.save_adapter(adapter_dir)
            _save_metrics(run_dir, metrics)
            # Clean up marker file
            marker_file.unlink(missing_ok=True)

        return run_dir
