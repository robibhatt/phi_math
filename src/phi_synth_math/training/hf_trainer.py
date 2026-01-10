"""HuggingFace Trainer with LoRA for fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from phi_synth_math.training.sft_dataset import SFTDataset

if TYPE_CHECKING:
    from phi_synth_math.core.config import TrainingConfig


class HFTrainer:
    """HuggingFace Trainer with LoRA for fine-tuning.

    Uses PEFT LoRA adapters for parameter-efficient fine-tuning with
    HuggingFace's Trainer API.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize HFTrainer.

        Args:
            config: Training configuration specifying model, LoRA settings,
                hyperparameters, and W&B logging options.
        """
        self._config = config
        self._peft_model = None
        self._tokenizer = None

    def _is_fsdp_enabled(self) -> bool:
        """Check if FSDP is enabled in config."""
        return self._config.fsdp is not None and self._config.fsdp.enabled

    def _build_fsdp_args(self) -> tuple[str, dict[str, Any]]:
        """Build fsdp and fsdp_config arguments for TrainingArguments.

        Returns:
            Tuple of (fsdp_options_str, fsdp_config_dict)
        """
        fsdp_cfg = self._config.fsdp
        if fsdp_cfg is None or not fsdp_cfg.enabled:
            return "", {}

        # Map sharding strategy to HF format
        sharding_map = {
            "FULL_SHARD": "full_shard",
            "SHARD_GRAD_OP": "shard_grad_op",
            "NO_SHARD": "no_shard",
            "HYBRID_SHARD": "hybrid_shard",
            "HYBRID_SHARD_ZERO2": "hybrid_shard_zero2",
        }

        # Build fsdp options list
        fsdp_options = [sharding_map[fsdp_cfg.sharding_strategy]]

        if fsdp_cfg.cpu_offload:
            fsdp_options.append("offload")

        if fsdp_cfg.auto_wrap_policy:
            fsdp_options.append("auto_wrap")

        # Build fsdp_config dict
        fsdp_config: dict[str, Any] = {
            "backward_prefetch": fsdp_cfg.backward_prefetch.lower(),
            "forward_prefetch": fsdp_cfg.forward_prefetch,
            "limit_all_gathers": fsdp_cfg.limit_all_gathers,
            "use_orig_params": fsdp_cfg.use_orig_params,
            "cpu_ram_efficient_loading": fsdp_cfg.cpu_ram_efficient_loading,
            "sync_module_states": fsdp_cfg.sync_module_states,
            "state_dict_type": fsdp_cfg.state_dict_type,
        }

        # Add wrapping policy config
        if fsdp_cfg.auto_wrap_policy == "TRANSFORMER_BASED_WRAP":
            if fsdp_cfg.transformer_layer_cls_to_wrap:
                fsdp_config["transformer_layer_cls_to_wrap"] = [
                    fsdp_cfg.transformer_layer_cls_to_wrap
                ]
        elif fsdp_cfg.auto_wrap_policy == "SIZE_BASED_WRAP":
            fsdp_config["min_num_params"] = fsdp_cfg.min_num_params

        # Activation checkpointing
        if fsdp_cfg.activation_checkpointing:
            fsdp_config["activation_checkpointing"] = True

        return " ".join(fsdp_options), fsdp_config

    def train(self, run_dir: Path | None = None) -> dict[str, Any]:
        """Run LoRA fine-tuning.

        Args:
            run_dir: Run directory for checkpoints. If provided, checkpoints
                are saved to run_dir/checkpoints/ for isolation. If None,
                falls back to results_root/checkpoints/ (legacy behavior).

        Returns:
            Dictionary of training metrics including train_loss, epochs, etc.
        """
        config = self._config

        # Determine checkpoint directory - prefer run_dir for isolation
        if run_dir is not None:
            checkpoint_dir = Path(run_dir) / "checkpoints"
        else:
            checkpoint_dir = Path(config.results_root) / "checkpoints"

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        # Load base model
        # FSDP is incompatible with device_map="auto" - must load without it
        if self._is_fsdp_enabled():
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype="auto",
                device_map="auto",
            )

        # Configure LoRA
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }
        peft_task_type = task_type_map.get(config.lora.task_type, TaskType.CAUSAL_LM)

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules),
            bias=config.lora.bias,
            task_type=peft_task_type,
        )

        # Apply LoRA
        self._peft_model = get_peft_model(model, lora_config)
        self._peft_model.print_trainable_parameters()

        # Create dataset
        dataset = SFTDataset(
            task_name=config.task_name,
            split=config.train_dataset.split,
            tokenizer=self._tokenizer,
            max_seq_length=config.hyperparams.max_seq_length,
            seed=config.seed,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,  # Causal LM, not masked
        )

        # Configure W&B
        report_to = "wandb" if config.wandb.enabled else "none"

        # Build FSDP arguments
        fsdp_options, fsdp_config = self._build_fsdp_args()

        # Training arguments
        # max_steps=-1 means use num_train_epochs instead
        max_steps = config.hyperparams.max_steps if config.hyperparams.max_steps else -1
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=config.hyperparams.num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=config.hyperparams.per_device_train_batch_size,
            gradient_accumulation_steps=config.hyperparams.gradient_accumulation_steps,
            learning_rate=config.hyperparams.learning_rate,
            warmup_ratio=config.hyperparams.warmup_ratio,
            weight_decay=config.hyperparams.weight_decay,
            max_grad_norm=config.hyperparams.max_grad_norm,
            lr_scheduler_type=config.hyperparams.lr_scheduler_type,
            logging_steps=config.hyperparams.logging_steps,
            save_steps=config.hyperparams.save_steps,
            save_total_limit=2,
            report_to=report_to,
            run_name=config.wandb.run_name if config.wandb.enabled else None,
            seed=config.seed,
            fp16=(config.hyperparams.mixed_precision == "fp16"),
            bf16=(config.hyperparams.mixed_precision == "bf16"),
            remove_unused_columns=False,
            # FSDP arguments (empty strings/dicts become None)
            fsdp=fsdp_options if fsdp_options else None,
            fsdp_config=fsdp_config if fsdp_config else None,
        )

        # Initialize W&B if enabled
        if config.wandb.enabled:
            import wandb

            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity if config.wandb.entity else None,
                name=config.wandb.run_name,
                tags=list(config.wandb.tags),
                config={
                    "base_model": config.base_model,
                    "task_name": config.task_name,
                    "lora_r": config.lora.r,
                    "lora_alpha": config.lora.lora_alpha,
                    "learning_rate": config.hyperparams.learning_rate,
                    "epochs": config.hyperparams.num_train_epochs,
                    "batch_size": config.hyperparams.per_device_train_batch_size,
                },
            )

        # Create trainer
        trainer = Trainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Extract metrics
        metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0.0),
            "train_runtime": train_result.metrics.get("train_runtime", 0.0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0.0
            ),
            "epochs": config.hyperparams.num_train_epochs,
        }

        # Finish W&B run
        if config.wandb.enabled:
            import wandb

            wandb.finish()

        return metrics

    def save_adapter(self, output_dir: Path) -> None:
        """Save LoRA adapter weights.

        Args:
            output_dir: Directory to save adapter files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._peft_model is not None:
            # Use PEFT's save method which creates adapter_config.json and adapter weights
            self._peft_model.save_pretrained(output_dir)
        else:
            # Training hasn't run yet, save minimal config
            import json

            adapter_config = {
                "peft_type": "LORA",
                "base_model_name_or_path": self._config.base_model,
                "r": self._config.lora.r,
                "lora_alpha": self._config.lora.lora_alpha,
                "lora_dropout": self._config.lora.lora_dropout,
                "target_modules": list(self._config.lora.target_modules),
                "bias": self._config.lora.bias,
                "task_type": self._config.lora.task_type,
                "trained": False,
            }

            config_path = output_dir / "adapter_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2)
