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

    def train(self) -> dict[str, Any]:
        """Run LoRA fine-tuning.

        Returns:
            Dictionary of training metrics including train_loss, epochs, etc.
        """
        config = self._config

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        # Load base model
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

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(Path(config.results_root) / "checkpoints"),
            num_train_epochs=config.hyperparams.num_train_epochs,
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
            bf16=True,
            remove_unused_columns=False,
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
