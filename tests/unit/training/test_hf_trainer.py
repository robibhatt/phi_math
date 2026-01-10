"""Unit tests for HFTrainer."""

import json
from pathlib import Path

import pytest

from phi_synth_math.core.config import (
    DatasetConfig,
    LoRAConfig,
    TrainingConfig,
    TrainingHyperparamsConfig,
    WandbConfig,
)
from phi_synth_math.training.hf_trainer import HFTrainer


@pytest.fixture
def valid_training_config() -> TrainingConfig:
    """Create a valid TrainingConfig for testing."""
    return TrainingConfig(
        task_name="dummy_math_addition",
        results_root="/tmp/train_results",
        seed=42,
        base_model="dummy",
        trainer="hf",
        lora=LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=("q_proj", "v_proj"),
            bias="none",
            task_type="CAUSAL_LM",
        ),
        hyperparams=TrainingHyperparamsConfig(
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=0.0002,
            warmup_ratio=0.03,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=500,
            max_seq_length=512,
        ),
        wandb=WandbConfig(
            project="test",
            entity="",
            run_name="test-run",
            tags=("test",),
            enabled=False,
        ),
        train_dataset=DatasetConfig(
            name="dummy_math_addition",
            split="train",
        ),
    )


class TestHFTrainer:
    """Tests for HFTrainer class."""

    @pytest.mark.unit
    def test_hf_trainer_instantiates_with_config(
        self, valid_training_config: TrainingConfig
    ):
        """HFTrainer accepts TrainingConfig."""
        trainer = HFTrainer(valid_training_config)

        assert trainer._config == valid_training_config
        assert trainer._peft_model is None
        assert trainer._tokenizer is None

    @pytest.mark.unit
    def test_hf_trainer_save_adapter_without_training(
        self, valid_training_config: TrainingConfig, tmp_path: Path
    ):
        """save_adapter() creates adapter directory with config when training hasn't run."""
        trainer = HFTrainer(valid_training_config)
        adapter_dir = tmp_path / "adapter"

        trainer.save_adapter(adapter_dir)

        assert adapter_dir.exists()
        config_path = adapter_dir / "adapter_config.json"
        assert config_path.exists()

        with config_path.open() as f:
            adapter_config = json.load(f)

        assert adapter_config["peft_type"] == "LORA"
        assert adapter_config["base_model_name_or_path"] == "dummy"
        assert adapter_config["r"] == 8
        assert adapter_config["lora_alpha"] == 16
        assert adapter_config["trained"] is False

    @pytest.mark.unit
    def test_hf_trainer_save_adapter_creates_directory(
        self, valid_training_config: TrainingConfig, tmp_path: Path
    ):
        """save_adapter() creates nested directories if needed."""
        trainer = HFTrainer(valid_training_config)
        adapter_dir = tmp_path / "nested" / "deep" / "adapter"

        trainer.save_adapter(adapter_dir)

        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()

    @pytest.mark.unit
    def test_hf_trainer_has_required_methods(
        self, valid_training_config: TrainingConfig
    ):
        """HFTrainer implements Trainer protocol methods."""
        trainer = HFTrainer(valid_training_config)

        # Check that required methods exist and are callable
        assert callable(getattr(trainer, "train", None))
        assert callable(getattr(trainer, "save_adapter", None))

    @pytest.mark.unit
    def test_train_accepts_run_dir_parameter(
        self, valid_training_config: TrainingConfig
    ):
        """train() method should accept run_dir for checkpoint isolation."""
        import inspect

        sig = inspect.signature(HFTrainer.train)
        assert "run_dir" in sig.parameters, (
            "train() should accept run_dir parameter for checkpoint isolation"
        )
