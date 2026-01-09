"""Integration tests for FSDP training configuration."""

import pytest
import yaml
from pathlib import Path

from phi_synth_math.core.config import load_training_config, TrainingConfig
from phi_synth_math.training.runner import TrainingRunner


@pytest.fixture
def fsdp_training_config_dict(tmp_dir: Path) -> dict:
    """Complete training config with FSDP enabled."""
    return {
        "task_name": "dummy_math_addition",
        "results_root": str(tmp_dir),
        "seed": 42,
        "base_model": "microsoft/phi-1_5",
        "trainer": "dummy",  # Use dummy trainer for tests
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "hyperparams": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0001,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "constant",
            "logging_steps": 1,
            "save_steps": 100,
            "max_seq_length": 128,
            "mixed_precision": "fp16",
        },
        "wandb": {
            "project": "test",
            "entity": "",
            "run_name": "test-fsdp",
            "tags": ["test"],
            "enabled": False,
        },
        "train_dataset": {
            "name": "dummy_math_addition",
            "split": "train",
        },
        "fsdp": {
            "enabled": True,
            "sharding_strategy": "FULL_SHARD",
            "transformer_layer_cls_to_wrap": "PhiDecoderLayer",
        },
    }


@pytest.fixture
def no_fsdp_training_config_dict(tmp_dir: Path) -> dict:
    """Complete training config without FSDP section."""
    return {
        "task_name": "dummy_math_addition",
        "results_root": str(tmp_dir),
        "seed": 42,
        "base_model": "microsoft/phi-1_5",
        "trainer": "dummy",
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "hyperparams": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0001,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "constant",
            "logging_steps": 1,
            "save_steps": 100,
            "max_seq_length": 128,
        },
        "wandb": {
            "project": "test",
            "entity": "",
            "run_name": "test",
            "tags": ["test"],
            "enabled": False,
        },
        "train_dataset": {
            "name": "dummy_math_addition",
            "split": "train",
        },
    }


class TestFSDPConfigIntegration:
    """Integration tests for FSDP configuration loading."""

    @pytest.mark.integration
    def test_load_config_with_fsdp_section(
        self, tmp_dir: Path, fsdp_training_config_dict: dict
    ):
        """Config with FSDP section loads correctly."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(fsdp_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp is not None
        assert config.fsdp.enabled is True
        assert config.fsdp.sharding_strategy == "FULL_SHARD"
        assert config.fsdp.transformer_layer_cls_to_wrap == "PhiDecoderLayer"

    @pytest.mark.integration
    def test_backward_compatibility_without_fsdp(
        self, tmp_dir: Path, no_fsdp_training_config_dict: dict
    ):
        """Config without FSDP section still works."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(no_fsdp_training_config_dict, f)

        config = load_training_config(config_path)

        # Should have fsdp=None
        assert config.fsdp is None
        # All other fields should still work
        assert config.task_name == "dummy_math_addition"
        assert config.trainer == "dummy"
        # mixed_precision should default to fp16
        assert config.hyperparams.mixed_precision == "fp16"

    @pytest.mark.integration
    def test_config_snapshot_includes_fsdp(
        self, tmp_dir: Path, fsdp_training_config_dict: dict
    ):
        """Runner saves FSDP config in snapshot."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(fsdp_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()
        run_dir = runner.run(config)

        # Check that config snapshot exists and contains FSDP section
        snapshot_path = run_dir / "config.yaml"
        assert snapshot_path.exists()

        with snapshot_path.open() as f:
            snapshot_data = yaml.safe_load(f)

        assert "fsdp" in snapshot_data
        assert snapshot_data["fsdp"]["enabled"] is True
        assert snapshot_data["fsdp"]["sharding_strategy"] == "FULL_SHARD"

    @pytest.mark.integration
    def test_config_snapshot_includes_mixed_precision(
        self, tmp_dir: Path, fsdp_training_config_dict: dict
    ):
        """Runner saves mixed_precision in config snapshot."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(fsdp_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()
        run_dir = runner.run(config)

        # Check that config snapshot contains mixed_precision
        snapshot_path = run_dir / "config.yaml"
        with snapshot_path.open() as f:
            snapshot_data = yaml.safe_load(f)

        assert "hyperparams" in snapshot_data
        assert snapshot_data["hyperparams"]["mixed_precision"] == "fp16"
