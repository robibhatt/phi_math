"""Integration tests for TrainingRunner using DummyTrainer."""

import json
import re
from pathlib import Path

import pytest
import yaml

from phi_synth_math.core.config import load_training_config
from phi_synth_math.training.runner import TrainingRunner


@pytest.fixture
def valid_training_config_dict(tmp_dir: Path) -> dict:
    """Complete valid training config dictionary."""
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
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "hyperparams": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0002,
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 500,
            "max_seq_length": 512,
        },
        "wandb": {
            "project": "phi-math-finetune",
            "entity": "",
            "run_name": "test-run",
            "tags": ["test"],
            "enabled": False,
        },
        "train_dataset": {
            "name": "dummy_math_addition",
            "split": "train",
        },
    }


class TestTrainingRunner:
    """Integration tests for the training pipeline."""

    @pytest.mark.integration
    def test_run_creates_timestamped_dir(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Auto-generates run directory like eval does."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()

        run_dir = runner.run(config)

        # Should create timestamped directory under results_root
        assert run_dir.exists()
        assert run_dir.parent == Path(config.results_root)

        # Directory name should contain task name and timestamp pattern
        # Format: {task_name}_{YYYYMMDD}_{HHMMSS}
        dir_name = run_dir.name
        assert dir_name.startswith(config.task_name)
        # Check for timestamp pattern (YYYYMMDD_HHMMSS)
        timestamp_pattern = r"\d{8}_\d{6}"
        assert re.search(timestamp_pattern, dir_name)

    @pytest.mark.integration
    def test_run_saves_config_snapshot(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Config YAML copied to run directory."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()

        run_dir = runner.run(config)

        # Config snapshot should exist
        snapshot_path = run_dir / "config.yaml"
        assert snapshot_path.exists()

        # Snapshot should be valid YAML with same content
        with snapshot_path.open() as f:
            snapshot_data = yaml.safe_load(f)

        assert snapshot_data["task_name"] == valid_training_config_dict["task_name"]
        assert snapshot_data["base_model"] == valid_training_config_dict["base_model"]
        assert snapshot_data["lora"]["r"] == valid_training_config_dict["lora"]["r"]

    @pytest.mark.integration
    def test_run_with_dummy_trainer_saves_adapter(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """End-to-end: config -> run -> adapter saved."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()

        run_dir = runner.run(config)

        # Adapter directory should exist
        adapter_dir = run_dir / "adapter"
        assert adapter_dir.exists()

        # adapter_config.json should exist (written by DummyTrainer)
        adapter_config_path = adapter_dir / "adapter_config.json"
        assert adapter_config_path.exists()

        # Verify adapter config content
        with adapter_config_path.open() as f:
            adapter_config = json.load(f)

        assert adapter_config["peft_type"] == "LORA"
        assert "r" in adapter_config
        assert "lora_alpha" in adapter_config

    @pytest.mark.integration
    def test_run_writes_metrics_json(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Training metrics saved to metrics.json."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()

        run_dir = runner.run(config)

        # metrics.json should exist
        metrics_path = run_dir / "metrics.json"
        assert metrics_path.exists()

        # Verify metrics content
        with metrics_path.open() as f:
            metrics = json.load(f)

        # DummyTrainer returns these metrics
        assert "train_loss" in metrics
        assert "epochs" in metrics
        assert metrics["train_loss"] == 0.5

    @pytest.mark.integration
    def test_run_full_pipeline(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Complete training pipeline integration test."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)
        runner = TrainingRunner()

        run_dir = runner.run(config)

        # Verify all expected outputs
        assert run_dir.exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "adapter").exists()
        assert (run_dir / "adapter" / "adapter_config.json").exists()
        assert (run_dir / "metrics.json").exists()

        # Verify run directory is under results_root
        assert str(run_dir).startswith(str(tmp_dir))
