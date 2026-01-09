"""Unit tests for training config loading and validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from phi_synth_math.core.config import (
    load_training_config,
    TrainingConfig,
    LoRAConfig,
    WandbConfig,
    TrainingHyperparamsConfig,
)


@pytest.fixture
def valid_training_config_dict() -> dict:
    """Complete valid training config dictionary.

    All fields are required - no defaults. Tests can modify this dict as needed.
    """
    return {
        "task_name": "gsm8k",
        "results_root": "/tmp/train_results",
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
            "run_name": "gsm8k-lora-run",
            "tags": ["lora", "gsm8k"],
            "enabled": True,
        },
        "train_dataset": {
            "name": "gsm8k",
            "split": "train",
        },
    }


class TestTrainingConfig:
    """Tests for TrainingConfig loading and validation."""

    @pytest.mark.unit
    def test_load_training_config_all_fields_required(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """All fields must be in YAML - no defaults."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert isinstance(config, TrainingConfig)
        assert config.task_name == "gsm8k"
        assert config.results_root == "/tmp/train_results"
        assert config.seed == 42
        assert config.base_model == "microsoft/phi-1_5"

    @pytest.mark.unit
    def test_load_training_config_parses_lora_section(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """LoRA config parsed from YAML."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert isinstance(config.lora, LoRAConfig)
        assert config.lora.r == 8
        assert config.lora.lora_alpha == 16
        assert config.lora.lora_dropout == 0.05
        assert config.lora.target_modules == ("q_proj", "v_proj", "k_proj", "o_proj")
        assert config.lora.bias == "none"
        assert config.lora.task_type == "CAUSAL_LM"

    @pytest.mark.unit
    def test_load_training_config_parses_wandb_section(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """W&B config parsed from YAML."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert isinstance(config.wandb, WandbConfig)
        assert config.wandb.project == "phi-math-finetune"
        assert config.wandb.entity == ""
        assert config.wandb.run_name == "gsm8k-lora-run"
        assert config.wandb.tags == ("lora", "gsm8k")
        assert config.wandb.enabled is True

    @pytest.mark.unit
    def test_load_training_config_parses_hyperparams(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Hyperparameters parsed from YAML."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert isinstance(config.hyperparams, TrainingHyperparamsConfig)
        assert config.hyperparams.num_train_epochs == 3
        assert config.hyperparams.per_device_train_batch_size == 4
        assert config.hyperparams.gradient_accumulation_steps == 4
        assert config.hyperparams.learning_rate == 0.0002
        assert config.hyperparams.warmup_ratio == 0.03
        assert config.hyperparams.weight_decay == 0.0
        assert config.hyperparams.max_grad_norm == 1.0
        assert config.hyperparams.lr_scheduler_type == "cosine"
        assert config.hyperparams.logging_steps == 10
        assert config.hyperparams.save_steps == 500
        assert config.hyperparams.max_seq_length == 512

    @pytest.mark.unit
    def test_missing_required_field_raises(self, tmp_dir: Path):
        """Missing any required field raises ValueError."""
        incomplete_config = {
            "task_name": "gsm8k",
            # Missing all other required fields
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(incomplete_config, f)

        with pytest.raises(ValueError, match="Missing required field"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_missing_lora_field_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Missing LoRA field raises ValueError."""
        del valid_training_config_dict["lora"]["r"]
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="Missing required field.*lora"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_missing_hyperparams_field_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Missing hyperparams field raises ValueError."""
        del valid_training_config_dict["hyperparams"]["learning_rate"]
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="Missing required field.*hyperparams"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_missing_wandb_field_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Missing wandb field raises ValueError."""
        del valid_training_config_dict["wandb"]["project"]
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="Missing required field.*wandb"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_invalid_lora_rank_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Non-positive lora.r raises ValueError."""
        valid_training_config_dict["lora"]["r"] = 0
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="lora.r.*must be > 0"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_invalid_lora_rank_negative_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Negative lora.r raises ValueError."""
        valid_training_config_dict["lora"]["r"] = -4
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="lora.r.*must be > 0"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_config_is_frozen(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """TrainingConfig is immutable."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        with pytest.raises(FrozenInstanceError):
            config.task_name = "other_task"

    @pytest.mark.unit
    def test_lora_config_is_frozen(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """LoRAConfig is immutable."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        with pytest.raises(FrozenInstanceError):
            config.lora.r = 16

    @pytest.mark.unit
    def test_hyperparams_config_is_frozen(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """TrainingHyperparamsConfig is immutable."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        with pytest.raises(FrozenInstanceError):
            config.hyperparams.learning_rate = 0.001

    @pytest.mark.unit
    def test_wandb_config_is_frozen(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """WandbConfig is immutable."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        with pytest.raises(FrozenInstanceError):
            config.wandb.enabled = False

    @pytest.mark.unit
    def test_config_file_not_found_raises(self, tmp_dir: Path):
        """Non-existent config file raises FileNotFoundError."""
        config_path = tmp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_invalid_yaml_raises(self, tmp_dir: Path):
        """Invalid YAML content raises appropriate error."""
        config_path = tmp_dir / "invalid.yaml"
        with config_path.open("w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(Exception):  # yaml.YAMLError
            load_training_config(config_path)

    @pytest.mark.unit
    def test_train_dataset_uses_dataset_config(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """train_dataset field uses existing DatasetConfig."""
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        # Should reuse DatasetConfig from existing code
        from phi_synth_math.core.config import DatasetConfig
        assert isinstance(config.train_dataset, DatasetConfig)
        assert config.train_dataset.name == "gsm8k"
        assert config.train_dataset.split == "train"

    @pytest.mark.unit
    def test_load_training_config_parses_trainer_field(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Trainer field parsed from YAML."""
        valid_training_config_dict["trainer"] = "hf"
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.trainer == "hf"

    @pytest.mark.unit
    def test_trainer_field_accepts_dummy(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Trainer field accepts 'dummy' value."""
        valid_training_config_dict["trainer"] = "dummy"
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.trainer == "dummy"

    @pytest.mark.unit
    def test_missing_trainer_field_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Missing trainer field raises ValueError."""
        del valid_training_config_dict["trainer"]
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="Missing required field 'trainer'"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_invalid_trainer_value_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Invalid trainer value raises ValueError."""
        valid_training_config_dict["trainer"] = "invalid_trainer"
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="trainer must be one of"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_trainer_must_be_string(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Non-string trainer raises ValueError."""
        valid_training_config_dict["trainer"] = 123
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="trainer must be a string"):
            load_training_config(config_path)
