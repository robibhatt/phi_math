"""Unit tests for config loading and validation."""

from pathlib import Path

import pytest
import yaml

from phi_synth_math.core.config import (
    load_eval_config,
    EvalConfig,
    ModelConfig,
    DatasetConfig,
    PromptConfig,
)


class TestLoadEvalConfig:
    """Tests for load_eval_config function."""

    @pytest.mark.unit
    def test_load_minimal_valid_config(self, tmp_dir: Path):
        """Test loading a minimal valid config file."""
        config_data = {
            "task_name": "dummy_math_addition",
            "results_root": "/tmp/results",
            "seed": 42,
            "n_examples": 10,
            "batch_size": 5,
            "model": {"name": "dummy"},
            "dataset": {"name": "dummy_math_addition", "split": "test"},
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        config = load_eval_config(config_path)

        assert isinstance(config, EvalConfig)
        assert config.task_name == "dummy_math_addition"
        assert config.seed == 42
        assert config.n_examples == 10
        assert config.batch_size == 5
        assert config.model.name == "dummy"
        assert config.dataset.name == "dummy_math_addition"
        assert config.dataset.split == "test"
        assert config.prompt is None

    @pytest.mark.unit
    def test_load_config_with_prompt_section(self, tmp_dir: Path):
        """Test loading config with prompt configuration."""
        config_data = {
            "task_name": "gsm8k",
            "results_root": "/tmp/results",
            "seed": 42,
            "n_examples": 10,
            "batch_size": 5,
            "model": {"name": "dummy"},
            "dataset": {"name": "gsm8k", "split": "test"},
            "prompt": {
                "few_shot_count": 8,
                "few_shot_split": "train",
                "example_format": "Q: {question}\nA: {answer}\n\n",
                "test_format": "Q: {question}\nA:",
            },
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        config = load_eval_config(config_path)

        assert config.prompt is not None
        assert config.prompt.few_shot_count == 8
        assert config.prompt.few_shot_split == "train"
        assert config.dataset.split == "test"

    @pytest.mark.unit
    def test_missing_required_field_raises(self, tmp_dir: Path):
        """Test that missing required fields raise ValueError."""
        config_data = {
            "task_name": "dummy_math_addition",
            # Missing: results_root, seed, n_examples, batch_size, model, dataset
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        with pytest.raises(ValueError, match="Missing required field"):
            load_eval_config(config_path)

    @pytest.mark.unit
    def test_invalid_batch_size_raises(self, tmp_dir: Path):
        """Test that non-positive batch_size raises ValueError."""
        config_data = {
            "task_name": "dummy_math_addition",
            "results_root": "/tmp/results",
            "seed": 42,
            "n_examples": 10,
            "batch_size": 0,  # Invalid: must be > 0
            "model": {"name": "dummy"},
            "dataset": {"name": "dummy_math_addition", "split": "test"},
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        with pytest.raises(ValueError, match="must be > 0"):
            load_eval_config(config_path)
