"""Unit tests for FSDP config loading and validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from phi_synth_math.core.config import (
    load_training_config,
    FSDPConfig,
    VALID_FSDP_SHARDING_STRATEGIES,
    VALID_FSDP_STATE_DICT_TYPES,
    VALID_FSDP_BACKWARD_PREFETCH,
)


@pytest.fixture
def valid_training_config_dict() -> dict:
    """Complete valid training config dictionary."""
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


class TestFSDPConfigParsing:
    """Tests for parsing FSDP section from YAML."""

    @pytest.mark.unit
    def test_fsdp_section_is_optional(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Config without fsdp section should have fsdp=None."""
        # Remove fsdp section if present
        valid_training_config_dict.pop("fsdp", None)
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp is None

    @pytest.mark.unit
    def test_fsdp_enabled_false_by_default(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """FSDP section with no enabled field defaults to False."""
        valid_training_config_dict["fsdp"] = {}
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp is not None
        assert config.fsdp.enabled is False

    @pytest.mark.unit
    def test_fsdp_enabled_true_parsed(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """FSDP enabled=true is correctly parsed."""
        valid_training_config_dict["fsdp"] = {"enabled": True}
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp.enabled is True

    @pytest.mark.unit
    def test_fsdp_sharding_strategy_parsed(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Sharding strategy is correctly parsed."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "sharding_strategy": "SHARD_GRAD_OP",
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp.sharding_strategy == "SHARD_GRAD_OP"

    @pytest.mark.unit
    def test_fsdp_sharding_strategy_defaults_to_full_shard(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Sharding strategy defaults to FULL_SHARD."""
        valid_training_config_dict["fsdp"] = {"enabled": True}
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp.sharding_strategy == "FULL_SHARD"

    @pytest.mark.unit
    def test_fsdp_invalid_sharding_strategy_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Invalid sharding strategy raises ValueError."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "sharding_strategy": "INVALID_STRATEGY",
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="sharding_strategy must be one of"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_fsdp_cpu_offload_parsed(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """CPU offload is correctly parsed."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "cpu_offload": True,
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp.cpu_offload is True

    @pytest.mark.unit
    def test_fsdp_transformer_layer_cls_parsed(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Transformer layer class is correctly parsed."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "transformer_layer_cls_to_wrap": "PhiDecoderLayer",
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp.transformer_layer_cls_to_wrap == "PhiDecoderLayer"

    @pytest.mark.unit
    def test_fsdp_all_fields_parsed(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """All FSDP fields are correctly parsed."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "sharding_strategy": "HYBRID_SHARD",
            "cpu_offload": True,
            "auto_wrap_policy": "SIZE_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "min_num_params": 50000000,
            "state_dict_type": "FULL_STATE_DICT",
            "backward_prefetch": "BACKWARD_POST",
            "forward_prefetch": True,
            "sync_module_states": False,
            "use_orig_params": False,
            "cpu_ram_efficient_loading": False,
            "limit_all_gathers": False,
            "activation_checkpointing": True,
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        assert config.fsdp.enabled is True
        assert config.fsdp.sharding_strategy == "HYBRID_SHARD"
        assert config.fsdp.cpu_offload is True
        assert config.fsdp.auto_wrap_policy == "SIZE_BASED_WRAP"
        assert config.fsdp.transformer_layer_cls_to_wrap == "LlamaDecoderLayer"
        assert config.fsdp.min_num_params == 50000000
        assert config.fsdp.state_dict_type == "FULL_STATE_DICT"
        assert config.fsdp.backward_prefetch == "BACKWARD_POST"
        assert config.fsdp.forward_prefetch is True
        assert config.fsdp.sync_module_states is False
        assert config.fsdp.use_orig_params is False
        assert config.fsdp.cpu_ram_efficient_loading is False
        assert config.fsdp.limit_all_gathers is False
        assert config.fsdp.activation_checkpointing is True

    @pytest.mark.unit
    def test_fsdp_config_is_frozen(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """FSDPConfig is immutable."""
        valid_training_config_dict["fsdp"] = {"enabled": True}
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        with pytest.raises(FrozenInstanceError):
            config.fsdp.enabled = False

    @pytest.mark.unit
    def test_fsdp_enabled_must_be_boolean(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """fsdp.enabled must be a boolean."""
        valid_training_config_dict["fsdp"] = {"enabled": "yes"}
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="fsdp.enabled must be a boolean"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_fsdp_invalid_state_dict_type_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Invalid state_dict_type raises ValueError."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "state_dict_type": "INVALID_TYPE",
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="state_dict_type must be one of"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_fsdp_invalid_backward_prefetch_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Invalid backward_prefetch raises ValueError."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "backward_prefetch": "INVALID_PREFETCH",
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="backward_prefetch must be one of"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_fsdp_min_num_params_negative_raises(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """Negative min_num_params raises ValueError."""
        valid_training_config_dict["fsdp"] = {
            "enabled": True,
            "min_num_params": -1,
        }
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        with pytest.raises(ValueError, match="min_num_params must be >= 0"):
            load_training_config(config_path)

    @pytest.mark.unit
    def test_fsdp_defaults_are_correct(
        self, tmp_dir: Path, valid_training_config_dict: dict
    ):
        """All FSDP defaults are set correctly."""
        valid_training_config_dict["fsdp"] = {"enabled": True}
        config_path = tmp_dir / "train_config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(valid_training_config_dict, f)

        config = load_training_config(config_path)

        # Verify all defaults
        assert config.fsdp.sharding_strategy == "FULL_SHARD"
        assert config.fsdp.cpu_offload is False
        assert config.fsdp.auto_wrap_policy == "TRANSFORMER_BASED_WRAP"
        assert config.fsdp.transformer_layer_cls_to_wrap is None
        assert config.fsdp.min_num_params == 100_000_000
        assert config.fsdp.state_dict_type == "SHARDED_STATE_DICT"
        assert config.fsdp.backward_prefetch == "BACKWARD_PRE"
        assert config.fsdp.forward_prefetch is False
        assert config.fsdp.sync_module_states is True
        assert config.fsdp.use_orig_params is True
        assert config.fsdp.cpu_ram_efficient_loading is True
        assert config.fsdp.limit_all_gathers is True
        assert config.fsdp.activation_checkpointing is False
