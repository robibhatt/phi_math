"""Unit tests for HFTrainer FSDP configuration building."""

import pytest
from unittest.mock import MagicMock

from phi_synth_math.training.hf_trainer import HFTrainer
from phi_synth_math.core.config import (
    TrainingConfig,
    FSDPConfig,
    LoRAConfig,
    TrainingHyperparamsConfig,
    WandbConfig,
    DatasetConfig,
)


@pytest.fixture
def base_training_config() -> TrainingConfig:
    """Create a base TrainingConfig without FSDP."""
    return TrainingConfig(
        task_name="test_task",
        results_root="/tmp/test",
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
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=0.0001,
            warmup_ratio=0.0,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_scheduler_type="constant",
            logging_steps=1,
            save_steps=100,
            max_seq_length=128,
            mixed_precision="fp16",
        ),
        wandb=WandbConfig(
            project="test",
            entity="",
            run_name="test",
            tags=("test",),
            enabled=False,
        ),
        train_dataset=DatasetConfig(
            name="dummy_math_addition",
            split="train",
        ),
        fsdp=None,
    )


@pytest.fixture
def fsdp_enabled_config(base_training_config: TrainingConfig) -> TrainingConfig:
    """Create a TrainingConfig with FSDP enabled."""
    return TrainingConfig(
        task_name=base_training_config.task_name,
        results_root=base_training_config.results_root,
        seed=base_training_config.seed,
        base_model=base_training_config.base_model,
        trainer=base_training_config.trainer,
        lora=base_training_config.lora,
        hyperparams=base_training_config.hyperparams,
        wandb=base_training_config.wandb,
        train_dataset=base_training_config.train_dataset,
        fsdp=FSDPConfig(
            enabled=True,
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_layer_cls_to_wrap="PhiDecoderLayer",
            min_num_params=100_000_000,
            state_dict_type="SHARDED_STATE_DICT",
            backward_prefetch="BACKWARD_PRE",
            forward_prefetch=False,
            sync_module_states=True,
            use_orig_params=True,
            cpu_ram_efficient_loading=True,
            limit_all_gathers=True,
            activation_checkpointing=False,
        ),
    )


@pytest.fixture
def fsdp_disabled_config(base_training_config: TrainingConfig) -> TrainingConfig:
    """Create a TrainingConfig with FSDP section but disabled."""
    return TrainingConfig(
        task_name=base_training_config.task_name,
        results_root=base_training_config.results_root,
        seed=base_training_config.seed,
        base_model=base_training_config.base_model,
        trainer=base_training_config.trainer,
        lora=base_training_config.lora,
        hyperparams=base_training_config.hyperparams,
        wandb=base_training_config.wandb,
        train_dataset=base_training_config.train_dataset,
        fsdp=FSDPConfig(enabled=False),
    )


class TestHFTrainerFSDPConfig:
    """Tests for HFTrainer FSDP argument building."""

    @pytest.mark.unit
    def test_is_fsdp_enabled_returns_false_when_fsdp_none(
        self, base_training_config: TrainingConfig
    ):
        """_is_fsdp_enabled returns False when fsdp is None."""
        trainer = HFTrainer(base_training_config)
        assert trainer._is_fsdp_enabled() is False

    @pytest.mark.unit
    def test_is_fsdp_enabled_returns_false_when_fsdp_disabled(
        self, fsdp_disabled_config: TrainingConfig
    ):
        """_is_fsdp_enabled returns False when fsdp.enabled is False."""
        trainer = HFTrainer(fsdp_disabled_config)
        assert trainer._is_fsdp_enabled() is False

    @pytest.mark.unit
    def test_is_fsdp_enabled_returns_true_when_fsdp_enabled(
        self, fsdp_enabled_config: TrainingConfig
    ):
        """_is_fsdp_enabled returns True when fsdp.enabled is True."""
        trainer = HFTrainer(fsdp_enabled_config)
        assert trainer._is_fsdp_enabled() is True

    @pytest.mark.unit
    def test_build_fsdp_args_returns_empty_when_fsdp_none(
        self, base_training_config: TrainingConfig
    ):
        """_build_fsdp_args returns empty when FSDP is None."""
        trainer = HFTrainer(base_training_config)
        fsdp_options, fsdp_config = trainer._build_fsdp_args()

        assert fsdp_options == ""
        assert fsdp_config == {}

    @pytest.mark.unit
    def test_build_fsdp_args_returns_empty_when_disabled(
        self, fsdp_disabled_config: TrainingConfig
    ):
        """_build_fsdp_args returns empty when FSDP is disabled."""
        trainer = HFTrainer(fsdp_disabled_config)
        fsdp_options, fsdp_config = trainer._build_fsdp_args()

        assert fsdp_options == ""
        assert fsdp_config == {}

    @pytest.mark.unit
    def test_build_fsdp_args_includes_sharding_strategy(
        self, fsdp_enabled_config: TrainingConfig
    ):
        """_build_fsdp_args includes sharding strategy in options."""
        trainer = HFTrainer(fsdp_enabled_config)
        fsdp_options, _ = trainer._build_fsdp_args()

        assert "full_shard" in fsdp_options

    @pytest.mark.unit
    def test_build_fsdp_args_includes_auto_wrap(
        self, fsdp_enabled_config: TrainingConfig
    ):
        """_build_fsdp_args includes auto_wrap when policy set."""
        trainer = HFTrainer(fsdp_enabled_config)
        fsdp_options, _ = trainer._build_fsdp_args()

        assert "auto_wrap" in fsdp_options

    @pytest.mark.unit
    def test_build_fsdp_args_includes_offload_when_enabled(
        self, base_training_config: TrainingConfig
    ):
        """_build_fsdp_args includes offload when cpu_offload is True."""
        config = TrainingConfig(
            task_name=base_training_config.task_name,
            results_root=base_training_config.results_root,
            seed=base_training_config.seed,
            base_model=base_training_config.base_model,
            trainer=base_training_config.trainer,
            lora=base_training_config.lora,
            hyperparams=base_training_config.hyperparams,
            wandb=base_training_config.wandb,
            train_dataset=base_training_config.train_dataset,
            fsdp=FSDPConfig(enabled=True, cpu_offload=True),
        )
        trainer = HFTrainer(config)
        fsdp_options, _ = trainer._build_fsdp_args()

        assert "offload" in fsdp_options

    @pytest.mark.unit
    def test_build_fsdp_args_config_includes_transformer_layer(
        self, fsdp_enabled_config: TrainingConfig
    ):
        """_build_fsdp_args config includes transformer layer class."""
        trainer = HFTrainer(fsdp_enabled_config)
        _, fsdp_config = trainer._build_fsdp_args()

        assert "transformer_layer_cls_to_wrap" in fsdp_config
        assert "PhiDecoderLayer" in fsdp_config["transformer_layer_cls_to_wrap"]

    @pytest.mark.unit
    def test_build_fsdp_args_config_includes_all_flags(
        self, fsdp_enabled_config: TrainingConfig
    ):
        """_build_fsdp_args config includes all boolean flags."""
        trainer = HFTrainer(fsdp_enabled_config)
        _, fsdp_config = trainer._build_fsdp_args()

        assert fsdp_config["forward_prefetch"] is False
        assert fsdp_config["sync_module_states"] is True
        assert fsdp_config["use_orig_params"] is True
        assert fsdp_config["limit_all_gathers"] is True

    @pytest.mark.unit
    def test_build_fsdp_args_maps_all_sharding_strategies(
        self, base_training_config: TrainingConfig
    ):
        """_build_fsdp_args correctly maps all sharding strategies."""
        strategy_map = {
            "FULL_SHARD": "full_shard",
            "SHARD_GRAD_OP": "shard_grad_op",
            "NO_SHARD": "no_shard",
            "HYBRID_SHARD": "hybrid_shard",
            "HYBRID_SHARD_ZERO2": "hybrid_shard_zero2",
        }

        for config_value, expected_hf_value in strategy_map.items():
            config = TrainingConfig(
                task_name=base_training_config.task_name,
                results_root=base_training_config.results_root,
                seed=base_training_config.seed,
                base_model=base_training_config.base_model,
                trainer=base_training_config.trainer,
                lora=base_training_config.lora,
                hyperparams=base_training_config.hyperparams,
                wandb=base_training_config.wandb,
                train_dataset=base_training_config.train_dataset,
                fsdp=FSDPConfig(enabled=True, sharding_strategy=config_value),
            )
            trainer = HFTrainer(config)
            fsdp_options, _ = trainer._build_fsdp_args()

            assert expected_hf_value in fsdp_options, f"Expected {expected_hf_value} for {config_value}"

    @pytest.mark.unit
    def test_build_fsdp_args_activation_checkpointing(
        self, base_training_config: TrainingConfig
    ):
        """_build_fsdp_args includes activation_checkpointing in config."""
        config = TrainingConfig(
            task_name=base_training_config.task_name,
            results_root=base_training_config.results_root,
            seed=base_training_config.seed,
            base_model=base_training_config.base_model,
            trainer=base_training_config.trainer,
            lora=base_training_config.lora,
            hyperparams=base_training_config.hyperparams,
            wandb=base_training_config.wandb,
            train_dataset=base_training_config.train_dataset,
            fsdp=FSDPConfig(enabled=True, activation_checkpointing=True),
        )
        trainer = HFTrainer(config)
        _, fsdp_config = trainer._build_fsdp_args()

        assert fsdp_config.get("activation_checkpointing") is True
