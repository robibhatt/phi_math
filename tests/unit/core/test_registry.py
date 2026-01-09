"""Unit tests for registry factory functions."""

import pytest

from phi_synth_math.core.config import (
    DatasetConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    TrainingHyperparamsConfig,
    WandbConfig,
)
from phi_synth_math.core.registry import make_model, make_dataset, make_trainer
from phi_synth_math.models.dummy import DummyModel


@pytest.fixture
def valid_training_config() -> TrainingConfig:
    """Create a valid TrainingConfig for testing."""
    return TrainingConfig(
        task_name="dummy_math_addition",
        results_root="/tmp/train_results",
        seed=42,
        base_model="dummy",
        trainer="dummy",
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


class TestMakeModel:
    """Tests for make_model factory."""

    @pytest.mark.unit
    def test_make_dummy_model(self):
        """Test creating a DummyModel via registry."""
        config = ModelConfig(name="dummy")
        model = make_model(config)

        assert isinstance(model, DummyModel)

    @pytest.mark.unit
    def test_unknown_model_raises(self):
        """Test that unknown model name raises ValueError."""
        config = ModelConfig(name="nonexistent_model")

        with pytest.raises(ValueError, match="Unknown model name"):
            make_model(config)


class TestMakeDataset:
    """Tests for make_dataset factory."""

    @pytest.mark.unit
    def test_make_dummy_addition_dataset(self):
        """Test creating DummyMathAdditionDataset via registry."""
        config = DatasetConfig(name="dummy_math_addition", split="test", max_int=10)
        dataset = make_dataset(config, n_examples=5, seed=42)

        examples = list(dataset)
        assert len(examples) == 5
        for ex in examples:
            assert "id" in ex
            assert "question" in ex
            assert "answer" in ex

    @pytest.mark.unit
    def test_unknown_dataset_raises(self):
        """Test that unknown dataset name raises ValueError."""
        config = DatasetConfig(name="nonexistent_dataset", split="test")

        with pytest.raises(ValueError, match="Unknown task"):
            make_dataset(config, n_examples=5, seed=42)


class TestTrainerRegistry:
    """Tests for make_trainer factory."""

    @pytest.mark.unit
    def test_make_trainer_dummy(self, valid_training_config: TrainingConfig):
        """make_trainer('dummy', config) returns DummyTrainer."""
        from phi_synth_math.training.dummy_trainer import DummyTrainer

        trainer = make_trainer("dummy", valid_training_config)

        assert isinstance(trainer, DummyTrainer)

    @pytest.mark.unit
    def test_make_trainer_hf(self, valid_training_config: TrainingConfig):
        """make_trainer('hf', config) returns HFTrainer."""
        from phi_synth_math.training.hf_trainer import HFTrainer

        trainer = make_trainer("hf", valid_training_config)

        assert isinstance(trainer, HFTrainer)

    @pytest.mark.unit
    def test_make_trainer_unknown_raises(self, valid_training_config: TrainingConfig):
        """Unknown trainer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown trainer"):
            make_trainer("nonexistent_trainer", valid_training_config)
