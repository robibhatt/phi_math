"""Unit tests for registry factory functions."""

import pytest

from phi_synth_math.core.config import ModelConfig, DatasetConfig
from phi_synth_math.core.registry import make_model, make_dataset
from phi_synth_math.models.dummy import DummyModel


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
    def test_make_trainer_dummy(self):
        """make_trainer('dummy', ...) returns DummyTrainer."""
        from phi_synth_math.core.registry import make_trainer
        from phi_synth_math.training.dummy_trainer import DummyTrainer

        trainer = make_trainer("dummy")

        assert isinstance(trainer, DummyTrainer)

    @pytest.mark.unit
    def test_make_trainer_unknown_raises(self):
        """Unknown trainer name raises ValueError."""
        from phi_synth_math.core.registry import make_trainer

        with pytest.raises(ValueError, match="Unknown trainer"):
            make_trainer("nonexistent_trainer")
