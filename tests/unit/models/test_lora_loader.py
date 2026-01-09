"""Unit tests for LoRA model loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from phi_synth_math.models.lora_loader import load_lora_model


@pytest.fixture
def dummy_adapter_dir(tmp_dir: Path) -> Path:
    """Create a minimal dummy adapter directory for testing."""
    adapter_dir = tmp_dir / "adapter"
    adapter_dir.mkdir(parents=True)

    # Write minimal adapter_config.json (like DummyTrainer does)
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "dummy_model",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    config_path = adapter_dir / "adapter_config.json"
    with config_path.open("w") as f:
        json.dump(adapter_config, f)

    return adapter_dir


class TestLoRALoader:
    """Tests for LoRA model loading."""

    @pytest.mark.unit
    def test_load_lora_model_from_adapter_path(self, dummy_adapter_dir: Path):
        """Load base model + LoRA adapter from saved path."""
        # For unit testing, we use a dummy/mock mode that doesn't require GPU
        model = load_lora_model(
            base_model="dummy",
            adapter_path=dummy_adapter_dir,
            device="cpu",
        )

        assert model is not None

    @pytest.mark.unit
    def test_load_lora_model_returns_model_protocol(self, dummy_adapter_dir: Path):
        """Loaded model satisfies Model protocol."""
        model = load_lora_model(
            base_model="dummy",
            adapter_path=dummy_adapter_dir,
            device="cpu",
        )

        # Model protocol requires generate method
        assert hasattr(model, "generate")
        assert callable(model.generate)

        # Test that generate works
        questions = ["What is 2 + 3?"]
        results = model.generate(questions)

        assert isinstance(results, list)
        assert len(results) == len(questions)

    @pytest.mark.unit
    def test_load_lora_model_missing_adapter_raises(self, tmp_dir: Path):
        """Missing adapter path raises FileNotFoundError."""
        nonexistent_path = tmp_dir / "nonexistent_adapter"

        with pytest.raises(FileNotFoundError):
            load_lora_model(
                base_model="dummy",
                adapter_path=nonexistent_path,
                device="cpu",
            )

    @pytest.mark.unit
    def test_load_lora_model_invalid_adapter_raises(self, tmp_dir: Path):
        """Adapter directory without config raises ValueError."""
        # Create empty adapter directory
        empty_adapter_dir = tmp_dir / "empty_adapter"
        empty_adapter_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="adapter_config.json"):
            load_lora_model(
                base_model="dummy",
                adapter_path=empty_adapter_dir,
                device="cpu",
            )

    @pytest.mark.unit
    def test_load_lora_model_batch_generation(self, dummy_adapter_dir: Path):
        """Loaded model handles batch generation."""
        model = load_lora_model(
            base_model="dummy",
            adapter_path=dummy_adapter_dir,
            device="cpu",
        )

        questions = [
            "What is 1 + 1?",
            "What is 2 + 2?",
            "What is 3 + 3?",
        ]
        results = model.generate(questions)

        assert len(results) == 3
        # DummyModel should answer these correctly
        assert results[0] == "2"
        assert results[1] == "4"
        assert results[2] == "6"
