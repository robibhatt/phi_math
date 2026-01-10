"""Dummy trainer for testing without GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phi_synth_math.core.config import TrainingConfig


class DummyTrainer:
    """Deterministic trainer for testing - no GPU needed.

    Returns fixed metrics and writes minimal adapter files for testing
    the training pipeline without actual model training.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize dummy trainer.

        Args:
            config: Training configuration (not used, but accepted for API consistency).
        """
        self._config = config
        self._trained = False

    def train(self, run_dir: Path | None = None) -> dict[str, Any]:
        """Run dummy training.

        Args:
            run_dir: Optional run directory for checkpoint output.
                If provided, creates a checkpoint directory to simulate
                real trainer behavior.

        Returns:
            Dictionary with fixed training metrics.
        """
        self._trained = True

        # Create checkpoint directory if run_dir provided (simulates real trainer)
        if run_dir is not None:
            checkpoint_dir = Path(run_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        return {
            "train_loss": 0.5,
            "train_runtime": 1.0,
            "train_samples_per_second": 100.0,
            "epochs": 1,
        }

    def save_adapter(self, output_dir: Path) -> None:
        """Save minimal adapter config for testing.

        Args:
            output_dir: Directory to save adapter files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write minimal adapter_config.json
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": "dummy_model",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "trained": self._trained,
        }

        config_path = output_dir / "adapter_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)
