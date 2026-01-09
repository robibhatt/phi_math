"""Base classes and protocols for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Trainer(Protocol):
    """Protocol defining the trainer interface.

    All trainers must implement:
    - train(): Run training and return metrics
    - save_adapter(): Save trained adapter weights
    """

    def train(self) -> dict[str, Any]:
        """Run training.

        Returns:
            Dictionary of training metrics (e.g., train_loss, epochs).
        """
        ...

    def save_adapter(self, output_dir: Path) -> None:
        """Save trained adapter to disk.

        Args:
            output_dir: Directory to save adapter files.
        """
        ...
