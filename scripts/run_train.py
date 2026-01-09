"""Entry point for training script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Ensure local src/ is on PYTHONPATH so `python -m scripts.run_train ...` works
# when running from the repo root without installing the package.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from phi_synth_math.core.config import TrainingConfig, load_training_config
from phi_synth_math.training.runner import TrainingRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    config: TrainingConfig = load_training_config(config_path)

    runner = TrainingRunner()
    run_dir = runner.run(config)

    print(f"Training complete for task '{config.task_name}'.")
    print(f"Run directory: {run_dir}")
    print(f"Adapter saved to: {run_dir / 'adapter'}")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
