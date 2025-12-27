from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Ensure local src/ is on PYTHONPATH so `python -m scripts.run_eval ...` works
# when running from the repo root without installing the package.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from phi_synth_math.core.config import EvalConfig, load_eval_config
from phi_synth_math.core.run_dir import make_run_dir, save_config_snapshot
from phi_synth_math.tasks.shared.runner import EvalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    config: EvalConfig = load_eval_config(config_path)
    results_root = Path(config.results_root).expanduser()

    run_dir = make_run_dir(results_root, config.task_name)
    save_config_snapshot(run_dir, config_path)

    runner = EvalRunner()
    metrics = runner.run(config, run_dir)

    accuracy = float(metrics.get("accuracy", 0.0))
    n_total = metrics.get("n_total", 0)
    n_correct = metrics.get("n_correct", 0)

    summary = (
        f"Run complete for task '{config.task_name}'. "
        f"Run directory: {run_dir}. "
        f"Metrics: accuracy={accuracy:.3f}, "
        f"n_total={n_total}, n_correct={n_correct}."
    )
    print(summary)


if __name__ == "__main__":
    main()
