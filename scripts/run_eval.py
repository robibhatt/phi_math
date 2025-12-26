from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from phi_synth_math.core.config import EvalConfig, load_eval_config
from phi_synth_math.core.run_dir import make_run_dir, save_config_snapshot
from phi_synth_math.tasks.eval.runner import EvalRunner


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
    config: EvalConfig = load_eval_config(str(config_path))

    run_dir = make_run_dir(config.results_root, config.task_name)
    save_config_snapshot(run_dir, str(config_path))

    runner = EvalRunner()
    metrics = runner.run(config, run_dir)

    summary = (
        f"Run complete for task '{config.task_name}'. "
        f"Run directory: {run_dir}. "
        f"Metrics: accuracy={metrics.get('accuracy'):.3f}, "
        f"n_total={metrics.get('n_total')}, n_correct={metrics.get('n_correct')}."
    )
    print(summary)


if __name__ == "__main__":
    main()
