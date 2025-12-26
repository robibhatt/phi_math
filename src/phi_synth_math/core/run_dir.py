from __future__ import annotations

import shutil
from pathlib import Path


def make_run_dir(results_root: str, task_name: str) -> str:
    base_dir = Path(results_root).expanduser() / task_name
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = [int(p.name) for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    next_id = max(existing, default=0) + 1
    run_dir = base_dir / str(next_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def save_config_snapshot(run_dir: str, config_path: str) -> None:
    dest = Path(run_dir) / "config.yaml"
    shutil.copy(Path(config_path), dest)
