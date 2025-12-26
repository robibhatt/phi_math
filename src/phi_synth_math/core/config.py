from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    max_int: int | None = None


@dataclass(frozen=True)
class EvalConfig:
    task_name: str
    results_root: str
    seed: int
    n_examples: int
    batch_size: int
    model: ModelConfig
    dataset: DatasetConfig


def _validate_config(data: dict[str, Any]) -> None:
    required_top = ["task_name", "results_root", "seed", "n_examples", "batch_size", "model", "dataset"]
    missing = [key for key in required_top if key not in data]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    if not isinstance(data["model"], dict) or "name" not in data["model"]:
        raise ValueError("Model config must include a 'name' field.")
    if not isinstance(data["dataset"], dict) or "name" not in data["dataset"]:
        raise ValueError("Dataset config must include a 'name' field.")


def load_eval_config(path: str) -> EvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping.")

    _validate_config(data)

    model_cfg = ModelConfig(name=str(data["model"]["name"]))
    dataset_cfg = DatasetConfig(
        name=str(data["dataset"]["name"]),
        max_int=int(data["dataset"]["max_int"]) if "max_int" in data["dataset"] and data["dataset"]["max_int"] is not None else None,
    )

    return EvalConfig(
        task_name=str(data["task_name"]),
        results_root=str(data["results_root"]),
        seed=int(data["seed"]),
        n_examples=int(data["n_examples"]),
        batch_size=int(data["batch_size"]),
        model=model_cfg,
        dataset=dataset_cfg,
    )
