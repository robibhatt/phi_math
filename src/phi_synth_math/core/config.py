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


def _require_mapping(obj: Any, *, ctx: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{ctx} must be a mapping (YAML dictionary). Got: {type(obj).__name__}")
    return obj


def _require_field(mapping: dict[str, Any], key: str, *, ctx: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in {ctx}.")
    return mapping[key]


def _require_int(x: Any, *, ctx: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"{ctx} must be an integer. Got: {x!r}") from e


def _validate_positive(n: int, *, ctx: str) -> None:
    if n <= 0:
        raise ValueError(f"{ctx} must be > 0. Got: {n}")


def load_eval_config(path: str) -> EvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data_any = yaml.safe_load(f)

    data = _require_mapping(data_any, ctx="Configuration file")

    task_name = str(_require_field(data, "task_name", ctx="top-level config"))
    results_root = str(_require_field(data, "results_root", ctx="top-level config"))
    seed = _require_int(_require_field(data, "seed", ctx="top-level config"), ctx="seed")
    n_examples = _require_int(_require_field(data, "n_examples", ctx="top-level config"), ctx="n_examples")
    batch_size = _require_int(_require_field(data, "batch_size", ctx="top-level config"), ctx="batch_size")
    _validate_positive(n_examples, ctx="n_examples")
    _validate_positive(batch_size, ctx="batch_size")

    model_map = _require_mapping(_require_field(data, "model", ctx="top-level config"), ctx="model config")
    model_name = str(_require_field(model_map, "name", ctx="model config"))
    model_cfg = ModelConfig(name=model_name)

    dataset_map = _require_mapping(_require_field(data, "dataset", ctx="top-level config"), ctx="dataset config")
    dataset_name = str(_require_field(dataset_map, "name", ctx="dataset config"))

    max_int = None
    if "max_int" in dataset_map and dataset_map["max_int"] is not None:
        max_int = _require_int(dataset_map["max_int"], ctx="dataset.max_int")
        _validate_positive(max_int, ctx="dataset.max_int")

    dataset_cfg = DatasetConfig(name=dataset_name, max_int=max_int)

    return EvalConfig(
        task_name=task_name,
        results_root=results_root,
        seed=seed,
        n_examples=n_examples,
        batch_size=batch_size,
        model=model_cfg,
        dataset=dataset_cfg,
    )
