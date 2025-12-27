from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_name: str | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    max_int: int | None = None
    split: str | None = None


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


def _validate_positive(n: int, *, ctx: str) -> None:
    if n <= 0:
        raise ValueError(f"{ctx} must be > 0. Got: {n}")


def load_eval_config(path: Path | str) -> EvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data_any = yaml.safe_load(f)

    data = _require_mapping(data_any, ctx="Configuration file")

    task_name = str(_require_field(data, "task_name", ctx="top-level config"))
    results_root = str(_require_field(data, "results_root", ctx="top-level config"))
    seed_value = _require_field(data, "seed", ctx="top-level config")
    if not isinstance(seed_value, int):
        raise ValueError(f"seed must be an integer. Got: {seed_value!r}")
    seed = seed_value

    n_examples_value = _require_field(data, "n_examples", ctx="top-level config")
    if not isinstance(n_examples_value, int):
        raise ValueError(f"n_examples must be an integer. Got: {n_examples_value!r}")
    n_examples = n_examples_value

    batch_size_value = _require_field(data, "batch_size", ctx="top-level config")
    if not isinstance(batch_size_value, int):
        raise ValueError(f"batch_size must be an integer. Got: {batch_size_value!r}")
    batch_size = batch_size_value
    _validate_positive(n_examples, ctx="n_examples")
    _validate_positive(batch_size, ctx="batch_size")

    model_map = _require_mapping(_require_field(data, "model", ctx="top-level config"), ctx="model config")
    model_name_raw = _require_field(model_map, "name", ctx="model config")
    model_name = str(model_name_raw)
    tensor_parallel_size = model_map.get("tensor_parallel_size")
    if tensor_parallel_size is not None:
        if not isinstance(tensor_parallel_size, int):
            raise ValueError(f"model.tensor_parallel_size must be an integer. Got: {tensor_parallel_size!r}")
        _validate_positive(tensor_parallel_size, ctx="model.tensor_parallel_size")

    gpu_memory_utilization = model_map.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        if not isinstance(gpu_memory_utilization, (int, float)):
            raise ValueError(
                f"model.gpu_memory_utilization must be numeric. Got: {gpu_memory_utilization!r}"
            )
        gpu_memory_utilization = float(gpu_memory_utilization)

    max_model_len = model_map.get("max_model_len")
    if max_model_len is not None:
        if not isinstance(max_model_len, int):
            raise ValueError(f"model.max_model_len must be an integer. Got: {max_model_len!r}")
        _validate_positive(max_model_len, ctx="model.max_model_len")

    max_tokens = model_map.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            raise ValueError(f"model.max_tokens must be an integer. Got: {max_tokens!r}")
        _validate_positive(max_tokens, ctx="model.max_tokens")

    temperature = model_map.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise ValueError(f"model.temperature must be numeric. Got: {temperature!r}")
        temperature = float(temperature)

    top_p = model_map.get("top_p")
    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            raise ValueError(f"model.top_p must be numeric. Got: {top_p!r}")
        top_p = float(top_p)

    model_name_override = model_map.get("model_name")
    if model_name_override is not None:
        model_name_override = str(model_name_override)

    dtype_override = model_map.get("dtype")
    if dtype_override is not None:
        dtype_override = str(dtype_override)

    seed_override = model_map.get("seed")
    if seed_override is not None:
        if not isinstance(seed_override, int):
            raise ValueError(f"model.seed must be an integer. Got: {seed_override!r}")

    model_cfg = ModelConfig(
        name=model_name,
        model_name=model_name_override,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype_override,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed_override,
    )

    dataset_map = _require_mapping(_require_field(data, "dataset", ctx="top-level config"), ctx="dataset config")
    dataset_name_raw = _require_field(dataset_map, "name", ctx="dataset config")
    dataset_name = str(dataset_name_raw)
    max_int_value = dataset_map.get("max_int")
    if max_int_value is not None:
        if not isinstance(max_int_value, int):
            raise ValueError(f"dataset.max_int must be an integer. Got: {max_int_value!r}")
        _validate_positive(max_int_value, ctx="dataset.max_int")
    dataset_split = dataset_map.get("split")
    if dataset_split is not None:
        dataset_split = str(dataset_split)
    dataset_cfg = DatasetConfig(
        name=dataset_name,
        max_int=max_int_value,
        split=dataset_split,
    )

    return EvalConfig(
        task_name=task_name,
        results_root=results_root,
        seed=seed,
        n_examples=n_examples,
        batch_size=batch_size,
        model=model_cfg,
        dataset=dataset_cfg,
    )
