from __future__ import annotations

from typing import Callable, Dict

from phi_synth_math.models.base import Model
from phi_synth_math.models.dummy import DummyModel
from phi_synth_math.tasks.datasets.base import Dataset
from phi_synth_math.tasks.dummy_addition.dataset import DummyMathAdditionDataset

from .config import DatasetConfig, ModelConfig


def _create_vllm_model(**kwargs: object) -> Model:
    from phi_synth_math.models.vllm_model import VLLMModel

    return VLLMModel(**kwargs)


def _create_gsm8k_dataset(**kwargs: object) -> Dataset:
    from phi_synth_math.tasks.gsm8k.dataset import GSM8KDataset

    return GSM8KDataset(**kwargs)


DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    "dummy_math_addition": DummyMathAdditionDataset,
    "gsm8k": _create_gsm8k_dataset,
}

MODEL_REGISTRY: Dict[str, Callable[..., Model]] = {
    "dummy": DummyModel,
    "vllm": _create_vllm_model,
}


def make_dataset(cfg: DatasetConfig, *, n_examples: int, seed: int) -> Dataset:
    factory = DATASET_REGISTRY.get(cfg.name)
    if factory is None:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset name '{cfg.name}'. Available: {available}")

    if cfg.name == "dummy_math_addition":
        max_int = cfg.max_int if cfg.max_int is not None else 20
        return factory(n_examples=n_examples, seed=seed, max_int=max_int)

    if cfg.name == "gsm8k":
        split = cfg.split if cfg.split is not None else "test"
        return factory(n_examples=n_examples, seed=seed, split=split)

    raise ValueError(f"No construction path for dataset '{cfg.name}'.")


def make_model(cfg: ModelConfig) -> Model:
    factory = MODEL_REGISTRY.get(cfg.name)
    if factory is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model name '{cfg.name}'. Available: {available}")
    if cfg.name == "dummy":
        return factory()

    if cfg.name == "vllm":
        return factory(
            model_name=cfg.model_name,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            dtype=cfg.dtype,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )

    raise ValueError(f"No construction path for model '{cfg.name}'.")
