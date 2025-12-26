from __future__ import annotations

from typing import Callable, Dict

from phi_synth_math.models.base import Model
from phi_synth_math.models.dummy import DummyModel
from phi_synth_math.tasks.datasets.base import Dataset
from phi_synth_math.tasks.datasets.dummy_math_addition import DummyMathAdditionDataset

from .config import DatasetConfig, ModelConfig

DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    "dummy_math_addition": DummyMathAdditionDataset,
}

MODEL_REGISTRY: Dict[str, Callable[..., Model]] = {
    "dummy": DummyModel,
}


def make_dataset(cfg: DatasetConfig, *, n_examples: int, seed: int) -> Dataset:
    factory = DATASET_REGISTRY.get(cfg.name)
    if factory is None:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset name '{cfg.name}'. Available: {available}")

    max_int = cfg.max_int if cfg.max_int is not None else 20
    return factory(n_examples=n_examples, seed=seed, max_int=max_int)


def make_model(cfg: ModelConfig) -> Model:
    factory = MODEL_REGISTRY.get(cfg.name)
    if factory is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model name '{cfg.name}'. Available: {available}")
    return factory()
