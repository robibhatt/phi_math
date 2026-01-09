from __future__ import annotations

from typing import Callable, Dict

from phi_synth_math.models.base import Model
from phi_synth_math.models.dummy import DummyModel
from phi_synth_math.models.vllm_model import VLLMModel
from phi_synth_math.tasks.core.dataset import Dataset
from phi_synth_math.tasks.core.metadata import TASK_SPECS, get_task_spec
from phi_synth_math.training.base import Trainer
from phi_synth_math.training.dummy_trainer import DummyTrainer
from phi_synth_math.training.hf_trainer import HFTrainer

from .config import DatasetConfig, ModelConfig, TrainingConfig


DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    name: spec.dataset_builder for name, spec in TASK_SPECS.items()
}

MODEL_REGISTRY: Dict[str, Callable[..., Model]] = {
    "dummy": DummyModel,
    "vllm": VLLMModel,
}

TRAINER_REGISTRY: Dict[str, Callable[..., Trainer]] = {
    "dummy": DummyTrainer,
    "hf": HFTrainer,
}


def make_dataset(cfg: DatasetConfig, *, n_examples: int, seed: int) -> Dataset:
    task_spec = get_task_spec(cfg.name)
    dataset_kwargs = dict(task_spec.default_dataset_params)
    if cfg.max_int is not None:
        dataset_kwargs["max_int"] = cfg.max_int
    # Split is now mandatory in config
    dataset_kwargs["split"] = cfg.split

    factory = DATASET_REGISTRY.get(cfg.name)
    if factory is None:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset name '{cfg.name}'. Available: {available}")
    return factory(n_examples=n_examples, seed=seed, **dataset_kwargs)


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
            stop=cfg.stop,
            repetition_penalty=cfg.repetition_penalty,
        )

    raise ValueError(f"No construction path for model '{cfg.name}'.")


def make_trainer(name: str, config: TrainingConfig) -> Trainer:
    """Create a trainer by name.

    Args:
        name: Name of the trainer (e.g., "dummy", "hf").
        config: Training configuration.

    Returns:
        Trainer instance.

    Raises:
        ValueError: If trainer name is unknown.
    """
    factory = TRAINER_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(TRAINER_REGISTRY))
        raise ValueError(f"Unknown trainer '{name}'. Available: {available}")
    return factory(config)
