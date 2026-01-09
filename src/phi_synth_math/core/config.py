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
    stop: list[str] | None = None
    repetition_penalty: float | None = None


VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
VALID_TRAINERS: tuple[str, ...] = ("dummy", "hf")
VALID_MIXED_PRECISION: tuple[str, ...] = ("no", "fp16", "bf16")

# FSDP validation constants
VALID_FSDP_SHARDING_STRATEGIES: tuple[str, ...] = (
    "FULL_SHARD",
    "SHARD_GRAD_OP",
    "NO_SHARD",
    "HYBRID_SHARD",
    "HYBRID_SHARD_ZERO2",
)
VALID_FSDP_STATE_DICT_TYPES: tuple[str, ...] = (
    "FULL_STATE_DICT",
    "LOCAL_STATE_DICT",
    "SHARDED_STATE_DICT",
)
VALID_FSDP_BACKWARD_PREFETCH: tuple[str, ...] = (
    "BACKWARD_PRE",
    "BACKWARD_POST",
    "NO_PREFETCH",
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str  # Required: must be one of VALID_SPLITS
    max_int: int | None = None


@dataclass(frozen=True)
class PromptConfig:
    few_shot_count: int = 0
    few_shot_split: str = "train"
    few_shot_seed: int | None = None
    example_format: str = "Q: {question}\nA: {answer}\n\n"
    test_format: str = "Q: {question}\nA:"
    static_examples: str | None = None  # e.g., "gsm8k_8shot" to use canonical examples


@dataclass(frozen=True)
class EvalConfig:
    task_name: str
    results_root: str
    seed: int
    n_examples: int
    batch_size: int
    model: ModelConfig
    dataset: DatasetConfig
    prompt: PromptConfig | None = None


@dataclass(frozen=True)
class LoRAConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    bias: str
    task_type: str


@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: str
    run_name: str
    tags: tuple[str, ...]
    enabled: bool


@dataclass(frozen=True)
class TrainingHyperparamsConfig:
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    max_seq_length: int
    mixed_precision: str = "fp16"  # Options: "no", "fp16", "bf16" (default fp16 for V100 compat)


@dataclass(frozen=True)
class FSDPConfig:
    """FSDP (Fully Sharded Data Parallel) configuration.

    Enables distributed training with model/gradient/optimizer sharding.
    Uses HuggingFace Trainer's built-in FSDP support via Accelerate.
    """

    enabled: bool = False
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP"
    transformer_layer_cls_to_wrap: str | None = None
    min_num_params: int = 100_000_000
    state_dict_type: str = "SHARDED_STATE_DICT"
    backward_prefetch: str = "BACKWARD_PRE"
    forward_prefetch: bool = False
    sync_module_states: bool = True
    use_orig_params: bool = True
    cpu_ram_efficient_loading: bool = True
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    task_name: str
    results_root: str
    seed: int
    base_model: str
    trainer: str
    lora: LoRAConfig
    hyperparams: TrainingHyperparamsConfig
    wandb: WandbConfig
    train_dataset: DatasetConfig
    fsdp: FSDPConfig | None = None  # Optional FSDP configuration


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

    stop_sequences = model_map.get("stop")
    if stop_sequences is not None:
        if not isinstance(stop_sequences, list):
            raise ValueError(f"model.stop must be a list of strings. Got: {stop_sequences!r}")
        for i, s in enumerate(stop_sequences):
            if not isinstance(s, str):
                raise ValueError(f"model.stop[{i}] must be a string. Got: {s!r}")

    repetition_penalty = model_map.get("repetition_penalty")
    if repetition_penalty is not None:
        if not isinstance(repetition_penalty, (int, float)):
            raise ValueError(f"model.repetition_penalty must be numeric. Got: {repetition_penalty!r}")
        repetition_penalty = float(repetition_penalty)

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
        stop=stop_sequences,
        repetition_penalty=repetition_penalty,
    )

    dataset_map = _require_mapping(_require_field(data, "dataset", ctx="top-level config"), ctx="dataset config")
    dataset_name_raw = _require_field(dataset_map, "name", ctx="dataset config")
    dataset_name = str(dataset_name_raw)
    max_int_value = dataset_map.get("max_int")
    if max_int_value is not None:
        if not isinstance(max_int_value, int):
            raise ValueError(f"dataset.max_int must be an integer. Got: {max_int_value!r}")
        _validate_positive(max_int_value, ctx="dataset.max_int")
    dataset_split_raw = _require_field(dataset_map, "split", ctx="dataset config")
    dataset_split = str(dataset_split_raw)
    if dataset_split not in VALID_SPLITS:
        raise ValueError(
            f"dataset.split must be one of {VALID_SPLITS}. Got: {dataset_split!r}"
        )
    dataset_cfg = DatasetConfig(
        name=dataset_name,
        split=dataset_split,
        max_int=max_int_value,
    )

    # Parse optional prompt config
    prompt_cfg: PromptConfig | None = None
    prompt_map = data.get("prompt")
    if prompt_map is not None:
        prompt_map = _require_mapping(prompt_map, ctx="prompt config")
        few_shot_count = prompt_map.get("few_shot_count", 0)
        if not isinstance(few_shot_count, int):
            raise ValueError(f"prompt.few_shot_count must be an integer. Got: {few_shot_count!r}")
        if few_shot_count < 0:
            raise ValueError(f"prompt.few_shot_count must be >= 0. Got: {few_shot_count}")

        few_shot_split = prompt_map.get("few_shot_split", "train")
        if not isinstance(few_shot_split, str):
            raise ValueError(f"prompt.few_shot_split must be a string. Got: {few_shot_split!r}")
        if few_shot_split not in VALID_SPLITS:
            raise ValueError(
                f"prompt.few_shot_split must be one of {VALID_SPLITS}. Got: {few_shot_split!r}"
            )

        few_shot_seed = prompt_map.get("few_shot_seed")
        if few_shot_seed is not None and not isinstance(few_shot_seed, int):
            raise ValueError(f"prompt.few_shot_seed must be an integer. Got: {few_shot_seed!r}")

        example_format = prompt_map.get("example_format", "Q: {question}\nA: {answer}\n\n")
        if not isinstance(example_format, str):
            raise ValueError(f"prompt.example_format must be a string. Got: {example_format!r}")

        test_format = prompt_map.get("test_format", "Q: {question}\nA:")
        if not isinstance(test_format, str):
            raise ValueError(f"prompt.test_format must be a string. Got: {test_format!r}")

        static_examples = prompt_map.get("static_examples")
        if static_examples is not None and not isinstance(static_examples, str):
            raise ValueError(f"prompt.static_examples must be a string. Got: {static_examples!r}")

        prompt_cfg = PromptConfig(
            few_shot_count=few_shot_count,
            few_shot_split=few_shot_split,
            few_shot_seed=few_shot_seed,
            example_format=example_format,
            test_format=test_format,
            static_examples=static_examples,
        )

    return EvalConfig(
        task_name=task_name,
        results_root=results_root,
        seed=seed,
        n_examples=n_examples,
        batch_size=batch_size,
        model=model_cfg,
        dataset=dataset_cfg,
        prompt=prompt_cfg,
    )


def load_training_config(path: Path | str) -> TrainingConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data_any = yaml.safe_load(f)

    data = _require_mapping(data_any, ctx="Configuration file")

    # Top-level required fields
    task_name = str(_require_field(data, "task_name", ctx="top-level config"))
    results_root = str(_require_field(data, "results_root", ctx="top-level config"))
    seed_value = _require_field(data, "seed", ctx="top-level config")
    if not isinstance(seed_value, int):
        raise ValueError(f"seed must be an integer. Got: {seed_value!r}")
    seed = seed_value
    base_model = str(_require_field(data, "base_model", ctx="top-level config"))

    # Parse trainer field
    trainer_value = _require_field(data, "trainer", ctx="top-level config")
    if not isinstance(trainer_value, str):
        raise ValueError(f"trainer must be a string. Got: {trainer_value!r}")
    trainer = trainer_value
    if trainer not in VALID_TRAINERS:
        raise ValueError(f"trainer must be one of {VALID_TRAINERS}. Got: {trainer!r}")

    # Parse LoRA config
    lora_map = _require_mapping(
        _require_field(data, "lora", ctx="top-level config"), ctx="lora config"
    )
    lora_r = _require_field(lora_map, "r", ctx="lora config")
    if not isinstance(lora_r, int):
        raise ValueError(f"Missing required field 'r' in lora config.")
    if lora_r <= 0:
        raise ValueError(f"lora.r must be > 0. Got: {lora_r}")
    lora_alpha = _require_field(lora_map, "lora_alpha", ctx="lora config")
    if not isinstance(lora_alpha, int):
        raise ValueError(f"Missing required field 'lora_alpha' in lora config.")
    lora_dropout = _require_field(lora_map, "lora_dropout", ctx="lora config")
    if not isinstance(lora_dropout, (int, float)):
        raise ValueError(f"Missing required field 'lora_dropout' in lora config.")
    lora_dropout = float(lora_dropout)
    target_modules = _require_field(lora_map, "target_modules", ctx="lora config")
    if not isinstance(target_modules, list):
        raise ValueError(f"Missing required field 'target_modules' in lora config.")
    bias = _require_field(lora_map, "bias", ctx="lora config")
    if not isinstance(bias, str):
        raise ValueError(f"Missing required field 'bias' in lora config.")
    task_type = _require_field(lora_map, "task_type", ctx="lora config")
    if not isinstance(task_type, str):
        raise ValueError(f"Missing required field 'task_type' in lora config.")

    lora_cfg = LoRAConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=tuple(target_modules),
        bias=bias,
        task_type=task_type,
    )

    # Parse hyperparams config
    hyperparams_map = _require_mapping(
        _require_field(data, "hyperparams", ctx="top-level config"),
        ctx="hyperparams config",
    )
    num_train_epochs = _require_field(
        hyperparams_map, "num_train_epochs", ctx="hyperparams config"
    )
    if not isinstance(num_train_epochs, int):
        raise ValueError(f"Missing required field 'num_train_epochs' in hyperparams config.")
    per_device_train_batch_size = _require_field(
        hyperparams_map, "per_device_train_batch_size", ctx="hyperparams config"
    )
    if not isinstance(per_device_train_batch_size, int):
        raise ValueError(
            f"Missing required field 'per_device_train_batch_size' in hyperparams config."
        )
    gradient_accumulation_steps = _require_field(
        hyperparams_map, "gradient_accumulation_steps", ctx="hyperparams config"
    )
    if not isinstance(gradient_accumulation_steps, int):
        raise ValueError(
            f"Missing required field 'gradient_accumulation_steps' in hyperparams config."
        )
    learning_rate = _require_field(
        hyperparams_map, "learning_rate", ctx="hyperparams config"
    )
    if not isinstance(learning_rate, (int, float)):
        raise ValueError(f"Missing required field 'learning_rate' in hyperparams config.")
    learning_rate = float(learning_rate)
    warmup_ratio = _require_field(
        hyperparams_map, "warmup_ratio", ctx="hyperparams config"
    )
    if not isinstance(warmup_ratio, (int, float)):
        raise ValueError(f"Missing required field 'warmup_ratio' in hyperparams config.")
    warmup_ratio = float(warmup_ratio)
    weight_decay = _require_field(
        hyperparams_map, "weight_decay", ctx="hyperparams config"
    )
    if not isinstance(weight_decay, (int, float)):
        raise ValueError(f"Missing required field 'weight_decay' in hyperparams config.")
    weight_decay = float(weight_decay)
    max_grad_norm = _require_field(
        hyperparams_map, "max_grad_norm", ctx="hyperparams config"
    )
    if not isinstance(max_grad_norm, (int, float)):
        raise ValueError(f"Missing required field 'max_grad_norm' in hyperparams config.")
    max_grad_norm = float(max_grad_norm)
    lr_scheduler_type = _require_field(
        hyperparams_map, "lr_scheduler_type", ctx="hyperparams config"
    )
    if not isinstance(lr_scheduler_type, str):
        raise ValueError(f"Missing required field 'lr_scheduler_type' in hyperparams config.")
    logging_steps = _require_field(
        hyperparams_map, "logging_steps", ctx="hyperparams config"
    )
    if not isinstance(logging_steps, int):
        raise ValueError(f"Missing required field 'logging_steps' in hyperparams config.")
    save_steps = _require_field(hyperparams_map, "save_steps", ctx="hyperparams config")
    if not isinstance(save_steps, int):
        raise ValueError(f"Missing required field 'save_steps' in hyperparams config.")
    max_seq_length = _require_field(
        hyperparams_map, "max_seq_length", ctx="hyperparams config"
    )
    if not isinstance(max_seq_length, int):
        raise ValueError(f"Missing required field 'max_seq_length' in hyperparams config.")

    # Parse optional mixed_precision (default: fp16 for V100 compatibility)
    mixed_precision = hyperparams_map.get("mixed_precision", "fp16")
    if not isinstance(mixed_precision, str):
        raise ValueError(f"hyperparams.mixed_precision must be a string. Got: {mixed_precision!r}")
    if mixed_precision not in VALID_MIXED_PRECISION:
        raise ValueError(
            f"hyperparams.mixed_precision must be one of {VALID_MIXED_PRECISION}. "
            f"Got: {mixed_precision!r}"
        )

    hyperparams_cfg = TrainingHyperparamsConfig(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_seq_length=max_seq_length,
        mixed_precision=mixed_precision,
    )

    # Parse wandb config
    wandb_map = _require_mapping(
        _require_field(data, "wandb", ctx="top-level config"), ctx="wandb config"
    )
    wandb_project = _require_field(wandb_map, "project", ctx="wandb config")
    if not isinstance(wandb_project, str):
        raise ValueError(f"Missing required field 'project' in wandb config.")
    wandb_entity = _require_field(wandb_map, "entity", ctx="wandb config")
    if not isinstance(wandb_entity, str):
        raise ValueError(f"Missing required field 'entity' in wandb config.")
    wandb_run_name = _require_field(wandb_map, "run_name", ctx="wandb config")
    if not isinstance(wandb_run_name, str):
        raise ValueError(f"Missing required field 'run_name' in wandb config.")
    wandb_tags = _require_field(wandb_map, "tags", ctx="wandb config")
    if not isinstance(wandb_tags, list):
        raise ValueError(f"Missing required field 'tags' in wandb config.")
    wandb_enabled = _require_field(wandb_map, "enabled", ctx="wandb config")
    if not isinstance(wandb_enabled, bool):
        raise ValueError(f"Missing required field 'enabled' in wandb config.")

    wandb_cfg = WandbConfig(
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=tuple(wandb_tags),
        enabled=wandb_enabled,
    )

    # Parse train_dataset config (reuse DatasetConfig)
    train_dataset_map = _require_mapping(
        _require_field(data, "train_dataset", ctx="top-level config"),
        ctx="train_dataset config",
    )
    train_dataset_name = _require_field(train_dataset_map, "name", ctx="train_dataset config")
    if not isinstance(train_dataset_name, str):
        raise ValueError(f"Missing required field 'name' in train_dataset config.")
    train_dataset_split = _require_field(train_dataset_map, "split", ctx="train_dataset config")
    if not isinstance(train_dataset_split, str):
        raise ValueError(f"Missing required field 'split' in train_dataset config.")
    if train_dataset_split not in VALID_SPLITS:
        raise ValueError(
            f"train_dataset.split must be one of {VALID_SPLITS}. Got: {train_dataset_split!r}"
        )

    train_dataset_cfg = DatasetConfig(
        name=train_dataset_name,
        split=train_dataset_split,
    )

    # Parse optional FSDP config
    fsdp_cfg: FSDPConfig | None = None
    fsdp_map = data.get("fsdp")
    if fsdp_map is not None:
        fsdp_map = _require_mapping(fsdp_map, ctx="fsdp config")

        # Parse enabled (default: False)
        fsdp_enabled = fsdp_map.get("enabled", False)
        if not isinstance(fsdp_enabled, bool):
            raise ValueError(f"fsdp.enabled must be a boolean. Got: {fsdp_enabled!r}")

        # Parse sharding_strategy (default: FULL_SHARD)
        sharding_strategy = fsdp_map.get("sharding_strategy", "FULL_SHARD")
        if not isinstance(sharding_strategy, str):
            raise ValueError(
                f"fsdp.sharding_strategy must be a string. Got: {sharding_strategy!r}"
            )
        if sharding_strategy not in VALID_FSDP_SHARDING_STRATEGIES:
            raise ValueError(
                f"fsdp.sharding_strategy must be one of {VALID_FSDP_SHARDING_STRATEGIES}. "
                f"Got: {sharding_strategy!r}"
            )

        # Parse cpu_offload (default: False)
        cpu_offload = fsdp_map.get("cpu_offload", False)
        if not isinstance(cpu_offload, bool):
            raise ValueError(f"fsdp.cpu_offload must be a boolean. Got: {cpu_offload!r}")

        # Parse auto_wrap_policy (default: TRANSFORMER_BASED_WRAP)
        auto_wrap_policy = fsdp_map.get("auto_wrap_policy", "TRANSFORMER_BASED_WRAP")
        if not isinstance(auto_wrap_policy, str):
            raise ValueError(
                f"fsdp.auto_wrap_policy must be a string. Got: {auto_wrap_policy!r}"
            )

        # Parse transformer_layer_cls_to_wrap (optional)
        transformer_layer_cls = fsdp_map.get("transformer_layer_cls_to_wrap")
        if transformer_layer_cls is not None and not isinstance(transformer_layer_cls, str):
            raise ValueError(
                f"fsdp.transformer_layer_cls_to_wrap must be a string. "
                f"Got: {transformer_layer_cls!r}"
            )

        # Parse min_num_params (default: 100_000_000)
        min_num_params = fsdp_map.get("min_num_params", 100_000_000)
        if not isinstance(min_num_params, int):
            raise ValueError(
                f"fsdp.min_num_params must be an integer. Got: {min_num_params!r}"
            )
        if min_num_params < 0:
            raise ValueError(f"fsdp.min_num_params must be >= 0. Got: {min_num_params}")

        # Parse state_dict_type (default: SHARDED_STATE_DICT)
        state_dict_type = fsdp_map.get("state_dict_type", "SHARDED_STATE_DICT")
        if not isinstance(state_dict_type, str):
            raise ValueError(
                f"fsdp.state_dict_type must be a string. Got: {state_dict_type!r}"
            )
        if state_dict_type not in VALID_FSDP_STATE_DICT_TYPES:
            raise ValueError(
                f"fsdp.state_dict_type must be one of {VALID_FSDP_STATE_DICT_TYPES}. "
                f"Got: {state_dict_type!r}"
            )

        # Parse backward_prefetch (default: BACKWARD_PRE)
        backward_prefetch = fsdp_map.get("backward_prefetch", "BACKWARD_PRE")
        if not isinstance(backward_prefetch, str):
            raise ValueError(
                f"fsdp.backward_prefetch must be a string. Got: {backward_prefetch!r}"
            )
        if backward_prefetch not in VALID_FSDP_BACKWARD_PREFETCH:
            raise ValueError(
                f"fsdp.backward_prefetch must be one of {VALID_FSDP_BACKWARD_PREFETCH}. "
                f"Got: {backward_prefetch!r}"
            )

        # Parse boolean flags with defaults
        forward_prefetch = fsdp_map.get("forward_prefetch", False)
        sync_module_states = fsdp_map.get("sync_module_states", True)
        use_orig_params = fsdp_map.get("use_orig_params", True)
        cpu_ram_efficient_loading = fsdp_map.get("cpu_ram_efficient_loading", True)
        limit_all_gathers = fsdp_map.get("limit_all_gathers", True)
        activation_checkpointing = fsdp_map.get("activation_checkpointing", False)

        # Validate all boolean flags
        for name, val in [
            ("forward_prefetch", forward_prefetch),
            ("sync_module_states", sync_module_states),
            ("use_orig_params", use_orig_params),
            ("cpu_ram_efficient_loading", cpu_ram_efficient_loading),
            ("limit_all_gathers", limit_all_gathers),
            ("activation_checkpointing", activation_checkpointing),
        ]:
            if not isinstance(val, bool):
                raise ValueError(f"fsdp.{name} must be a boolean. Got: {val!r}")

        fsdp_cfg = FSDPConfig(
            enabled=fsdp_enabled,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            transformer_layer_cls_to_wrap=transformer_layer_cls,
            min_num_params=min_num_params,
            state_dict_type=state_dict_type,
            backward_prefetch=backward_prefetch,
            forward_prefetch=forward_prefetch,
            sync_module_states=sync_module_states,
            use_orig_params=use_orig_params,
            cpu_ram_efficient_loading=cpu_ram_efficient_loading,
            limit_all_gathers=limit_all_gathers,
            activation_checkpointing=activation_checkpointing,
        )

    return TrainingConfig(
        task_name=task_name,
        results_root=results_root,
        seed=seed,
        base_model=base_model,
        trainer=trainer,
        lora=lora_cfg,
        hyperparams=hyperparams_cfg,
        wandb=wandb_cfg,
        train_dataset=train_dataset_cfg,
        fsdp=fsdp_cfg,
    )
