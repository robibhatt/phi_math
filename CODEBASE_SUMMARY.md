## High-level overview
- Minimal evaluation harness for synthetic math tasks: resolves CLI config paths to `Path`, loads YAML configs, creates incrementing run directories, snapshots configs, and executes the evaluation loop with `pathlib.Path`-based filesystem handling.【F:scripts/run_eval.py†L28-L45】【F:src/phi_synth_math/core/run_dir.py†L7-L20】
- The evaluation pipeline constructs dataset/model instances from registries, batches questions, scores predictions with exact match, and writes predictions/metrics/mistakes under the run directory.【F:src/phi_synth_math/tasks/eval/runner.py†L13-L103】

## Directory structure (depth ≤4)
- `configs/` — evaluation configuration files (YAML).【F:configs/eval/dummy_math.yaml†L1-L10】
  - `configs/eval/` — task-specific evaluation configs (dummy math).【F:configs/eval/dummy_math.yaml†L1-L10】
- `scripts/` — runnable scripts; `run_eval.py` is the CLI entrypoint.【F:scripts/run_eval.py†L7-L45】
- `src/` — Python package source tree (add to `PYTHONPATH`).【F:README.md†L1-L39】
  - `src/phi_synth_math/` — core utilities, models, and tasks.【F:src/phi_synth_math/__init__.py†L1-L5】
    - `core/` — config parsing, registries, run directory helpers, and JSONL utilities.【F:src/phi_synth_math/core/config.py†L10-L70】【F:src/phi_synth_math/core/run_dir.py†L7-L20】【F:src/phi_synth_math/core/registry.py†L12-L34】【F:src/phi_synth_math/core/jsonl.py†L8-L21】
    - `models/` — model protocol and dummy implementation.【F:src/phi_synth_math/models/base.py†L6-L10】【F:src/phi_synth_math/models/dummy.py†L9-L24】
    - `tasks/` — dataset definitions, evaluation loop, and scoring helpers.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】【F:src/phi_synth_math/tasks/eval/runner.py†L13-L103】
- `logs/` — captured job outputs from external runs (GPU test artifacts).【F:logs/test_gpu_cpu.out†L1-L60】

## Config system
- YAML configs: `configs/eval/dummy_math.yaml` defines task name, results root, RNG seed, number of examples, batch size, model name, dataset name, and dataset max integer.【F:configs/eval/dummy_math.yaml†L1-L10】
- Dataclasses (code): `ModelConfig` (`name`), `DatasetConfig` (`name`, optional `max_int`), and `EvalConfig` (`task_name`, `results_root`, `seed`, `n_examples`, `batch_size`, `model`, `dataset`). Validation checks required fields and nested `name` keys.【F:src/phi_synth_math/core/config.py†L10-L70】
- Config loader `load_eval_config(path: Path | str)` accepts file paths as `Path` or `str`, verifies existence, parses YAML, and returns `EvalConfig` while preserving the YAML-provided `results_root` string for later conversion to `Path`.【F:src/phi_synth_math/core/config.py†L43-L70】

## Core abstractions
- Dataset interface: iterable protocol yielding dicts with `id`, `question`, and `answer`.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】
- Model interface: protocol with `generate(questions, max_tokens=None) -> List[str]` for batch text outputs.【F:src/phi_synth_math/models/base.py†L6-L10】
- Run directory helpers: `make_run_dir(results_root: Path, task_name: str) -> Path` creates `<results_root>/<task_name>/<next_id>` and returns a `Path`; `save_config_snapshot(run_dir: Path, config_path: Path)` copies the YAML into `config.yaml`.【F:src/phi_synth_math/core/run_dir.py†L7-L20】
- Scoring helpers: `normalize_answer` (lowercase, strip spaces/commas) and `exact_match` comparator.【F:src/phi_synth_math/tasks/eval/scoring.py†L4-L9】

## Registries
- Dataset registry (`DATASET_REGISTRY`): keys → constructors. Available: `dummy_math_addition`; factory defaults `max_int` to 20 when absent.【F:src/phi_synth_math/core/registry.py†L12-L28】
- Model registry (`MODEL_REGISTRY`): keys → constructors. Available: `dummy`.【F:src/phi_synth_math/core/registry.py†L16-L34】

## Evaluation pipeline (CLI to outputs)
- CLI `python -m scripts.run_eval --config <yaml>` resolves the config path to `Path`, loads the config, converts `results_root` to `Path`, builds the next numeric run directory, and snapshots the config before running evaluation.【F:scripts/run_eval.py†L28-L38】
- `EvalRunner.run(config, run_dir: Path)` ensures the run directory exists, builds dataset/model from registries, streams through the dataset in batches, and calls the model’s `generate` per batch.【F:src/phi_synth_math/tasks/eval/runner.py†L13-L47】
- Each example produces a record with `id`, `question`, `gold`, `pred`, `correct` written to `predictions.jsonl` (JSON Lines).【F:src/phi_synth_math/tasks/eval/runner.py†L23-L103】
- Metrics (`accuracy`, `n_total`, `n_correct`) are written to `metrics.json` (pretty-printed JSON).【F:src/phi_synth_math/tasks/eval/runner.py†L49-L57】
- Up to 50 incorrect predictions are logged to `mistakes.txt` with tab-separated metadata when any mistakes occur.【F:src/phi_synth_math/tasks/eval/runner.py†L55-L63】【F:src/phi_synth_math/tasks/eval/runner.py†L79-L103】

## Dummy components
- Dummy dataset (`dummy_math_addition`): deterministic RNG (seeded) samples `n_examples` pairs of integers in `[0, max_int]`, emits question strings “What is a + b?” with stringified sum as `answer` and sequential `id`s (`ex_000001`, ...).【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】
- Dummy model (`dummy`): regex-extracts two integers from prompts of the form “what is X + Y” (case-insensitive) and returns their sum; otherwise outputs “I don't know”.【F:src/phi_synth_math/models/dummy.py†L9-L24】
- Assumptions/limits: model only handles simple addition phrasing; dataset currently generates only addition questions with non-negative integers.

## Reproducibility & determinism
- Seeding: dataset uses `random.Random(seed)` for repeatable number generation; model has deterministic regex logic, so outputs repeat for identical inputs and batches.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L16-L27】【F:src/phi_synth_math/models/dummy.py†L12-L24】
- Run directory determinism: run IDs auto-increment based on existing numeric subfolders; repeated runs with the same config create new numbered folders but content is deterministic given identical seeds/configs.【F:src/phi_synth_math/core/run_dir.py†L7-L20】
- No global seeding beyond dataset RNG; randomness confined to dataset sampling.

## Extensibility notes
- Add datasets: implement the `Dataset` protocol (yield `id`, `question`, `answer`) under `src/phi_synth_math/tasks/datasets/` and register in `DATASET_REGISTRY` with a unique key.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】【F:src/phi_synth_math/core/registry.py†L12-L28】
- Add models: implement the `Model` protocol under `src/phi_synth_math/models/` and register in `MODEL_REGISTRY`.【F:src/phi_synth_math/models/base.py†L6-L10】【F:src/phi_synth_math/core/registry.py†L16-L34】
- Modify evaluation logic: extend `EvalRunner` or scoring helpers in `src/phi_synth_math/tasks/eval/` to change batching, metrics, or output formats.【F:src/phi_synth_math/tasks/eval/runner.py†L13-L103】【F:src/phi_synth_math/tasks/eval/scoring.py†L4-L9】
