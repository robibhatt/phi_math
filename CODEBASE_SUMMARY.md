## High-level overview
- Minimal evaluation harness for synthetic math tasks: loads a YAML config, instantiates a dataset and model from registries, runs batch inference, and writes predictions plus metrics to a run directory.【F:scripts/run_eval.py†L12-L45】【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】
- Primary entrypoint: `python -m scripts.run_eval --config <path>` resolves the config, creates an incrementing run folder, snapshots the config, and executes the evaluation loop.【F:scripts/run_eval.py†L7-L45】

## Directory structure (depth ≤4)
- `configs/` — evaluation configuration files (YAML).【F:configs/eval/dummy_math.yaml†L1-L10】
  - `configs/eval/` — task-specific evaluation configs (dummy math).【F:configs/eval/dummy_math.yaml†L1-L10】
- `scripts/` — runnable scripts; `run_eval.py` drives evaluations from the CLI.【F:scripts/run_eval.py†L7-L45】
- `src/` — Python package source tree (add to `PYTHONPATH`).【F:README.md†L1-L39】
  - `src/phi_synth_math/` — root package for core utilities, models, and tasks.【F:src/phi_synth_math/__init__.py†L1-L5】
    - `core/` — config parsing, registries, run directory helpers, and JSONL utilities.【F:src/phi_synth_math/core/config.py†L10-L70】【F:src/phi_synth_math/core/run_dir.py†L7-L20】【F:src/phi_synth_math/core/registry.py†L12-L34】【F:src/phi_synth_math/core/jsonl.py†L8-L21】
    - `models/` — model protocol and dummy implementation.【F:src/phi_synth_math/models/base.py†L6-L10】【F:src/phi_synth_math/models/dummy.py†L9-L24】
    - `tasks/` — dataset definitions and evaluation loop + scoring.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】
- `logs/` — captured job outputs from external runs (GPU test artifacts).【F:logs/test_gpu_cpu.out†L1-L60】

## Config system
- YAML configs:
  - `configs/eval/dummy_math.yaml`: defines a dummy math evaluation specifying task name, results root, RNG seed, number of examples, batch size, model name, dataset name, and dataset max integer.【F:configs/eval/dummy_math.yaml†L1-L10】
- Dataclasses (in code):
  - `ModelConfig`: `name`.【F:src/phi_synth_math/core/config.py†L10-L13】
  - `DatasetConfig`: `name`, optional `max_int`.【F:src/phi_synth_math/core/config.py†L15-L19】
  - `EvalConfig`: `task_name`, `results_root`, `seed`, `n_examples`, `batch_size`, `model`, `dataset`. Validation enforces presence of required fields and nested `name` keys for model/dataset.【F:src/phi_synth_math/core/config.py†L21-L70】

## Core abstractions
- Dataset interface: iterable protocol yielding dicts with `id`, `question`, and `answer`.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】
- Model interface: protocol with `generate(questions, max_tokens=None) -> List[str]` for batch text outputs.【F:src/phi_synth_math/models/base.py†L6-L10】
- Eval runner: orchestrates dataset/model instantiation, batches questions, scores predictions via exact match, writes `predictions.jsonl`, aggregates metrics, and logs first 50 mistakes to `mistakes.txt` when present.【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】
- Scoring helpers: `normalize_answer` (lowercase, strip spaces/commas) and `exact_match` comparator.【F:src/phi_synth_math/tasks/eval/scoring.py†L4-L9】

## Registries
- Dataset registry (`DATASET_REGISTRY`): keys → constructors. Available: `dummy_math_addition`. Factory applies `max_int` default of 20 when not provided.【F:src/phi_synth_math/core/registry.py†L12-L28】
- Model registry (`MODEL_REGISTRY`): keys → constructors. Available: `dummy`.【F:src/phi_synth_math/core/registry.py†L16-L34】

## Evaluation pipeline (CLI to outputs)
- CLI `python -m scripts.run_eval --config <yaml>` loads the YAML into `EvalConfig`, ensures `src` on `sys.path`, and creates the next numeric run directory under `<results_root>/<task_name>/` while copying the config to `config.yaml`.【F:scripts/run_eval.py†L7-L35】
- `EvalRunner.run` builds dataset/model from registries, streams through dataset in batches of `batch_size`, and calls the model’s `generate` per batch.【F:src/phi_synth_math/tasks/eval/runner.py†L16-L47】
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
- Modify evaluation logic: extend `EvalRunner` or scoring helpers in `src/phi_synth_math/tasks/eval/` to change batching, metrics, or output formats.【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】【F:src/phi_synth_math/tasks/eval/scoring.py†L4-L9】
