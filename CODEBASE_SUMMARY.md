## High-level overview
- Evaluation harness loads a YAML config (Path-aware), instantiates a dataset and model from registries (dummy math or GSM8K + dummy/vLLM models), runs batch inference, and writes predictions plus metrics to a run directory.【F:scripts/run_eval.py†L31-L54】【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】【F:src/phi_synth_math/core/registry.py†L12-L53】
- Primary entrypoint: `python -m scripts.run_eval --config <path>` resolves the config path to a `Path`, converts the configured `results_root` to a `Path`, creates an incrementing run folder, snapshots the config, and executes the evaluation loop.【F:scripts/run_eval.py†L31-L54】
- GSM8K support adds numeric-answer prompting plus scoring that extracts the last number from predictions for comparison against the gold answer.【F:src/phi_synth_math/tasks/datasets/gsm8k.py†L28-L46】【F:src/phi_synth_math/tasks/eval/scoring.py†L11-L27】

## Directory structure (depth ≤4)
- `configs/` — evaluation configuration files (YAML).【F:configs/eval/dummy_math.yaml†L1-L10】【F:configs/eval/gsm8k_vllm_phi15.yaml†L1-L19】
  - `configs/eval/` — task-specific evaluation configs (dummy math, GSM8K + vLLM).【F:configs/eval/dummy_math.yaml†L1-L10】【F:configs/eval/gsm8k_vllm_phi15.yaml†L1-L19】
- `scripts/` — runnable scripts; `run_eval.py` drives evaluations from the CLI.【F:scripts/run_eval.py†L7-L54】
- `src/` — Python package source tree (add to `PYTHONPATH`).【F:README.md†L1-L39】
  - `src/phi_synth_math/` — root package for core utilities, models, and tasks.【F:src/phi_synth_math/__init__.py†L1-L5】
    - `core/` — config parsing, registries, run directory helpers, and JSONL utilities.【F:src/phi_synth_math/core/config.py†L10-L112】【F:src/phi_synth_math/core/run_dir.py†L7-L26】【F:src/phi_synth_math/core/registry.py†L12-L53】【F:src/phi_synth_math/core/jsonl.py†L8-L21】
    - `models/` — model protocol plus dummy and vLLM-backed implementations.【F:src/phi_synth_math/models/base.py†L6-L10】【F:src/phi_synth_math/models/dummy.py†L9-L24】【F:src/phi_synth_math/models/vllm_model.py†L8-L73】
    - `tasks/` — dataset definitions (dummy math, GSM8K), evaluation loop, and scoring helpers.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】【F:src/phi_synth_math/tasks/datasets/gsm8k.py†L8-L46】【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】【F:src/phi_synth_math/tasks/eval/scoring.py†L1-L27】
- `logs/` — captured job outputs from external runs (GPU test artifacts).【F:logs/test_gpu_cpu.out†L1-L60】

## Config system
- YAML configs:
  - `configs/eval/dummy_math.yaml`: defines a dummy math evaluation specifying task name, results root, RNG seed, number of examples, batch size, model name, dataset name, and dataset max integer.【F:configs/eval/dummy_math.yaml†L1-L10】
  - `configs/eval/gsm8k_vllm_phi15.yaml`: sample GSM8K evaluation pointing to a vLLM-backed Phi-1.5 model with sampling/parallelism parameters and dataset split selection.【F:configs/eval/gsm8k_vllm_phi15.yaml†L1-L19】
- Dataclasses (in code):
  - `ModelConfig`: `name` plus optional `model_name`, `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `dtype`, `max_tokens`, `temperature`, `top_p`, and `seed` for vLLM setup and sampling defaults.【F:src/phi_synth_math/core/config.py†L10-L38】
  - `DatasetConfig`: `name`, optional `max_int`, and optional `split` (e.g., for GSM8K).【F:src/phi_synth_math/core/config.py†L40-L44】【F:src/phi_synth_math/core/config.py†L80-L89】
  - `EvalConfig`: `task_name`, `results_root` (string from YAML boundary), `seed`, `n_examples`, `batch_size`, `model`, `dataset`. Validation enforces presence of required fields, numeric positivity checks, and typed optional parsing helpers; `load_eval_config` accepts a `Path` or string and resolves the file before parsing.【F:src/phi_synth_math/core/config.py†L46-L112】

## Core abstractions
- Dataset interface: iterable protocol yielding dicts with `id`, `question`, and `answer`.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】
- Model interface: protocol with `generate(questions, max_tokens=None) -> List[str]` for batch text outputs.【F:src/phi_synth_math/models/base.py†L6-L10】
- Eval runner: orchestrates dataset/model instantiation, batches questions, scores predictions via dataset-aware scoring, writes `predictions.jsonl`, aggregates metrics, and logs first 50 mistakes to `mistakes.txt` when present.【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】
- Scoring helpers: `normalize_answer`, `exact_match`, `extract_last_number`, and `score_prediction` (uses numeric extraction for GSM8K, otherwise exact match).【F:src/phi_synth_math/tasks/eval/scoring.py†L1-L27】

## Registries
- Dataset registry (`DATASET_REGISTRY`): keys → constructors. Available: `dummy_math_addition` (defaults `max_int=20` when absent) and `gsm8k` (respects split and seeded subsampling).【F:src/phi_synth_math/core/registry.py†L12-L37】
- Model registry (`MODEL_REGISTRY`): keys → constructors. Available: `dummy` and `vllm`, with explicit construction paths for per-model configuration fields.【F:src/phi_synth_math/core/registry.py†L39-L53】

## Evaluation pipeline (CLI to outputs)
- CLI `python -m scripts.run_eval --config <yaml>` loads the YAML into `EvalConfig`, ensures `src` on `sys.path`, expands the configured `results_root` to a `Path`, and creates the next numeric run directory under `<results_root>/<task_name>/` while copying the config to `config.yaml`.【F:scripts/run_eval.py†L7-L40】【F:src/phi_synth_math/core/run_dir.py†L7-L26】
- `EvalRunner.run` builds dataset/model from registries, streams through dataset in batches of `batch_size`, and calls the model’s `generate` per batch.【F:src/phi_synth_math/tasks/eval/runner.py†L16-L47】
- Each example produces a record with `id`, `question`, `gold`, `pred`, `correct` written to `predictions.jsonl` (JSON Lines).【F:src/phi_synth_math/tasks/eval/runner.py†L23-L103】
- Metrics (`accuracy`, `n_total`, `n_correct`) are written to `metrics.json` (pretty-printed JSON).【F:src/phi_synth_math/tasks/eval/runner.py†L49-L57】
- Up to 50 incorrect predictions are logged to `mistakes.txt` with tab-separated metadata when any mistakes occur.【F:src/phi_synth_math/tasks/eval/runner.py†L55-L63】【F:src/phi_synth_math/tasks/eval/runner.py†L79-L103】

## Math datasets
- Dummy dataset (`dummy_math_addition`): deterministic RNG (seeded) samples `n_examples` pairs of integers in `[0, max_int]`, emits question strings “What is a + b?” with stringified sum as `answer` and sequential `id`s (`ex_000001`, ...).【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】
- GSM8K (`gsm8k`): wraps `datasets.load_dataset("gsm8k", "main", split=<split>)`, formats prompts to request only the final numeric answer, extracts the final answer text after `####`, and performs seeded subsampling when `n_examples` is smaller than the split size.【F:src/phi_synth_math/tasks/datasets/gsm8k.py†L8-L46】

## Models
- Dummy model (`dummy`): regex-extracts two integers from prompts of the form “what is X + Y” (case-insensitive) and returns their sum; otherwise outputs “I don't know”.【F:src/phi_synth_math/models/dummy.py†L9-L24】
- vLLM-backed model (`vllm`): constructs `vllm.LLM` with optional tensor-parallel, memory utilization, max length, and dtype settings; generates batches with `SamplingParams` built from config/default sampling fields (max tokens, temperature, top_p, seed) and returns the first sampled output text per prompt.【F:src/phi_synth_math/models/vllm_model.py†L8-L73】

## Reproducibility & determinism
- Seeding: datasets use `random.Random(seed)` for repeatable number generation/subsampling; dummy model has deterministic regex logic, so outputs repeat for identical inputs and batches.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L16-L27】【F:src/phi_synth_math/tasks/datasets/gsm8k.py†L18-L46】【F:src/phi_synth_math/models/dummy.py†L12-L24】
- Run directory determinism: run IDs auto-increment based on existing numeric subfolders; repeated runs with the same config create new numbered folders but content is deterministic given identical seeds/configs.【F:src/phi_synth_math/core/run_dir.py†L7-L20】
- No global seeding beyond dataset RNG; randomness confined to dataset sampling and any model-level randomness controlled by optional seeds (e.g., vLLM sampling seed).【F:src/phi_synth_math/models/vllm_model.py†L37-L73】

## Extensibility notes
- Add datasets: implement the `Dataset` protocol (yield `id`, `question`, `answer`) under `src/phi_synth_math/tasks/datasets/` and register in `DATASET_REGISTRY` with a unique key.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】【F:src/phi_synth_math/core/registry.py†L12-L37】
- Add models: implement the `Model` protocol under `src/phi_synth_math/models/` and register in `MODEL_REGISTRY`.【F:src/phi_synth_math/models/base.py†L6-L10】【F:src/phi_synth_math/core/registry.py†L39-L53】
- Modify evaluation logic: extend `EvalRunner` or scoring helpers in `src/phi_synth_math/tasks/eval/` to change batching, metrics, or output formats (e.g., dataset-aware scoring hooks).【F:src/phi_synth_math/tasks/eval/runner.py†L16-L103】【F:src/phi_synth_math/tasks/eval/scoring.py†L1-L27】
