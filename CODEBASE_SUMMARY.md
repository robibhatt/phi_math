# 1. Project Overview
- Minimal evaluation harness for math tasks (“phi-synth-math”) that reads a YAML config, instantiates a dataset and model, runs batched inference, and writes predictions plus metrics to a run directory.【F:scripts/run_eval.py†L22-L54】【F:src/phi_synth_math/tasks/shared/runner.py†L16-L103】
- Focus areas:
  - Math evaluations: dummy addition generator and GSM8K prompting/scoring for numeric answers.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L18-L49】
  - Phi-style models: exemplar config targets `microsoft/phi-1_5` via vLLM backend.【F:configs/eval/gsm8k_vllm_phi15.yaml†L6-L16】
- High-level workflow: CLI loads config → create run dir (auto-incremented) → snapshot config → instantiate dataset/model from registries → iterate dataset in batches → model.generate → score predictions → write predictions/metrics/mistakes.【F:scripts/run_eval.py†L22-L54】【F:src/phi_synth_math/core/run_dir.py†L7-L26】【F:src/phi_synth_math/core/registry.py†L12-L53】【F:src/phi_synth_math/tasks/shared/runner.py†L16-L103】

# 2. Repository Layout
- Core
  - `scripts/run_eval.py`: CLI entrypoint driving config load, run directory creation, and evaluation invocation.【F:scripts/run_eval.py†L7-L54】
  - `src/phi_synth_math/core/`: config parsing/validation, registry construction, run directory helpers, JSONL utilities.【F:src/phi_synth_math/core/config.py†L10-L112】【F:src/phi_synth_math/core/registry.py†L12-L53】【F:src/phi_synth_math/core/run_dir.py†L7-L26】【F:src/phi_synth_math/core/jsonl.py†L8-L21】
  - `src/phi_synth_math/tasks/`: datasets (dummy math, GSM8K), evaluation runner, scoring logic.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L18-L49】【F:src/phi_synth_math/tasks/shared/runner.py†L16-L103】【F:src/phi_synth_math/tasks/shared/scoring.py†L1-L34】
  - `src/phi_synth_math/models/`: model protocol plus dummy and vLLM-backed implementations.【F:src/phi_synth_math/models/base.py†L6-L10】【F:src/phi_synth_math/models/dummy.py†L9-L24】【F:src/phi_synth_math/models/vllm_model.py†L8-L73】
- Supporting
  - `configs/`: ready-to-run YAMLs (dummy math, GSM8K+Phi via vLLM).【F:configs/eval/dummy_math.yaml†L1-L10】【F:configs/eval/gsm8k_vllm_phi15.yaml†L1-L19】
  - `README.md`: quickstart and output structure summary.【F:README.md†L1-L39】
  - `logs/`: captured external job outputs (non-critical to pipeline).【F:logs/test_gpu_cpu.out†L1-L60】

# 3. Evaluation Pipeline (CRITICAL)
- Launch/entry:
  - `python -m scripts.run_eval --config <yaml>` ensures `src/` on `PYTHONPATH`, resolves YAML path, and loads it into `EvalConfig`.【F:scripts/run_eval.py†L7-L37】
  - Creates next numeric run directory under `<results_root>/<task_name>/` and copies the YAML to `config.yaml` for reproducibility.【F:scripts/run_eval.py†L35-L40】【F:src/phi_synth_math/core/run_dir.py†L7-L26】
- Config system:
  - `load_eval_config` parses YAML into dataclasses with validation for required fields and positivity of `n_examples`/`batch_size`; optional model/dataset fields are typed helpers.【F:src/phi_synth_math/core/config.py†L10-L112】
- Dataset loading/preprocessing:
  - `make_dataset` chooses constructor from `DATASET_REGISTRY` and applies dataset-specific defaults (dummy `max_int=20` when unspecified; GSM8K `split="test"` by default).【F:src/phi_synth_math/core/registry.py†L12-L37】
  - Dummy dataset synthesizes addition questions with seeded RNG; GSM8K loads Hugging Face dataset, extracts final numeric answer (post-`####`), formats prompt requesting numeric answer only, and applies deterministic subsampling when limiting `n_examples`.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L18-L49】
- Prompt construction:
  - Dummy dataset produces plain “What is a + b?” text; GSM8K prefixes instructions (“Solve the problem. Give ONLY the final numeric answer.”) and appends “Answer:” to cue direct responses.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L16-L23】【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L30-L44】
- Model invocation:
  - `make_model` maps `ModelConfig` to registered models; dummy returns deterministic regex-based solver, vLLM model wraps `vllm.LLM` with optional tensor-parallel/memory/dtype settings and sampling controls (max_tokens/temperature/top_p/seed).【F:src/phi_synth_math/core/registry.py†L39-L53】【F:src/phi_synth_math/models/dummy.py†L9-L24】【F:src/phi_synth_math/models/vllm_model.py†L16-L73】
- Batching + repeat logic:
  - `EvalRunner.run` accumulates `batch_size` questions before calling `model.generate`; leftover partial batch is processed after iteration ends.【F:src/phi_synth_math/tasks/shared/runner.py†L28-L47】
  - Per-batch processing checks length alignment of predictions vs. examples to guard against model misbehavior.【F:src/phi_synth_math/tasks/shared/runner.py†L66-L75】
- Output collection:
  - For each example, writes JSONL record with `id`, `question`, `gold`, `pred`, `correct` to `predictions.jsonl` in the run directory.【F:src/phi_synth_math/tasks/shared/runner.py†L77-L93】
  - First 50 mistakes captured in-memory and flushed to `mistakes.txt` for quick inspection.【F:src/phi_synth_math/tasks/shared/runner.py†L49-L63】【F:src/phi_synth_math/tasks/shared/runner.py†L77-L103】
- Metric computation:
  - Tracks running counts; final metrics dict: `accuracy` (n_correct / n_total), `n_total`, `n_correct` written to `metrics.json` (pretty JSON).【F:src/phi_synth_math/tasks/shared/runner.py†L43-L63】
  - Scoring uses dataset-aware `score_prediction`: GSM8K compares normalized last-number extraction; all others use normalized exact match.【F:src/phi_synth_math/tasks/shared/scoring.py†L9-L34】

# 4. Model Abstractions
- Protocol: `Model.generate(questions: List[str], max_tokens: int | None = None) -> List[str>` defines batch API.【F:src/phi_synth_math/models/base.py†L6-L10】
- Dummy model: regex parses “what is X + Y” (case-insensitive) and returns stringified sum or “I don't know”. Deterministic per input; ignores `max_tokens`.【F:src/phi_synth_math/models/dummy.py†L9-L24】
- VLLM model:
  - Constructor requires `model_name`; forwards optional tensor parallelism, GPU memory utilization, max model length, and dtype to `vllm.LLM`.【F:src/phi_synth_math/models/vllm_model.py†L16-L46】
  - Sampling parameters built from config/defaults: `max_tokens` (default 16 if unspecified), `temperature`, `top_p`, `seed`; uses first output text from each generation. Suitable for Phi-1.5 via config example but general to any vLLM-compatible model.【F:src/phi_synth_math/models/vllm_model.py†L48-L73】【F:configs/eval/gsm8k_vllm_phi15.yaml†L6-L16】
- Model selection keyed by `model.name` in YAML mapped through `MODEL_REGISTRY` (`dummy`, `vllm`).【F:src/phi_synth_math/core/registry.py†L39-L53】

# 5. Data & Datasets
- Interface: iterable yielding dicts with `id`, `question`, `answer` fields.【F:src/phi_synth_math/tasks/datasets/base.py†L6-L10】
- Supported datasets:
  - `dummy_math_addition`: synthetic addition samples with bounded integers, deterministic via seed; IDs formatted `ex_######`. Configurable `max_int`.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L10-L27】【F:configs/eval/dummy_math.yaml†L6-L10】
  - `gsm8k`: wraps Hugging Face GSM8K “main” split; prompt urges numeric final answer; gold extracted after delimiter “####”. Supports `split` selection and seeded subsampling to `n_examples`.【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L18-L49】【F:configs/eval/gsm8k_vllm_phi15.yaml†L17-L19】
- Dataset selection uses YAML `dataset.name` mapped via `DATASET_REGISTRY`; dataset-specific optional fields (`max_int`, `split`) parsed/validated in `DatasetConfig`.【F:src/phi_synth_math/core/config.py†L40-L89】【F:src/phi_synth_math/core/registry.py†L12-L37】

# 6. Metrics & Judging
- Normalization: lowercase + remove spaces/commas before comparison (`normalize_answer`).【F:src/phi_synth_math/tasks/shared/scoring.py†L1-L13】
- Exact match: normalized string equality for most datasets.【F:src/phi_synth_math/tasks/shared/scoring.py†L9-L18】
- GSM8K-specific: extract last numeric substring from prediction/gold (strip commas) then normalized equality; falls back to exact match when numbers missing. Captures robustness to reasoning traces with trailing numeric answers.【F:src/phi_synth_math/tasks/shared/scoring.py†L15-L34】
- Aggregation: accuracy = n_correct / n_total plus raw counts stored in `metrics.json`.【F:src/phi_synth_math/tasks/shared/runner.py†L43-L63】

# 7. Results & Artifacts
- Run directory: `<results_root>/<task_name>/<run_id>/` where `run_id` auto-increments based on existing numeric folders.【F:src/phi_synth_math/core/run_dir.py†L7-L20】
- Persisted files:
  - `config.yaml`: snapshot of the input YAML.【F:src/phi_synth_math/core/run_dir.py†L21-L26】
  - `predictions.jsonl`: one JSON record per example with question, gold, prediction, and correctness flag.【F:src/phi_synth_math/tasks/shared/runner.py†L77-L93】
  - `metrics.json`: accuracy and counts (pretty-printed JSON).【F:src/phi_synth_math/tasks/shared/runner.py†L43-L57】
  - `mistakes.txt`: up to 50 incorrect predictions, tab-separated for quick inspection (only created when mistakes exist).【F:src/phi_synth_math/tasks/shared/runner.py†L49-L63】【F:src/phi_synth_math/tasks/shared/runner.py†L77-L103】
- README illustrates expected directory tree under `results/eval/<task>/run_id/`.【F:README.md†L11-L34】

# 8. Configuration Files
- `configs/eval/dummy_math.yaml`: dummy model + dummy dataset; 25 examples, batch size 8, `max_int` 20; outputs under `results/eval/dummy_math/<run_id>/`.【F:configs/eval/dummy_math.yaml†L1-L10】
- `configs/eval/gsm8k_vllm_phi15.yaml`: vLLM with `microsoft/phi-1_5`, tensor parallel 1, GPU memory util 0.8, max length 1024, deterministic sampling (temperature 0, top_p 1, seed 0/None), GSM8K test split with 200 examples and batch size 32.【F:configs/eval/gsm8k_vllm_phi15.yaml†L1-L19】
- Config parsing lives in `core/config.py`; missing required fields raise errors, and optional fields are coerced to proper types with positivity checks for counts/lengths.【F:src/phi_synth_math/core/config.py†L10-L112】

# 9. Execution Examples
- Smoke/dummy eval:
  ```bash
  PYTHONPATH=src python -m scripts.run_eval --config configs/eval/dummy_math.yaml
  ```
  Produces numbered run under `results/eval/dummy_math/` with predictions/metrics/mistakes files.【F:scripts/run_eval.py†L35-L54】【F:configs/eval/dummy_math.yaml†L1-L10】
- GSM8K with Phi-1.5 via vLLM (requires vLLM + model weights available):
  ```bash
  PYTHONPATH=src python -m scripts.run_eval --config configs/eval/gsm8k_vllm_phi15.yaml
  ```
  Writes run under `results/eval/gsm8k_vllm_phi15/` using vLLM backend for generation.【F:scripts/run_eval.py†L35-L54】【F:configs/eval/gsm8k_vllm_phi15.yaml†L1-L19】

# 10. Known Assumptions / Constraints
- Hardware/backends:
  - vLLM backend expects a compatible GPU environment; config exposes tensor parallelism and GPU memory utilization hints (no CPU fallback coded).【F:src/phi_synth_math/models/vllm_model.py†L16-L46】
  - GSM8K dataset download requires network access via `datasets.load_dataset`.【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L18-L28】
- Sampling/batching limits:
  - Default `max_tokens` for vLLM generations is 16 when unspecified; can override via config or call-time parameter.【F:src/phi_synth_math/models/vllm_model.py†L48-L62】
  - Batch sizing strictly follows YAML `batch_size`; no dynamic padding or streaming—datasets are iterated sequentially with batch accumulation and final partial batch flush.【F:src/phi_synth_math/tasks/shared/runner.py†L28-L47】
- Determinism:
  - Seeds propagate to dataset sampling (dummy RNG, GSM8K subsampling) and optionally to vLLM sampling; run directories auto-increment to avoid overwrites rather than reusing IDs.【F:src/phi_synth_math/tasks/datasets/dummy_math_addition.py†L16-L27】【F:src/phi_synth_math/tasks/gsm8k/dataset.py†L18-L49】【F:src/phi_synth_math/models/vllm_model.py†L48-L73】【F:src/phi_synth_math/core/run_dir.py†L7-L20】
