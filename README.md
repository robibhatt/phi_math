# phi-synth-math

Minimal, YAML-driven evaluation skeleton for synthetic math tasks. It reads an eval config, creates a timestamped run directory, evaluates a dummy model on a toy dataset, and writes predictions and metrics.

## Quickstart

From the repository root:

```bash
python -m scripts.run_eval --config configs/eval/dummy_math.yaml
```

This will:

1. Load the YAML config.
2. Create an auto-incremented run directory under `results/eval/<task_name>/<run_id>/`.
3. Save a copy of the config to `config.yaml` inside the run directory.
4. Run a small evaluation loop with a deterministic dummy model.
5. Write:
   - `predictions.jsonl`
   - `metrics.json`
   - `mistakes.txt` (only when there are incorrect predictions)

### Output structure

```
results/
  eval/
    dummy_math/
      1/
        config.yaml
        predictions.jsonl
        metrics.json
        mistakes.txt
      2/
        ...
```

## Adding new tasks

Benchmarks are registered through a `TaskSpec` (see `phi_synth_math/tasks/core/metadata.py`) that bundles everything the runner needs:

- **dataset_builder**: Callable accepting `(n_examples: int, seed: int, **dataset_params)` and yielding dicts with `id`, `question`, and `answer`. The `question` should be the raw problem text; the runner applies prompts separately.
- **scorer**: Callable `(pred: str, gold: str) -> bool` used to evaluate each example.
- **default_dataset_params**: Mapping of dataset kwargs applied unless overridden in YAML (e.g., `split` or `max_int`).
- **prompt_template**: Format string applied as `prompt_template.format(question=question_text)` before sending to the model.

To add a task, create the dataset/scorer, define its `TaskSpec`, and add it to `TASK_SPECS`.
