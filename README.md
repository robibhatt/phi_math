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
