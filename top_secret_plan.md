# Top Secret Refactor Plan: Data & Config Architecture

## What We Did Today

1. **Refactored prompt examples** from Python files to JSON files
   - `gsm8k/prompts/8shot_cot.json` instead of `few_shot_examples.py`
   - Auto-discovery via path: `static_examples: "gsm8k/8shot_cot"`
   - No more manual registry in `prompt_builder.py`

2. **Cleaned up configs** - single `gsm8k.yaml` instead of per-prompt-variant files

---

## The Bigger Refactor We're Planning

### Problem with Current Approach
- Each dataset needs a Python class (`GSM8KDataset`, `DummyMathAdditionDataset`)
- `metadata.py` manually wires dataset_builder + scorer + prompt_template
- Adding new dataset = new Python files + registry updates
- Synthetic data generation outputting Python files is weird

### Inspiration: How the Pros Do It
- **lm-evaluation-harness**: Everything is YAML. Task = config file pointing to data + prompt template + scorer
- **Anthropic evals**: Data is just JSONL files, organized thematically

### Target Architecture

```
data/
├── gsm8k/
│   ├── train.jsonl
│   └── test.jsonl
└── synthetic/
    └── experiment_v1/
        ├── train.jsonl
        └── test.jsonl

configs/
├── eval/
│   └── gsm8k.yaml          # ONE file = complete experiment
└── finetune/
    └── (future fine-tuning configs)

src/phi_synth_math/
├── data/
│   └── loader.py           # ONE generic JSONLDataset (no per-task classes)
├── scoring/
│   ├── registry.py
│   └── numeric.py          # extract_numeric, exact_match, etc.
└── eval/
    └── runner.py
```

### Single Eval Config (One YAML = One Experiment)

```yaml
# configs/eval/gsm8k.yaml
data:
  path: data/gsm8k
  split: test

prompt:
  template: "Q: {{question}}\nA:"
  few_shot: gsm8k/8shot_cot
  few_shot_count: 5

scoring: extract_numeric

model:
  name: vllm
  model_name: microsoft/phi-1_5
  ...
```

---

## Open Design Questions (To Discuss Next Time)

1. **Few-shot config** - where does it live? Task defaults + eval overrides?

2. **HuggingFace → JSONL conversion** - one-time script? on-demand caching? manual?

3. **Scorer registration** - simple dict or fancy decorators?
   ```python
   # Simple
   SCORERS = {"extract_numeric": extract_numeric}

   # Fancy
   @register_scorer("extract_numeric")
   def extract_numeric(pred, gold): ...
   ```

---

## Decisions Made

- One YAML per experiment (no artificial task/ vs eval/ split)
- Keep `configs/eval/` as a directory (since we'll also have `configs/finetune/`)
- Data should be JSONL files, not Python classes
- Generic loader that knows nothing about task semantics
