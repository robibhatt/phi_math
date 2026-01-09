# Test Templates for New Benchmarks

This directory contains template files for adding tests when creating new benchmarks.

## When Adding a New Benchmark

1. **Create your benchmark** in `src/phi_synth_math/tasks/benchmarks/{benchmark_name}/`
   - `__init__.py`
   - `dataset.py` - Dataset class
   - `scoring.py` - `score(pred, gold) -> bool` function
   - `prompts/` (optional) - Static few-shot examples as JSON

2. **Register your benchmark** in `src/phi_synth_math/tasks/core/metadata.py`
   - Add entry to `TASK_SPECS` dict

3. **Copy and customize test template**
   ```bash
   cp tests/templates/test_benchmark_template.py tests/unit/scoring/test_{benchmark_name}_scoring.py
   ```

4. **Update the copied file**
   - Replace `{benchmark_name}` with your benchmark name
   - Replace `{BenchmarkName}` with PascalCase version
   - Uncomment and update the test cases
   - Remove the template docstring header

## Template Checklist

When customizing the template, ensure you test:

- [ ] Correct predictions score as True
- [ ] Incorrect predictions score as False
- [ ] Edge cases (whitespace, formatting, special characters)
- [ ] Number extraction (if applicable)
- [ ] Dataset yields valid examples with id, question, answer fields
- [ ] Dataset is deterministic with same seed

## Example: GSM8K Tests

See `tests/unit/scoring/test_gsm8k_scoring.py` for a complete example of benchmark-specific tests.

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run with coverage
pytest --cov=phi_synth_math --cov-report=term-missing
```
