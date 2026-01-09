"""Integration tests for EvalRunner using DummyModel."""

import json
from pathlib import Path

import pytest
import yaml

from phi_synth_math.core.config import load_eval_config
from phi_synth_math.tasks.core.runner import EvalRunner


class TestEvalRunnerIntegration:
    """Integration tests for the full evaluation pipeline."""

    @pytest.mark.integration
    def test_run_dummy_addition_full_pipeline(self, tmp_dir: Path):
        """Test complete pipeline with DummyModel on dummy_math_addition task."""
        # Create config
        config_data = {
            "task_name": "dummy_math_addition",
            "results_root": str(tmp_dir),
            "seed": 42,
            "n_examples": 5,
            "batch_size": 2,
            "model": {"name": "dummy"},
            "dataset": {"name": "dummy_math_addition", "max_int": 10, "split": "test"},
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        config = load_eval_config(config_path)
        run_dir = tmp_dir / "run_1"
        run_dir.mkdir()

        runner = EvalRunner()
        metrics = runner.run(config, run_dir)

        # DummyModel should get 100% accuracy on dummy_math_addition
        assert metrics["accuracy"] == 1.0
        assert metrics["n_total"] == 5
        assert metrics["n_correct"] == 5

        # Verify output files exist
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "predictions.txt").exists()
        assert (run_dir / "metrics.json").exists()

        # Verify predictions.jsonl content
        with (run_dir / "predictions.jsonl").open() as f:
            predictions = [json.loads(line) for line in f]
        assert len(predictions) == 5
        for pred in predictions:
            assert pred["correct"] is True
            assert "id" in pred
            assert "question" in pred
            assert "gold" in pred
            assert "pred" in pred

    @pytest.mark.integration
    def test_run_with_few_shot_prompting(self, tmp_dir: Path):
        """Test pipeline with few-shot prompting using static examples."""
        config_data = {
            "task_name": "gsm8k",
            "results_root": str(tmp_dir),
            "seed": 42,
            "n_examples": 3,
            "batch_size": 3,
            "model": {"name": "dummy"},
            "dataset": {"name": "dummy_math_addition", "max_int": 5, "split": "test"},
            "prompt": {
                "few_shot_count": 2,
                "static_examples": "gsm8k/8shot_cot",
                "example_format": "Q: {question}\nA: {answer}\n\n",
                "test_format": "Q: {question}\nA:",
            },
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        config = load_eval_config(config_path)
        run_dir = tmp_dir / "run_2"
        run_dir.mkdir()

        runner = EvalRunner()
        metrics = runner.run(config, run_dir)

        # Should still work (DummyModel extracts numbers regardless of prompt format)
        assert metrics["n_total"] == 3
        assert (run_dir / "predictions.jsonl").exists()

        # Verify the prompts contain few-shot examples
        with (run_dir / "predictions.jsonl").open() as f:
            predictions = [json.loads(line) for line in f]
        # The question field should contain the full prompt with examples
        for pred in predictions:
            # Should have multiple Q: patterns from few-shot examples
            assert pred["question"].count("Q:") >= 2

    @pytest.mark.integration
    def test_batch_processing_handles_tail_batch(self, tmp_dir: Path):
        """Test that partial final batch is processed correctly."""
        config_data = {
            "task_name": "dummy_math_addition",
            "results_root": str(tmp_dir),
            "seed": 42,
            "n_examples": 7,  # Not divisible by batch_size
            "batch_size": 3,  # Will have batches: 3, 3, 1 (tail)
            "model": {"name": "dummy"},
            "dataset": {"name": "dummy_math_addition", "max_int": 10, "split": "test"},
        }
        config_path = tmp_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config_data, f)

        config = load_eval_config(config_path)
        run_dir = tmp_dir / "run_3"
        run_dir.mkdir()

        runner = EvalRunner()
        metrics = runner.run(config, run_dir)

        # All 7 examples should be processed
        assert metrics["n_total"] == 7
        assert metrics["n_correct"] == 7

        with (run_dir / "predictions.jsonl").open() as f:
            predictions = [json.loads(line) for line in f]
        assert len(predictions) == 7
