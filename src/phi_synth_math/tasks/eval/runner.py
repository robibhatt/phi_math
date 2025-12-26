from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, TextIO

from phi_synth_math.core.config import EvalConfig
from phi_synth_math.core.registry import make_dataset, make_model
from phi_synth_math.models.base import Model
from phi_synth_math.tasks.eval.scoring import exact_match


class EvalRunner:
    """Runs evaluation for a given config and run directory."""

    def run(self, config: EvalConfig, run_dir: str) -> dict[str, Any]:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        dataset = make_dataset(config.dataset, n_examples=config.n_examples, seed=config.seed)
        model = make_model(config.model)

        predictions_path = run_path / "predictions.jsonl"
        metrics_path = run_path / "metrics.json"
        mistakes_path = run_path / "mistakes.txt"

        n_total = 0
        n_correct = 0
        mistakes: List[str] = []

        batch_questions: List[str] = []
        batch_examples: List[dict[str, Any]] = []

        with predictions_path.open("w", encoding="utf-8") as pred_file:
            for example in dataset:
                batch_questions.append(example["question"])
                batch_examples.append(example)

                if len(batch_questions) >= config.batch_size:
                    batch_result = self._process_batch(model, batch_examples, batch_questions)
                    n_total, n_correct = self._write_results(batch_result, pred_file, mistakes, n_total, n_correct)
                    batch_questions = []
                    batch_examples = []

            if batch_questions:
                batch_result = self._process_batch(model, batch_examples, batch_questions)
                n_total, n_correct = self._write_results(batch_result, pred_file, mistakes, n_total, n_correct)

        metrics = {
            "accuracy": (n_correct / n_total) if n_total > 0 else 0.0,
            "n_total": n_total,
            "n_correct": n_correct,
        }

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if mistakes:
            mistakes_path.parent.mkdir(parents=True, exist_ok=True)
            with mistakes_path.open("w", encoding="utf-8") as f:
                for line in mistakes[:50]:
                    f.write(line + "\n")

        return metrics

    def _process_batch(
        self,
        model: Model,
        examples: List[dict[str, Any]],
        questions: List[str],
    ) -> List[tuple[dict[str, Any], str, bool]]:
        predictions = model.generate(questions)
        results: List[tuple[dict[str, Any], str, bool]] = []
        for example, pred in zip(examples, predictions):
            correct = exact_match(pred, example["answer"])
            results.append((example, pred, correct))
        return results

    def _write_results(
        self,
        batch_result: List[tuple[dict[str, Any], str, bool]],
        pred_file: TextIO,
        mistakes: List[str],
        n_total: int,
        n_correct: int,
    ) -> tuple[int, int]:
        for example, pred, correct in batch_result:
            record = {
                "id": example["id"],
                "question": example["question"],
                "gold": example["answer"],
                "pred": pred,
                "correct": correct,
            }
            pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not correct and len(mistakes) < 50:
                mistakes.append(
                    f"{example['id']}\tQ: {example['question']}\tGold: {example['answer']}\tPred: {pred}"
                )
            n_total += 1
            if correct:
                n_correct += 1
        return n_total, n_correct
