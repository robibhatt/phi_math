from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, TextIO

from phi_synth_math.core.config import EvalConfig
from phi_synth_math.core.registry import make_dataset, make_model
from phi_synth_math.models.base import Model
from phi_synth_math.tasks.dummy_addition import scoring as dummy_addition_scoring
from phi_synth_math.tasks.gsm8k import scoring as gsm8k_scoring
from phi_synth_math.tasks.shared import scoring as shared_scoring


def score_prediction(dataset_name: str, pred: str, gold: str) -> bool:
    if dataset_name == "gsm8k":
        return gsm8k_scoring.score(pred, gold)
    if dataset_name == "dummy_math_addition":
        return dummy_addition_scoring.score(pred, gold)
    return shared_scoring.exact_match(pred, gold)


class EvalRunner:
    """Runs evaluation for a given config and run directory."""

    def run(self, config: EvalConfig, run_dir: Path) -> dict[str, Any]:
        # ------------------------------------------------------------------
        # Basic config sanity (fail fast with a clear message)
        # ------------------------------------------------------------------
        if getattr(config, "batch_size", None) is None:
            raise ValueError("EvalConfig.batch_size is missing.")
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {config.batch_size}).")

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
                # ------------------------------------------------------------------
                # Dataset contract validation (clear error if a dataset breaks it)
                # ------------------------------------------------------------------
                if not isinstance(example, dict):
                    raise TypeError(f"Dataset yielded non-dict example: {type(example)}")
                for k in ("id", "question", "answer"):
                    if k not in example:
                        raise KeyError(
                            f"Dataset example missing key '{k}'. "
                            f"Present keys: {sorted(example.keys())}"
                        )

                batch_questions.append(example["question"])
                batch_examples.append(example)

                if len(batch_questions) >= config.batch_size:
                    batch_result = self._process_batch(
                        model=model,
                        examples=batch_examples,
                        questions=batch_questions,
                        dataset_name=config.dataset.name,
                        # Prefer model max_tokens from config if present.
                        max_tokens=getattr(config.model, "max_tokens", None),
                    )
                    n_total, n_correct = self._write_results(
                        batch_result, pred_file, mistakes, n_total, n_correct
                    )
                    batch_questions = []
                    batch_examples = []

            # Tail batch
            if batch_questions:
                batch_result = self._process_batch(
                    model=model,
                    examples=batch_examples,
                    questions=batch_questions,
                    dataset_name=config.dataset.name,
                    max_tokens=getattr(config.model, "max_tokens", None),
                )
                n_total, n_correct = self._write_results(
                    batch_result, pred_file, mistakes, n_total, n_correct
                )

        metrics = {
            "accuracy": (n_correct / n_total) if n_total > 0 else 0.0,
            "n_total": n_total,
            "n_correct": n_correct,
        }

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if mistakes:
            with mistakes_path.open("w", encoding="utf-8") as f:
                for line in mistakes[:50]:
                    f.write(line + "\n")

        return metrics

    def _process_batch(
        self,
        model: Model,
        examples: List[dict[str, Any]],
        questions: List[str],
        dataset_name: str,
        max_tokens: int | None = None,
    ) -> List[tuple[dict[str, Any], str, bool]]:
        # Helpful context if generation fails
        ids_preview = [ex.get("id", "<missing-id>") for ex in examples[:10]]
        try:
            # Pass through max_tokens if the backend honors it.
            predictions = model.generate(questions, max_tokens=max_tokens)
        except Exception as e:
            raise RuntimeError(
                "Model.generate failed for a batch. "
                f"dataset={dataset_name}, batch_size={len(questions)}, "
                f"example_ids_preview={ids_preview}"
            ) from e

        if len(predictions) != len(examples):
            raise RuntimeError(
                f"Model returned {len(predictions)} predictions for {len(examples)} examples. "
                f"dataset={dataset_name}, example_ids_preview={ids_preview}"
            )

        results: List[tuple[dict[str, Any], str, bool]] = []
        for example, pred in zip(examples, predictions):
            correct = score_prediction(dataset_name, pred, example["answer"])
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
