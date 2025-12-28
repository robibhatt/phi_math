from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, TextIO

from phi_synth_math.core.config import EvalConfig
from phi_synth_math.core.registry import make_dataset, make_model
from phi_synth_math.models.base import Model
from phi_synth_math.tasks.core.metadata import get_task_spec
from phi_synth_math.tasks.core.predictions_formatter import format_predictions
from phi_synth_math.tasks.core.prompt_builder import PromptBuilder


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

        task_spec = get_task_spec(config.dataset.name)

        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        # Set up prompt builder if few-shot prompting is configured
        prompt_builder: PromptBuilder | None = None
        if config.prompt is not None:
            prompt_builder = PromptBuilder(config.prompt, task_spec)
            prompt_builder.load_few_shot_examples(config.seed)

        dataset = make_dataset(config.dataset, n_examples=config.n_examples, seed=config.seed)
        model = make_model(config.model)

        predictions_path = run_path / "predictions.jsonl"
        predictions_txt_path = run_path / "predictions.txt"
        metrics_path = run_path / "metrics.json"

        n_total = 0
        n_correct = 0

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

                batch_questions.append(str(example["question"]))
                batch_examples.append(example)

                if len(batch_questions) >= config.batch_size:
                    batch_result = self._process_batch(
                        model=model,
                        examples=batch_examples,
                        questions=batch_questions,
                        dataset_name=config.dataset.name,
                        prompt_template=task_spec.prompt_template,
                        prompt_builder=prompt_builder,
                        scorer=task_spec.scorer,
                        # Prefer model max_tokens from config if present.
                        max_tokens=getattr(config.model, "max_tokens", None),
                    )
                    n_total, n_correct = self._write_results(
                        batch_result, pred_file, n_total, n_correct
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
                    prompt_template=task_spec.prompt_template,
                    prompt_builder=prompt_builder,
                    scorer=task_spec.scorer,
                    max_tokens=getattr(config.model, "max_tokens", None),
                )
                n_total, n_correct = self._write_results(
                    batch_result, pred_file, n_total, n_correct
                )

        metrics = {
            "accuracy": (n_correct / n_total) if n_total > 0 else 0.0,
            "n_total": n_total,
            "n_correct": n_correct,
        }

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Generate human-readable predictions.txt
        format_predictions(predictions_path, predictions_txt_path)

        return metrics

    def _process_batch(
        self,
        model: Model,
        examples: List[dict[str, Any]],
        questions: List[str],
        dataset_name: str,
        prompt_template: str,
        prompt_builder: PromptBuilder | None,
        scorer: Callable[[str, str], bool],
        max_tokens: int | None = None,
    ) -> List[tuple[dict[str, Any], str, str, bool]]:
        # Helpful context if generation fails
        ids_preview = [ex.get("id", "<missing-id>") for ex in examples[:10]]
        prompts = []
        for q in questions:
            try:
                if prompt_builder is not None:
                    prompts.append(prompt_builder.build_prompt(q))
                else:
                    prompts.append(prompt_template.format(question=q))
            except Exception as e:
                raise ValueError(
                    f"Failed to format prompt for dataset '{dataset_name}' with template: {prompt_template}"
                ) from e
        try:
            # Pass through max_tokens if the backend honors it.
            predictions = model.generate(prompts, max_tokens=max_tokens)
        except Exception as e:
            raise RuntimeError(
                "Model.generate failed for a batch. "
                f"dataset={dataset_name}, batch_size={len(prompts)}, "
                f"example_ids_preview={ids_preview}"
            ) from e

        if len(predictions) != len(prompts):
            raise RuntimeError(
                f"Model returned {len(predictions)} predictions for {len(prompts)} examples. "
                f"dataset={dataset_name}, example_ids_preview={ids_preview}"
            )

        results: List[tuple[dict[str, Any], str, str, bool]] = []
        for example, prompt, pred in zip(examples, prompts, predictions):
            correct = scorer(pred, example["answer"])
            results.append((example, prompt, pred, correct))
        return results

    def _write_results(
        self,
        batch_result: List[tuple[dict[str, Any], str, str, bool]],
        pred_file: TextIO,
        n_total: int,
        n_correct: int,
    ) -> tuple[int, int]:
        for example, prompt, pred, correct in batch_result:
            record = {
                "id": example["id"],
                "question": prompt,
                "gold": example["answer"],
                "pred": pred,
                "correct": correct,
            }
            pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_total += 1
            if correct:
                n_correct += 1

        return n_total, n_correct
