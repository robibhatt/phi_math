"""Prompt builder for few-shot prompting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from phi_synth_math.core.config import PromptConfig
from phi_synth_math.tasks.core.metadata import TaskSpec


def _load_static_examples(name: str) -> list[dict[str, str]]:
    """Load static examples from a JSON file.

    Args:
        name: Path in format "task/prompt_name" (e.g., "gsm8k/8shot_cot").
              Resolves to: benchmarks/{task}/prompts/{prompt_name}.json

    Returns:
        List of example dicts with "question" and "answer" keys.

    Raises:
        ValueError: If name format is invalid.
        FileNotFoundError: If the JSON file doesn't exist.
    """
    parts = name.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"static_examples must be in 'task/name' format (e.g., 'gsm8k/8shot_cot'), "
            f"got: {name!r}"
        )
    task, prompt_name = parts

    # Resolve path relative to this file: core/ -> benchmarks/{task}/prompts/
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
    json_path = benchmarks_dir / task / "prompts" / f"{prompt_name}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Prompt examples file not found: {json_path}\n"
            f"Create a JSON file with a list of {{'question': ..., 'answer': ...}} objects."
        )

    with json_path.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    if not isinstance(examples, list):
        raise ValueError(f"Expected a JSON array in {json_path}, got {type(examples).__name__}")

    return examples


class PromptBuilder:
    """Builds prompts with few-shot examples from a dataset's train split."""

    def __init__(self, config: PromptConfig, task_spec: TaskSpec) -> None:
        self.config = config
        self.task_spec = task_spec
        self._few_shot_examples: list[dict[str, Any]] = []

    def load_few_shot_examples(self, seed: int) -> None:
        """Load few-shot examples from the configured split or static set.

        Args:
            seed: Random seed for reproducible example selection.
                  Can be overridden by config.few_shot_seed if set.
        """
        if self.config.few_shot_count <= 0:
            self._few_shot_examples = []
            return

        # Check if using static examples from file
        if self.config.static_examples is not None:
            static = _load_static_examples(self.config.static_examples)
            # Use up to few_shot_count examples from the static set
            self._few_shot_examples = static[: self.config.few_shot_count]
            return

        # Use few_shot_seed if specified, otherwise use the provided seed
        effective_seed = self.config.few_shot_seed if self.config.few_shot_seed is not None else seed

        # Load examples from the few-shot split using the task's dataset builder
        # We request more examples than needed to allow for sampling
        dataset = self.task_spec.dataset_builder(
            n_examples=self.config.few_shot_count * 10,  # Get extra for sampling
            seed=effective_seed,
            split=self.config.few_shot_split,
        )

        # Collect all available examples
        all_examples = list(dataset)

        # Sample the requested number of examples
        if len(all_examples) <= self.config.few_shot_count:
            self._few_shot_examples = all_examples
        else:
            rng = random.Random(effective_seed)
            self._few_shot_examples = rng.sample(all_examples, self.config.few_shot_count)

    def build_prompt(self, question: str) -> str:
        """Build full prompt with few-shot examples followed by test question.

        Args:
            question: The test question to append after few-shot examples.

        Returns:
            Complete prompt string ready for the model.
        """
        parts: list[str] = []

        # Add few-shot examples
        for ex in self._few_shot_examples:
            parts.append(self.config.example_format.format(
                question=ex["question"],
                answer=ex["answer"],
            ))

        # Add the test question
        parts.append(self.config.test_format.format(question=question))

        return "".join(parts)
