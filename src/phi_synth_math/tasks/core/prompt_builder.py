"""Prompt builder for few-shot prompting."""

from __future__ import annotations

import random
from typing import Any

from phi_synth_math.core.config import PromptConfig
from phi_synth_math.tasks.core.metadata import TaskSpec

# Registry of static few-shot example sets
_STATIC_EXAMPLES_REGISTRY: dict[str, list[dict[str, str]]] = {}


def _get_static_examples(name: str) -> list[dict[str, str]]:
    """Get static examples by name, loading them lazily."""
    if name not in _STATIC_EXAMPLES_REGISTRY:
        if name == "gsm8k_8shot":
            from phi_synth_math.tasks.benchmarks.gsm8k.few_shot_examples import (
                GSM8K_8SHOT_EXAMPLES,
            )
            _STATIC_EXAMPLES_REGISTRY[name] = GSM8K_8SHOT_EXAMPLES
        else:
            raise ValueError(f"Unknown static examples set: {name!r}")
    return _STATIC_EXAMPLES_REGISTRY[name]


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

        # Check if using static (hardcoded) examples
        if self.config.static_examples is not None:
            static = _get_static_examples(self.config.static_examples)
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
