from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from phi_synth_math.tasks.benchmarks.dummy_addition.dataset import DummyMathAdditionDataset
from phi_synth_math.tasks.benchmarks.dummy_addition import scoring as dummy_addition_scoring
from phi_synth_math.tasks.benchmarks.gsm8k.dataset import GSM8KDataset
from phi_synth_math.tasks.benchmarks.gsm8k import scoring as gsm8k_scoring
from phi_synth_math.tasks.core.dataset import Dataset


@dataclass(frozen=True)
class TaskSpec:
    """Describes a benchmark's dataset, scoring, defaults, and prompt template.

    Contract:
    - dataset_builder: callable that accepts (n_examples: int, seed: int, **dataset_params)
      and returns an iterable of examples with ``id``, ``question``, and ``answer`` fields.
      The dataset is responsible only for providing the raw question text (no instruction
      prompt applied).
    - scorer: callable (pred: str, gold: str) -> bool that is used by the evaluation loop.
    - default_dataset_params: mapping of dataset kwargs applied unless overridden by config.
    - prompt_template: format string applied to the dataset's question via
      ``prompt_template.format(question=question_text)`` before calling the model.
    """

    dataset_builder: Callable[..., Dataset]
    scorer: Callable[[str, str], bool]
    default_dataset_params: Mapping[str, Any] = field(default_factory=dict)
    prompt_template: str = "{question}"


TASK_SPECS: dict[str, TaskSpec] = {
    "dummy_math_addition": TaskSpec(
        dataset_builder=DummyMathAdditionDataset,
        scorer=dummy_addition_scoring.score,
        default_dataset_params={"max_int": 20},
        prompt_template="{question}",
    ),
    "gsm8k": TaskSpec(
        dataset_builder=GSM8KDataset,
        scorer=gsm8k_scoring.score,
        default_dataset_params={"split": "test"},
        prompt_template=(
            "Solve the problem. Give ONLY the final numeric answer.\n\n"
            "Problem: {question}\n"
            "Answer: "
        ),
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    if name not in TASK_SPECS:
        available = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return TASK_SPECS[name]
