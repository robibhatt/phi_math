from __future__ import annotations

from collections.abc import Callable

from phi_synth_math.tasks.benchmarks.dummy_addition.scoring import score as score_dummy_addition
from phi_synth_math.tasks.benchmarks.gsm8k.scoring import score as score_gsm8k

from .scoring_utils import exact_match


SCORING_REGISTRY: dict[str, Callable[[str, str], bool]] = {
    "gsm8k": score_gsm8k,
    "dummy_math_addition": score_dummy_addition,
}


def score_prediction(dataset_name: str, pred: str, gold: str) -> bool:
    scorer = SCORING_REGISTRY.get(dataset_name, exact_match)
    return scorer(pred, gold)
