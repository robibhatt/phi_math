from __future__ import annotations

from phi_synth_math.tasks.core.scoring_utils import exact_match


def score(pred: str, gold: str) -> bool:
    return exact_match(pred, gold)
