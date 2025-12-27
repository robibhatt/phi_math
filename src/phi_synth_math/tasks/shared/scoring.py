from __future__ import annotations

import re
from collections.abc import Callable

# Match integers or decimals with optional sign, allowing comma separators.
# Crucially: if there's a '.', require at least one digit after it.
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def normalize_answer(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch not in {" ", ","})


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def _score_gsm8k(pred: str, gold: str) -> bool:
    # Local import avoids circular dependency when gsm8k.scoring imports helpers here.
    from phi_synth_math.tasks import gsm8k

    return gsm8k.score(pred, gold)


def _score_dummy_addition(pred: str, gold: str) -> bool:
    from phi_synth_math.tasks.dummy_addition import scoring as dummy_addition_scoring

    return dummy_addition_scoring.score(pred, gold)


SCORING_REGISTRY: dict[str, Callable[[str, str], bool]] = {
    "gsm8k": _score_gsm8k,
    "dummy_math_addition": _score_dummy_addition,
}


def score_prediction(dataset_name: str, pred: str, gold: str) -> bool:
    scorer = SCORING_REGISTRY.get(dataset_name, exact_match)
    return scorer(pred, gold)
