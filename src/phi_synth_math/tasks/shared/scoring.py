from __future__ import annotations

import re


def normalize_answer(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch not in {" ", ","})


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


# Match integers or decimals with optional sign, allowing comma separators.
# Crucially: if there's a '.', require at least one digit after it.
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def score_prediction(dataset_name: str, pred: str, gold: str) -> bool:
    # Local imports avoid circular dependencies with per-task scoring modules.
    if dataset_name == "gsm8k":
        from phi_synth_math.tasks import gsm8k

        return gsm8k.score(pred, gold)

    if dataset_name == "dummy_math_addition":
        from phi_synth_math.tasks.dummy_addition import scoring as dummy_addition_scoring

        return dummy_addition_scoring.score(pred, gold)

    # Default to an exact match for unknown datasets.
    return exact_match(pred, gold)
