from __future__ import annotations

from phi_synth_math.tasks.core.scoring_utils import (
    exact_match,
    extract_answer_is_number,
    extract_last_number,
    normalize_answer,
)


def score(pred: str, gold: str) -> bool:
    """Score GSM8K prediction against gold answer.

    First tries to extract numbers from 'The answer is X' pattern (CoT format),
    then falls back to extracting the last number in the text.
    """
    # Try "The answer is X" pattern first (chain-of-thought format)
    pred_number = extract_answer_is_number(pred)
    gold_number = extract_answer_is_number(gold)

    # Fall back to last number extraction
    if pred_number is None:
        pred_number = extract_last_number(pred)
    if gold_number is None:
        gold_number = extract_last_number(gold)

    if pred_number is not None and gold_number is not None:
        return normalize_answer(pred_number) == normalize_answer(gold_number)
    return exact_match(pred, gold)
