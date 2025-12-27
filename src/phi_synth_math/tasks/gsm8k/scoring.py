from __future__ import annotations

from phi_synth_math.tasks.shared.scoring import exact_match, extract_last_number, normalize_answer


def score(pred: str, gold: str) -> bool:
    pred_number = extract_last_number(pred)
    gold_number = extract_last_number(gold)
    if pred_number is not None and gold_number is not None:
        return normalize_answer(pred_number) == normalize_answer(gold_number)
    return exact_match(pred, gold)
