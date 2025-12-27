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
    if dataset_name == "gsm8k":
        pred_number = extract_last_number(pred)
        gold_number = extract_last_number(gold)
        if pred_number is not None and gold_number is not None:
            return normalize_answer(pred_number) == normalize_answer(gold_number)
    return exact_match(pred, gold)
