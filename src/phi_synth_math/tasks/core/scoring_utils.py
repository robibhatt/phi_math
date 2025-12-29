from __future__ import annotations

import re

# Match integers or decimals with optional sign, allowing comma separators.
# Crucially: if there's a '.', require at least one digit after it.
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

# Match "The answer is X." pattern (case-insensitive).
# The period after the number is optional but not captured.
# Matches: The answer is 6., The answer is -3.14, The answer is 1,000
_ANSWER_IS_RE = re.compile(r"[Tt]he answer is\s*(\-?[\d,]+(?:\.\d+)?)")


def normalize_answer(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch not in {" ", ","})


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def extract_answer_is_number(text: str) -> str | None:
    """Extract number after 'The answer is' pattern.

    Used for chain-of-thought evaluation where the model is expected
    to end its response with 'The answer is X.'
    """
    match = _ANSWER_IS_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None
