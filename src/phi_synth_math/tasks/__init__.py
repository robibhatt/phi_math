"""Task definitions and evaluation utilities."""

from phi_synth_math.tasks.core import (
    Dataset,
    score_prediction,
)
from phi_synth_math.tasks.core.scoring_utils import exact_match, extract_last_number, normalize_answer
from . import benchmarks, core

# Note: EvalRunner removed to avoid circular import.
# Import directly: from phi_synth_math.tasks.core.runner import EvalRunner

__all__ = [
    "Dataset",
    "exact_match",
    "extract_last_number",
    "normalize_answer",
    "score_prediction",
    "core",
    "benchmarks",
]
