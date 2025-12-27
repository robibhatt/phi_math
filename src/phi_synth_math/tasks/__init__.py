"""Task definitions and evaluation utilities."""

from phi_synth_math.tasks.core import (
    Dataset,
    EvalRunner,
    score_prediction,
)
from phi_synth_math.tasks.core.scoring_utils import exact_match, extract_last_number, normalize_answer
from . import benchmarks, core

__all__ = [
    "Dataset",
    "EvalRunner",
    "exact_match",
    "extract_last_number",
    "normalize_answer",
    "score_prediction",
    "core",
    "benchmarks",
]
