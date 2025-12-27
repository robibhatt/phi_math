"""Task definitions and evaluation utilities."""

from phi_synth_math.tasks.core import (
    Dataset,
    EvalRunner,
    exact_match,
    extract_last_number,
    normalize_answer,
    score_prediction,
)
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
