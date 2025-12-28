from phi_synth_math.tasks.core.dataset import Dataset
from phi_synth_math.tasks.core.scoring import score_prediction
from phi_synth_math.tasks.core.scoring_utils import exact_match, extract_last_number, normalize_answer
from phi_synth_math.tasks.core.metadata import TaskSpec, get_task_spec

# Note: EvalRunner removed to avoid circular import.
# Import directly: from phi_synth_math.tasks.core.runner import EvalRunner

__all__ = [
    "Dataset",
    "TaskSpec",
    "get_task_spec",
    "exact_match",
    "extract_last_number",
    "normalize_answer",
    "score_prediction",
]
