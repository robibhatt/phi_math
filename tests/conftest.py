"""Shared fixtures for phi_synth_math tests.

Design principle: Minimal fixtures, no parametrized fixtures.
Each test file manages its own test data for clarity and template-friendliness.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict() -> dict:
    """Minimal valid config dictionary for testing.

    Tests can modify this dict as needed rather than using parametrized fixtures.
    """
    return {
        "task_name": "dummy_math_addition",
        "results_root": "/tmp/test_results",
        "seed": 42,
        "n_examples": 5,
        "batch_size": 2,
        "model": {
            "name": "dummy",
        },
        "dataset": {
            "name": "dummy_math_addition",
            "max_int": 10,
        },
    }


@pytest.fixture
def sample_config_path(tmp_dir: Path, sample_config_dict: dict) -> Path:
    """Write sample config to a temp file and return path."""
    import yaml

    config_path = tmp_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(sample_config_dict, f)
    return config_path
