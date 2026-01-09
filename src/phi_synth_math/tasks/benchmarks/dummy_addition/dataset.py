from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from phi_synth_math.tasks.core.dataset import Dataset

# Fixed seeds for reproducible splits
SPLIT_SEEDS: dict[str, int] = {
    "train": 1000,
    "val": 2000,
    "test": 3000,
}

# Maximum examples per split
SPLIT_MAX_EXAMPLES: int = 1000


@dataclass
class DummyMathAdditionDataset(Dataset):
    n_examples: int
    seed: int  # Kept for API compatibility, but split seed is used
    max_int: int
    split: str = "test"

    def __iter__(self) -> Iterable[dict[str, Any]]:
        if self.split not in SPLIT_SEEDS:
            raise ValueError(
                f"Unknown split '{self.split}'. Must be one of: {list(SPLIT_SEEDS.keys())}"
            )

        # Use fixed seed for the split to ensure deterministic examples
        split_seed = SPLIT_SEEDS[self.split]
        rng = random.Random(split_seed)

        # Cap at SPLIT_MAX_EXAMPLES for consistent split sizes
        n_to_generate = min(self.n_examples, SPLIT_MAX_EXAMPLES)

        for idx in range(1, n_to_generate + 1):
            a = rng.randint(0, self.max_int)
            b = rng.randint(0, self.max_int)
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            yield {
                "id": f"dummy_addition_{self.split}_{idx:06d}",
                "question": question,
                "answer": answer,
            }
