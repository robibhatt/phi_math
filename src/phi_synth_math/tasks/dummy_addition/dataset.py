from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from phi_synth_math.tasks.datasets.base import Dataset


@dataclass
class DummyMathAdditionDataset(Dataset):
    n_examples: int
    seed: int
    max_int: int

    def __iter__(self) -> Iterable[dict[str, Any]]:
        rng = random.Random(self.seed)
        for idx in range(1, self.n_examples + 1):
            a = rng.randint(0, self.max_int)
            b = rng.randint(0, self.max_int)
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            yield {
                "id": f"ex_{idx:06d}",
                "question": question,
                "answer": answer,
            }
