from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from datasets import load_dataset

from .base import Dataset


def _extract_final_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


@dataclass
class GSM8KDataset(Dataset):
    n_examples: int
    seed: int
    split: str = "test"

    def __iter__(self) -> Iterable[dict[str, Any]]:
        dataset = load_dataset("gsm8k", "main", split=self.split)
        total = len(dataset)
        take = min(self.n_examples, total)

        indices = list(range(total))
        if take < total:
            rng = random.Random(self.seed)
            indices = sorted(rng.sample(indices, take))
        else:
            indices = indices[:take]

        for idx, dataset_idx in enumerate(indices, start=1):
            example = dataset[dataset_idx]
            question_text = example.get("question", "")
            answer_text = _extract_final_answer(str(example.get("answer", "")))
            prompt = (
                "Solve the problem. Give ONLY the final numeric answer.\n\n"
                f"Problem: {question_text}\nAnswer:"
            )
            yield {
                "id": f"gsm8k_{idx:06d}",
                "question": prompt,
                "answer": answer_text,
            }
