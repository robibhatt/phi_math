from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from datasets import load_dataset

from phi_synth_math.tasks.core.dataset import Dataset

# Split configuration: carve validation from HuggingFace's train split
# HF train has 7473 examples, HF test has 1319 examples
TRAIN_VAL_RATIO = 0.8  # First 80% of HF train -> our train, last 20% -> our val

VALID_SPLITS = ("train", "val", "test")


@dataclass
class GSM8KDataset(Dataset):
    n_examples: int
    seed: int
    split: str

    def __iter__(self) -> Iterable[dict[str, Any]]:
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Unknown split '{self.split}'. Must be one of: {list(VALID_SPLITS)}"
            )

        # Determine which HuggingFace split to load and the index range
        if self.split == "test":
            hf_split = "test"
            dataset = load_dataset("gsm8k", "main", split=hf_split)
            start_idx = 0
            end_idx = len(dataset)
        else:
            # Both train and val come from HuggingFace's train split
            hf_split = "train"
            dataset = load_dataset("gsm8k", "main", split=hf_split)
            total = len(dataset)
            split_point = int(total * TRAIN_VAL_RATIO)

            if self.split == "train":
                start_idx = 0
                end_idx = split_point
            else:  # val
                start_idx = split_point
                end_idx = total

        # Build indices for this split range
        available_indices = list(range(start_idx, end_idx))
        take = min(self.n_examples, len(available_indices))

        if take < len(available_indices):
            rng = random.Random(self.seed)
            # Deterministic subset; keep ascending order for reproducible iteration
            indices = sorted(rng.sample(available_indices, take))
        else:
            indices = available_indices[:take]

        for idx, dataset_idx in enumerate(indices, start=1):
            example = dataset[dataset_idx]
            question_text = str(example.get("question", ""))
            answer_text = str(example.get("answer", "")).strip() + " #####"

            yield {
                "id": f"gsm8k_{self.split}_{idx:06d}",
                "question": question_text,
                "answer": answer_text,
            }
