"""Supervised fine-tuning dataset wrapping existing task datasets."""

from __future__ import annotations

from typing import Any, Protocol

from phi_synth_math.tasks.core.metadata import get_task_spec


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""

    def __call__(
        self,
        text: str,
        max_length: int | None = None,
        truncation: bool = False,
        padding: str | bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Any]: ...


class SFTDataset:
    """Supervised fine-tuning dataset wrapping existing task datasets.

    Loads examples from TASK_SPECS, formats them as Q/A text, and provides
    tokenized outputs for training.
    """

    def __init__(
        self,
        task_name: str,
        split: str,
        tokenizer: Tokenizer,
        max_seq_length: int,
        seed: int,
        n_examples: int | None = None,
    ) -> None:
        """Initialize SFT dataset.

        Args:
            task_name: Name of the task (must exist in TASK_SPECS).
            split: Dataset split ("train", "val", or "test").
            tokenizer: Tokenizer for encoding text.
            max_seq_length: Maximum sequence length for tokenization.
            seed: Random seed for reproducibility.
            n_examples: Number of examples to load. If None, loads all available.
        """
        self._task_name = task_name
        self._split = split
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._seed = seed

        # Get task spec (raises ValueError if unknown task)
        task_spec = get_task_spec(task_name)

        # Build dataset kwargs from defaults
        dataset_kwargs = dict(task_spec.default_dataset_params)
        dataset_kwargs["split"] = split

        # Load all examples into memory for random access
        # Use a large n_examples if not specified to get all available
        load_n = n_examples if n_examples is not None else 100000
        dataset = task_spec.dataset_builder(
            n_examples=load_n, seed=seed, **dataset_kwargs
        )

        self._examples: list[dict[str, Any]] = list(dataset)

        # Trim to requested n_examples if specified
        if n_examples is not None and len(self._examples) > n_examples:
            self._examples = self._examples[:n_examples]

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self._examples)

    def get_formatted_text(self, idx: int) -> str:
        """Get formatted Q/A text for an example.

        Args:
            idx: Index of the example.

        Returns:
            Formatted text combining question and answer.
        """
        example = self._examples[idx]
        question = example["question"]
        answer = example["answer"]

        # Format as Q/A pair for supervised fine-tuning
        return f"Question: {question}\nAnswer: {answer}"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get tokenized example.

        Args:
            idx: Index of the example.

        Returns:
            Dictionary with 'input_ids' and 'labels' keys.
        """
        text = self.get_formatted_text(idx)

        # Tokenize the text
        encoded = self._tokenizer(
            text,
            max_length=self._max_seq_length,
            truncation=True,
        )

        input_ids = encoded["input_ids"]

        # For causal LM, labels are the same as input_ids
        # (the model learns to predict next token)
        labels = input_ids.copy() if isinstance(input_ids, list) else input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
