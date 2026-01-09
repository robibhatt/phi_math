"""Unit tests for SFT dataset."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from phi_synth_math.training.sft_dataset import SFTDataset


class MockTokenizer:
    """Mock tokenizer for testing without real model dependencies."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(
        self,
        text: str,
        max_length: int | None = None,
        truncation: bool = False,
        padding: str | bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        # Simple mock: convert each character to a token id
        input_ids = [ord(c) % self.vocab_size for c in text]
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """Provide a mock tokenizer for testing."""
    return MockTokenizer()


class TestSFTDataset:
    """Tests for SFTDataset."""

    @pytest.mark.unit
    def test_sft_dataset_from_task_name(self, mock_tokenizer: MockTokenizer):
        """Create SFT dataset from task_name using existing TASK_SPECS."""
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
        )

        assert dataset is not None
        assert len(dataset) > 0

    @pytest.mark.unit
    def test_sft_dataset_formats_qa_pairs(self, mock_tokenizer: MockTokenizer):
        """Dataset formats examples as Q/A text."""
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
        )

        # Get a raw example to check formatting
        raw_text = dataset.get_formatted_text(0)

        # Should contain question and answer in some format
        assert "What is" in raw_text  # Question from dummy_math_addition
        assert "+" in raw_text  # Part of the addition question

    @pytest.mark.unit
    def test_sft_dataset_length(self, mock_tokenizer: MockTokenizer):
        """__len__ returns correct count."""
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
            n_examples=10,
        )

        assert len(dataset) == 10

    @pytest.mark.unit
    def test_sft_dataset_length_default(self, mock_tokenizer: MockTokenizer):
        """__len__ returns all available examples when n_examples not specified."""
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
        )

        # Should have some examples (dummy_math_addition has up to 1000 per split)
        assert len(dataset) > 0

    @pytest.mark.unit
    def test_sft_dataset_getitem_returns_tokenized(self, mock_tokenizer: MockTokenizer):
        """__getitem__ returns tokenized dict with input_ids, labels."""
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], list)
        assert isinstance(item["labels"], list)
        assert len(item["input_ids"]) == len(item["labels"])

    @pytest.mark.unit
    def test_sft_dataset_getitem_different_indices(self, mock_tokenizer: MockTokenizer):
        """Different indices return different examples."""
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
            n_examples=5,
        )

        item0 = dataset[0]
        item1 = dataset[1]

        # Different examples should have different input_ids
        assert item0["input_ids"] != item1["input_ids"]

    @pytest.mark.unit
    def test_sft_dataset_unknown_task_raises(self, mock_tokenizer: MockTokenizer):
        """Unknown task_name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            SFTDataset(
                task_name="nonexistent_task",
                split="train",
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                seed=42,
            )

    @pytest.mark.unit
    def test_sft_dataset_respects_max_seq_length(self, mock_tokenizer: MockTokenizer):
        """Dataset respects max_seq_length for tokenization."""
        max_len = 32
        dataset = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=max_len,
            seed=42,
        )

        item = dataset[0]

        assert len(item["input_ids"]) <= max_len
        assert len(item["labels"]) <= max_len

    @pytest.mark.unit
    def test_sft_dataset_deterministic(self, mock_tokenizer: MockTokenizer):
        """Same seed produces same dataset."""
        dataset1 = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
            n_examples=5,
        )
        dataset2 = SFTDataset(
            task_name="dummy_math_addition",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
            n_examples=5,
        )

        for i in range(len(dataset1)):
            assert dataset1[i]["input_ids"] == dataset2[i]["input_ids"]

    @pytest.mark.unit
    def test_sft_dataset_with_gsm8k(self, mock_tokenizer: MockTokenizer):
        """SFT dataset works with gsm8k task."""
        dataset = SFTDataset(
            task_name="gsm8k",
            split="train",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=42,
            n_examples=5,
        )

        assert len(dataset) == 5
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
