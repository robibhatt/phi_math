"""Unit tests for GSM8K scoring logic."""

import pytest

from phi_synth_math.tasks.benchmarks.gsm8k.scoring import score


class TestGSM8KScore:
    """Tests for GSM8K score function."""

    @pytest.mark.unit
    def test_score_with_answer_is_pattern(self):
        """Test scoring when both use 'The answer is X' pattern."""
        # Standard CoT format matching
        pred = "Let me solve this. 5 + 3 = 8. The answer is 8."
        gold = "Working through: The answer is 8."
        assert score(pred, gold) is True

        # Mismatch
        pred_wrong = "The answer is 7."
        assert score(pred_wrong, gold) is False

        # Both with commas in numbers
        pred_comma = "The answer is 1,234."
        gold_comma = "The answer is 1234."
        assert score(pred_comma, gold_comma) is True

    @pytest.mark.unit
    def test_score_with_last_number_fallback(self):
        """Test scoring falls back to last number extraction."""
        # Gold has 'The answer is X', pred uses raw number
        gold = "The total is 42. The answer is 42. #####"
        pred = "The calculation gives us 42"
        assert score(pred, gold) is True

        # Both use last number (neither has pattern)
        gold_no_pattern = "The result: 100 #####"
        pred_no_pattern = "Working... 100"
        assert score(pred_no_pattern, gold_no_pattern) is True

        # Mismatch in last numbers
        pred_wrong = "I got 99"
        gold_correct = "The answer is 100"
        assert score(pred_wrong, gold_correct) is False

        # Falls back to exact_match when no numbers found
        pred_text = "forty-two"
        gold_text = "forty-two"
        assert score(pred_text, gold_text) is True

        pred_text_wrong = "forty-two"
        gold_text_diff = "forty-three"
        assert score(pred_text_wrong, gold_text_diff) is False
