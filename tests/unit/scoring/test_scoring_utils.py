"""Unit tests for scoring_utils.py functions."""

import pytest

from phi_synth_math.tasks.core.scoring_utils import (
    normalize_answer,
    exact_match,
    extract_last_number,
    extract_answer_is_number,
)


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    @pytest.mark.unit
    def test_normalize_removes_spaces_and_commas(self):
        """Test that spaces, commas are removed and text lowercased."""
        assert normalize_answer("  Hello, World  ") == "helloworld"
        assert normalize_answer("1,000") == "1000"
        assert normalize_answer("  42  ") == "42"
        assert normalize_answer("") == ""
        assert normalize_answer("UPPERCASE") == "uppercase"


class TestExactMatch:
    """Tests for exact_match function."""

    @pytest.mark.unit
    def test_exact_match_ignores_whitespace_and_case(self):
        """Test exact_match with normalization."""
        assert exact_match("42", "  42  ") is True
        assert exact_match("Hello", "HELLO") is True
        assert exact_match("1,000", "1000") is True
        assert exact_match("42", "43") is False
        assert exact_match("", "") is True
        assert exact_match("abc", "") is False


class TestExtractLastNumber:
    """Tests for extract_last_number function."""

    @pytest.mark.unit
    def test_extracts_last_number_from_text(self):
        """Test extracting the last number from various text formats."""
        assert extract_last_number("The result is 42") == "42"
        assert extract_last_number("First 10, then 20, finally 30") == "30"
        assert extract_last_number("Price: $1,234.56") == "1234.56"
        assert extract_last_number("No numbers here") is None

    @pytest.mark.unit
    def test_handles_negative_and_decimal_numbers(self):
        """Test extracting negative and decimal numbers."""
        assert extract_last_number("The answer is -5") == "-5"
        assert extract_last_number("Temperature: -10.5 degrees") == "-10.5"
        assert extract_last_number("Pi is approximately 3.14159") == "3.14159"
        # Comma-separated thousands
        assert extract_last_number("Population: 1,000,000") == "1000000"


class TestExtractAnswerIsNumber:
    """Tests for extract_answer_is_number function (CoT format)."""

    @pytest.mark.unit
    def test_extracts_from_answer_is_pattern(self):
        """Test extracting number from 'The answer is X' pattern."""
        assert extract_answer_is_number("So the answer is 42.") == "42"
        assert extract_answer_is_number("Therefore, The answer is 100") == "100"
        assert extract_answer_is_number("the answer is -5") == "-5"
        assert extract_answer_is_number("The answer is 1,234") == "1234"

    @pytest.mark.unit
    def test_returns_none_when_no_pattern(self):
        """Test returns None when pattern not found."""
        assert extract_answer_is_number("The result is 42") is None
        assert extract_answer_is_number("42 is the answer") is None
        assert extract_answer_is_number("No pattern here") is None
