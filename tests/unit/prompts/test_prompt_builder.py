"""Unit tests for PromptBuilder."""

import pytest

from phi_synth_math.core.config import PromptConfig
from phi_synth_math.tasks.core.metadata import get_task_spec
from phi_synth_math.tasks.core.prompt_builder import PromptBuilder, _load_static_examples


class TestLoadStaticExamples:
    """Tests for _load_static_examples function."""

    @pytest.mark.unit
    def test_load_gsm8k_8shot_cot(self):
        """Test loading the canonical GSM8K 8-shot CoT examples."""
        examples = _load_static_examples("gsm8k/8shot_cot")

        assert isinstance(examples, list)
        assert len(examples) == 8
        for ex in examples:
            assert "question" in ex
            assert "answer" in ex

    @pytest.mark.unit
    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="must be in 'task/name' format"):
            _load_static_examples("invalid_format")

        with pytest.raises(ValueError, match="must be in 'task/name' format"):
            _load_static_examples("a/b/c")


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.mark.unit
    def test_build_prompt_with_static_examples(self):
        """Test building prompt with static few-shot examples."""
        config = PromptConfig(
            few_shot_count=2,
            static_examples="gsm8k/8shot_cot",
            example_format="Q: {question}\nA: {answer}\n\n",
            test_format="Q: {question}\nA:",
        )
        task_spec = get_task_spec("gsm8k")
        builder = PromptBuilder(config, task_spec)
        builder.load_few_shot_examples(seed=42)

        prompt = builder.build_prompt("What is 2+2?")

        # Should contain 2 examples plus the test question
        assert "Q:" in prompt
        assert "A:" in prompt
        assert "What is 2+2?" in prompt
        # Verify it ends with the test format (no trailing answer)
        assert prompt.rstrip().endswith("A:")
        # Should have exactly 3 Q: occurrences (2 examples + 1 test)
        assert prompt.count("Q:") == 3

    @pytest.mark.unit
    def test_build_prompt_zero_shot(self):
        """Test building prompt with no few-shot examples."""
        config = PromptConfig(
            few_shot_count=0,
            example_format="Q: {question}\nA: {answer}\n\n",
            test_format="Q: {question}\nA:",
        )
        task_spec = get_task_spec("dummy_math_addition")
        builder = PromptBuilder(config, task_spec)
        builder.load_few_shot_examples(seed=42)

        prompt = builder.build_prompt("What is 5+3?")

        assert prompt == "Q: What is 5+3?\nA:"
