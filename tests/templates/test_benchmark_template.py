"""
TEMPLATE: Copy this file when adding a new benchmark.

Instructions:
1. Copy this file to tests/unit/scoring/test_{benchmark_name}_scoring.py
2. Replace all TODO placeholders with your benchmark-specific code
3. Update the test cases with appropriate test data for your scoring logic
4. Remove this docstring header

Your benchmark should have:
- src/phi_synth_math/tasks/benchmarks/{benchmark_name}/scoring.py with score() function
- src/phi_synth_math/tasks/benchmarks/{benchmark_name}/dataset.py with Dataset class
"""

import pytest

# TODO: Update import path to your benchmark
# from phi_synth_math.tasks.benchmarks.{benchmark_name}.scoring import score


class TestBenchmarkScore:
    """Tests for {benchmark_name} score function.

    TODO: Rename this class to Test{BenchmarkName}Score
    """

    @pytest.mark.unit
    def test_score_correct_prediction(self):
        """Test scoring when prediction matches gold answer.

        TODO: Replace with actual test data for your benchmark.
        """
        # TODO: Uncomment and update
        # pred = "42"  # Realistic prediction
        # gold = "42"  # Realistic gold answer
        # assert score(pred, gold) is True
        pass

    @pytest.mark.unit
    def test_score_incorrect_prediction(self):
        """Test scoring when prediction does not match gold answer.

        TODO: Replace with actual test data for your benchmark.
        """
        # TODO: Uncomment and update
        # pred = "41"  # Wrong prediction
        # gold = "42"  # Gold answer
        # assert score(pred, gold) is False
        pass

    @pytest.mark.unit
    def test_score_handles_edge_cases(self):
        """Test scoring edge cases specific to this benchmark.

        TODO: Add edge cases relevant to your benchmark's scoring logic.
        Examples:
        - Whitespace handling
        - Number format variations (commas, decimals)
        - Special characters
        - Empty strings
        - Case sensitivity
        """
        # TODO: Add edge case tests
        pass


class TestBenchmarkDataset:
    """Tests for {benchmark_name} dataset.

    TODO: Rename this class to Test{BenchmarkName}Dataset

    Only needed if your dataset has complex logic (e.g., external data loading).
    For simple generated datasets, these tests are optional.
    """

    @pytest.mark.unit
    def test_dataset_yields_valid_examples(self):
        """Test that dataset yields examples with required fields.

        TODO: Update import and instantiation for your dataset.
        """
        # TODO: Uncomment and update
        # from phi_synth_math.tasks.benchmarks.{benchmark_name}.dataset import {DatasetClass}
        #
        # dataset = {DatasetClass}(n_examples=3, seed=42)
        # examples = list(dataset)
        #
        # assert len(examples) == 3
        # for ex in examples:
        #     assert "id" in ex
        #     assert "question" in ex
        #     assert "answer" in ex
        pass

    @pytest.mark.unit
    def test_dataset_is_deterministic(self):
        """Test that same seed produces same examples.

        TODO: Update for your dataset.
        """
        # TODO: Uncomment and update
        # from phi_synth_math.tasks.benchmarks.{benchmark_name}.dataset import {DatasetClass}
        #
        # dataset1 = list({DatasetClass}(n_examples=3, seed=42))
        # dataset2 = list({DatasetClass}(n_examples=3, seed=42))
        #
        # assert dataset1 == dataset2
        pass
