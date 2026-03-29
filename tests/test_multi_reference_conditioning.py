import unittest
from concurrent.futures import Future
from unittest import mock

import torch

from indextts.reference_conditioning import (
    get_or_create_cached,
    merge_variable_length_tensors,
    merge_weighted_vectors,
    normalize_reference_inputs,
    resolve_unique_results,
)


class MultiReferenceConditioningTests(unittest.TestCase):
    def test_normalize_reference_inputs_defaults_to_equal_weights(self):
        refs = normalize_reference_inputs(
            ["speaker_a.wav", "speaker_b.wav", "speaker_c.wav"],
            None,
            label="speaker reference",
        )
        self.assertEqual([ref.path.endswith(name) for ref, name in zip(refs, ["speaker_a.wav", "speaker_b.wav", "speaker_c.wav"])], [True, True, True])
        self.assertEqual([ref.weight for ref in refs], [1 / 3, 1 / 3, 1 / 3])

    def test_normalize_reference_inputs_normalizes_custom_weights(self):
        refs = normalize_reference_inputs(
            ["speaker_a.wav", "speaker_b.wav"],
            [2, 6],
            label="speaker reference",
        )
        self.assertEqual([ref.weight for ref in refs], [0.25, 0.75])

    def test_normalize_reference_inputs_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            normalize_reference_inputs(["speaker_a.wav", "speaker_b.wav"], [1], label="speaker reference")

        with self.assertRaises(ValueError):
            normalize_reference_inputs(["speaker_a.wav", "speaker_b.wav"], [1, -1], label="speaker reference")

        with self.assertRaises(ValueError):
            normalize_reference_inputs(["speaker_a.wav", "speaker_b.wav"], [0, 0], label="speaker reference")

    def test_merge_variable_length_tensors_renormalizes_tails(self):
        first = torch.tensor([[[1.0], [3.0], [5.0]]])
        second = torch.tensor([[[10.0]]])
        merged = merge_variable_length_tensors([first, second], [0.2, 0.8], seq_dim=1)
        expected = torch.tensor([[[8.2], [3.0], [5.0]]])
        self.assertTrue(torch.allclose(merged, expected))

    def test_merge_variable_length_tensors_supports_middle_sequence_axis(self):
        first = torch.ones(1, 2, 3)
        second = torch.full((1, 4, 3), 3.0)
        merged = merge_variable_length_tensors([first, second], [0.25, 0.75], seq_dim=1)
        expected = torch.tensor(
            [[[2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]]
        )
        self.assertTrue(torch.allclose(merged, expected))

    def test_merge_variable_length_tensors_rejects_wrong_sequence_axis(self):
        first = torch.ones(1, 2, 3)
        second = torch.ones(1, 4, 3)
        with self.assertRaises(ValueError):
            merge_variable_length_tensors([first, second], [0.5, 0.5], seq_dim=2)

    def test_merge_weighted_vectors(self):
        merged = merge_weighted_vectors(
            [torch.tensor([[2.0, 4.0]]), torch.tensor([[20.0, 40.0]])],
            [0.25, 0.75],
        )
        self.assertTrue(torch.allclose(merged, torch.tensor([[15.5, 31.0]])))

    def test_get_or_create_cached_reuses_factory_result(self):
        cache = {}
        calls = {"count": 0}

        def factory():
            calls["count"] += 1
            return {"value": 42}

        first = get_or_create_cached(cache, "emotion.wav", factory)
        second = get_or_create_cached(cache, "emotion.wav", factory)

        self.assertIs(first, second)
        self.assertEqual(calls["count"], 1)

    def test_resolve_unique_results_deduplicates_keys(self):
        calls = []

        def factory(key):
            calls.append(key)
            return key.upper()

        results = resolve_unique_results(["speaker_a", "speaker_b", "speaker_a"], factory, max_workers=4)

        self.assertEqual(results, {"speaker_a": "SPEAKER_A", "speaker_b": "SPEAKER_B"})
        self.assertEqual(calls, ["speaker_a", "speaker_b"])

    def test_resolve_unique_results_uses_executor_when_parallelism_is_enabled(self):
        submitted = []

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def submit(self, fn, key):
                submitted.append((self.max_workers, key))
                future = Future()
                future.set_result(fn(key))
                return future

        with mock.patch("indextts.reference_conditioning.ThreadPoolExecutor", FakeExecutor):
            results = resolve_unique_results(["a", "b"], lambda key: key * 2, max_workers=8)

        self.assertEqual(results, {"a": "aa", "b": "bb"})
        self.assertEqual(submitted, [(2, "a"), (2, "b")])


if __name__ == "__main__":
    unittest.main()
