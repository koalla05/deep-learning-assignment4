"""Check domain labels, dependency grouping, and uncertainty calculations."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.domain_classifier import mask_cues
from generation_pilot.run_adversarial import (
    _bootstrap,
    _check_partition,
    _dependency_groups,
)


class AdversarialTests(unittest.TestCase):
    def test_masking_preserves_location_text_and_matches_old_notebook(self):
        text = "🚌 У Львові\n\nДеталі https://example.org/a та www.example.org 🧤"
        masked = mask_cues(text)
        self.assertIn("У Львові", masked)
        for removed in ["\n", "https://", "www.", "🚌", "🧤"]:
            self.assertNotIn(removed, masked)

    def test_dependency_groups_are_transitive(self):
        rows = [
            {"template_family": "fit_a", "parent_group_ids": ["p1"]},
            {"template_family": "fit_b", "parent_group_ids": ["p1", "p2"]},
            {"template_family": "fit_c", "parent_group_ids": ["p2"]},
            {"template_family": "fit_d", "parent_group_ids": []},
            {"template_family": "fit_d", "parent_group_ids": []},
        ]
        groups = _dependency_groups(rows)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertEqual(groups[3], groups[4])
        self.assertNotEqual(groups[0], groups[3])

    def test_split_rejects_shared_groups_even_with_distinct_text(self):
        rows = pd.DataFrame(
            [
                {"domain_pool": "fit", "group_id": "shared", "text": "one", "label": 0},
                {"domain_pool": "fit", "group_id": "f", "text": "two", "label": 1},
                {"domain_pool": "evaluation", "group_id": "shared", "text": "three", "label": 0},
                {"domain_pool": "evaluation", "group_id": "e", "text": "four", "label": 1},
            ]
        )
        with self.assertRaisesRegex(ValueError, "dependency group"):
            _check_partition(rows)

    def test_bootstrap_retains_auc_orientation_and_ties(self):
        original = [
            {"example_id": "a", "label": 0, "group_id": "a", "score": 0.0},
            {"example_id": "b", "label": 0, "group_id": "b", "score": 0.1},
            {"example_id": "x", "label": 1, "group_id": "x", "score": 0.9},
            {"example_id": "y", "label": 1, "group_id": "y", "score": 1.0},
        ]
        synthetic = [{**r, "score": 0.5} for r in original]
        result = _bootstrap({"original": original, "synthetic": synthetic}, 30, 42)
        self.assertEqual(result["original"], [1.0, 1.0])
        self.assertEqual(result["synthetic"], [0.5, 0.5])
        self.assertEqual(result["delta"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
