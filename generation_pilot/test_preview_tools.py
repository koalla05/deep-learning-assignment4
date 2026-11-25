"""Focused correctness checks for Unicode span creation and slot validation."""

import tempfile
import unittest
from pathlib import Path

from preview_tools import assemble, build


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.bank = {
            "city": {"resource_id": "city", "surface": "Чернівцях"},
            "street": {"resource_id": "street", "surface": "вул. Солом’янська, 28-б"},
        }

    def test_unicode_repeated_mentions_and_full_address(self):
        draft = "👨‍👩‍👧 У <LOC_1>.\nНа <LOC_2>; знову у <LOC_3> е\u0301."
        specs = [
            {"placeholder": f"<LOC_{i}>", "resource_id": key}
            for i, key in enumerate(["city", "street", "city"], 1)
        ]
        text, spans, locations, edit_map = assemble(draft, specs, self.bank)
        self.assertEqual(locations, ["Чернівцях", "вул. Солом’янська, 28-б", "Чернівцях"])
        self.assertEqual([text[a:b] for a, b in spans], locations)
        self.assertEqual(spans[0][0], len("👨‍👩‍👧 У "))
        self.assertTrue(text.endswith("е\u0301."))
        self.assertEqual(len([e for e in edit_map if e["kind"] == "location"]), 3)

    def test_negative(self):
        text, spans, locations, _ = assemble("Завтра прийому не буде.", [], self.bank)
        self.assertEqual(text, "Завтра прийому не буде.")
        self.assertEqual(spans, [])
        self.assertEqual(locations, [])

    def test_repeated_or_missing_placeholder_rejected(self):
        slot = [{"placeholder": "<LOC_1>", "resource_id": "city"}]
        for draft in ["<LOC_1> і <LOC_1>", "У місті.", "<LOC_2>"]:
            with self.subTest(draft=draft), self.assertRaises(ValueError):
                assemble(draft, slot, self.bank)

    def test_undeclared_and_malformed_slots_rejected(self):
        for draft in ["<LOC_1>", "<LOC_bad>"]:
            with self.subTest(draft=draft), self.assertRaises(ValueError):
                assemble(draft, [], self.bank)

    def test_build_preserves_existing_reviewed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            records = batch / "candidates.jsonl"
            records.write_text("reviewed records must remain unchanged\n")
            with self.assertRaisesRegex(ValueError, "Candidates already exist"):
                build(batch)
            self.assertEqual(records.read_text(), "reviewed records must remain unchanged\n")

    def test_build_rejects_conflicting_correction_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "either a recipe revision or editorial"):
                build(Path(directory), apply_revision=True, apply_editorial=True)


if __name__ == "__main__":
    unittest.main()
