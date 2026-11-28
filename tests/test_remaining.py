"""Resume preparation must preserve the rescue and select only valid recovery state."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ner.artifacts import publish_bundle, sha256, write_json

spec = importlib.util.spec_from_file_location(
    "prepare_remaining", Path(__file__).resolve().parents[1] / "transformers/prepare_remaining.py"
)
remaining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remaining)


class RemainingTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backup, self.working = self.root / "rescue", self.root / "training"
        self.current = {"inputs": {}, "source": {}, "packages": {}}
        self.name = "04-mdeberta-fixture-lr3e-05-fold1"
        run = self.backup / "runs" / self.name
        write_json(run / "config.json", {"name": self.name, "revision": "a" * 40})
        write_json(run / "provenance.json", self.current)
        for step in (1750, 2000):
            checkpoint = run / "work/stage-1/checkpoints" / f"checkpoint-{step}"
            write_json(checkpoint / "trainer_state.json", {"global_step": step})
            weights = checkpoint / "model.safetensors"
            weights.write_bytes(f"saved weights at {step}".encode())
            publish_bundle(checkpoint, [weights, checkpoint / "trainer_state.json"])
        partial = run / "work/stage-1/checkpoints/checkpoint-2250"
        partial.mkdir()
        (partial / "tokenizer_config.json").write_bytes(b"")
        write_json(self.backup / "runs/04-mdeberta-fixture-memory.json", {"status": "passed"})
        write_json(
            self.backup / "migration-complete.json",
            {
                "files": {
                    str(path.relative_to(self.backup)): sha256(path)
                    for path in self.backup.rglob("*")
                    if path.is_file()
                }
            },
        )

    def test_only_the_newest_valid_checkpoint_is_copied_and_the_rescue_is_unchanged(self):
        before = {str(path): sha256(path) for path in self.backup.rglob("*") if path.is_file()}
        staged = remaining.stage_runs(self.backup, self.working, self.current)
        checkpoints = self.working / "runs" / self.name / "work/stage-1/checkpoints"
        self.assertEqual([path.name for path in checkpoints.iterdir()], ["checkpoint-2000"])
        self.assertIn("checkpoint-2000", staged[0]["checkpoints"][0])
        self.assertEqual(
            before, {str(path): sha256(path) for path in self.backup.rglob("*") if path.is_file()}
        )

    def test_preparation_never_rolls_back_a_newer_working_checkpoint(self):
        remaining.stage_runs(self.backup, self.working, self.current)
        checkpoint = self.working / "runs" / self.name / "work/stage-1/checkpoints/checkpoint-2500"
        write_json(checkpoint / "trainer_state.json", {"global_step": 2500})
        publish_bundle(checkpoint, [checkpoint / "trainer_state.json"])
        staged = remaining.stage_runs(self.backup, self.working, self.current)
        self.assertIn("checkpoint-2500", staged[0]["checkpoints"][0])

    def test_changed_training_inputs_stop_preparation_before_copying(self):
        changed = {**self.current, "inputs": {"training": "different"}}
        with self.assertRaisesRegex(ValueError, "differ"):
            remaining.stage_runs(self.backup, self.working, changed)
        self.assertFalse((self.working / "runs").exists())

    def test_completed_trials_are_copied_as_verified_inference_bundles(self):
        name = "04-mdeberta-fixture-lr3e-05-fold0"
        source = self.backup / "runs" / name
        write_json(source / "config.json", {"name": name, "revision": "a" * 40})
        write_json(source / "provenance.json", self.current)
        weights = source / "model" / "model.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"completed trial weights")
        publish_bundle(source, [source / "config.json", source / "provenance.json", weights])
        write_json(
            self.backup / "migration-complete.json",
            {
                "files": {
                    str(path.relative_to(self.backup)): sha256(path)
                    for path in self.backup.rglob("*")
                    if path.is_file() and path.name != "migration-complete.json"
                }
            },
        )
        remaining.stage_runs(self.backup, self.working, self.current)
        remaining.verify_bundle(self.working / "runs" / name)
        self.assertEqual(
            (self.working / "runs" / name / "model/model.safetensors").read_bytes(),
            weights.read_bytes(),
        )

    def test_notebook_copies_pin_revisions_without_changing_training_settings(self):
        source = Path(__file__).resolve().parents[1] / "transformers"
        destination = self.root / "prepared"
        hashes = remaining.prepare_notebooks(
            source, destination, {"mdeberta": "a" * 40, "xlmr": "b" * 40}
        )
        for model, filename in [("mdeberta", "04_mdeberta.ipynb"), ("xlmr", "03_xlmr_large.ipynb")]:
            original = json.loads((source / filename).read_text())
            prepared = json.loads((destination / filename).read_text())
            for old, new in zip(original["cells"], prepared["cells"]):
                expected = "".join(old["source"]).replace(
                    "revision=freeze_model_revision(MODEL)",
                    f"revision={('a' if model == 'mdeberta' else 'b') * 40!r}",
                )
                self.assertEqual("".join(new["source"]), expected)
            self.assertEqual(hashes[filename], sha256(destination / filename))
