"""The overnight queue must finish one worker before starting the next."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "run_sequence", Path(__file__).resolve().parents[1] / "transformers/run_sequence.py"
)
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


class SequenceTests(unittest.TestCase):
    def test_models_run_in_order_with_separate_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            started = []

            def worker(command, **kwargs):
                model = command[command.index("--worker") + 1]
                started.append(model)
                self.assertTrue(kwargs["check"])
                self.assertEqual(kwargs["env"]["MPLBACKEND"], "Agg")
                kwargs["stdout"].write(f"Finished {model}\n")

            path = sequence.run_sequence(
                ["mbert", "xlmr"], Path(directory) / "runs", None, runner=worker
            )
            self.assertEqual(started, ["mbert", "xlmr"])
            status = json.loads(path.read_text())
            self.assertEqual(status["status"], "passed")
            for job in status["jobs"]:
                self.assertEqual(job["status"], "passed")
                self.assertIn(job["model"], Path(job["log"]).read_text())

    def test_failure_stops_the_queue_and_preserves_its_log(self):
        with tempfile.TemporaryDirectory() as directory:
            started = []

            def worker(command, **kwargs):
                started.append(command[command.index("--worker") + 1])
                kwargs["stdout"].write("training failed\n")
                raise subprocess.CalledProcessError(1, command)

            with self.assertRaises(subprocess.CalledProcessError):
                sequence.run_sequence(
                    ["mbert", "xlmr"], Path(directory) / "runs", None, runner=worker
                )
            self.assertEqual(started, ["mbert"])
            status = json.loads(next((Path(directory) / "sequence").glob("*.json")).read_text())
            self.assertEqual(status["status"], "interrupted_or_failed")
            self.assertIn("training failed", Path(status["jobs"][0]["log"]).read_text())

    def test_duplicate_jobs_are_rejected_before_starting(self):
        with self.assertRaises(ValueError):
            sequence.run_sequence(["mbert", "mbert"], "/unused/runs", None)
