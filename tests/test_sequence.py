"""The overnight queue must finish one worker before starting the next."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "run_sequence", Path(__file__).resolve().parents[1] / "transformers/run_sequence.py"
)
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


class SequenceTests(unittest.TestCase):
    def test_terminating_the_queue_stops_its_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "transformers"
            scripts.mkdir()
            script = scripts / "run_sequence.py"
            script.write_text(Path(sequence.__file__).read_text())
            source = (
                "import os,time\n"
                "OUT.mkdir(parents=True, exist_ok=True)\n"
                "(OUT / 'worker.pid').write_text(str(os.getpid()))\n"
                "time.sleep(60)\n"
            )
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["REPO_REF = 'fixture'\nfrom ner.experiment import setup"],
                    },
                    {"cell_type": "code", "source": [source]},
                ]
            }
            (scripts / "05_mbert.ipynb").write_text(json.dumps(notebook))
            output = root / "runs"
            with (root / "queue.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sequence.sys.executable,
                        str(script),
                        "--models",
                        "mbert",
                        "--output",
                        str(output),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                worker_pid = None
                try:
                    deadline = time.monotonic() + 10
                    while not (output / "worker.pid").exists() and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertTrue((output / "worker.pid").exists())
                    worker_pid = int((output / "worker.pid").read_text())
                    duplicate = subprocess.run(
                        [
                            sequence.sys.executable,
                            str(script),
                            "--models",
                            "mbert",
                            "--output",
                            str(output),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(duplicate.returncode, 0)
                    self.assertIn("already using", duplicate.stderr)
                    process.terminate()
                    process.wait(timeout=10)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(worker_pid, 0)
                    status = json.loads(next((root / "sequence").glob("*.json")).read_text())
                    self.assertEqual(status["status"], "interrupted_or_failed")
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    if worker_pid:
                        try:
                            os.kill(worker_pid, 9)
                        except ProcessLookupError:
                            pass

    def test_two_real_workers_train_export_and_reload_before_advancing(self):
        """Use real processes and notebook cells, replacing only data and encoder size."""
        repo = Path(sequence.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "transformers"
            scripts.mkdir()
            (scripts / "run_sequence.py").write_text(Path(sequence.__file__).read_text())
            prepared = root / "prepared-notebooks"
            prepared.mkdir()
            for model in ("mbert", "mdeberta"):
                notebook = json.loads(
                    (repo / "transformers" / sequence.NOTEBOOKS[model]).read_text()
                )
                prelude = f'''
import os
from unittest.mock import patch
from test_training_artifacts import LocalTrainingTests
from ner import recipes
from ner.preflight import memory_check as cpu_memory_check

assert os.environ["NER_QUEUE_TEST_CREDENTIAL"] == "fixture-only"
fixture = LocalTrainingTests()
fixture.setUp()
recipes.OPTIMIZATION = dict(epochs=1, batch_size=2, max_length=16, stride=4, precision="fp32")
recipes.STAGE2 = dict(stage2_data=fixture.cfg.stage2_data, stage2_epochs=1, stage2_lr=2e-5)
patch("ner.experiment.freeze_model_revision", return_value=None).start()
patch("ner.preflight.memory_check", side_effect=lambda cfg, out: cpu_memory_check(cfg, out, require_cuda=False)).start()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "{model}.pid").write_text(str(os.getpid()))
'''
                first_training = True
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    if "MODEL = " in source:
                        import re

                        source = re.sub(
                            r"^MODEL = .+$",
                            "MODEL = str(fixture.model_dir)",
                            source,
                            flags=re.MULTILINE,
                        )
                        if first_training:
                            source = prelude + source
                            first_training = False
                    cell["source"] = source.splitlines(keepends=True)
                (prepared / sequence.NOTEBOOKS[model]).write_text(json.dumps(notebook))

            output = root / "runs"
            environment = {
                "PYTHONPATH": os.pathsep.join([str(repo / "tests"), str(repo)]),
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "NER_QUEUE_TEST_CREDENTIAL": "fixture-only",
            }
            with patch.dict(os.environ, environment):
                with (root / "supervisor.log").open("w") as log:
                    result = subprocess.run(
                        [
                            sequence.sys.executable,
                            str(scripts / "run_sequence.py"),
                            "--models",
                            "mbert",
                            "mdeberta",
                            "--output",
                            str(output),
                            "--notebooks-dir",
                            str(prepared),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
            self.assertEqual(result.returncode, 0, (root / "supervisor.log").read_text())
            from ner.artifacts import verify_bundle

            statuses = list((root / "sequence").glob("*.json"))
            status = json.loads(statuses[0].read_text())
            self.assertEqual(status["status"], "passed")
            self.assertEqual([job["model"] for job in status["jobs"]], ["mbert", "mdeberta"])
            self.assertEqual(len(list(output.glob("*/complete.json"))), 8)
            self.assertEqual(len(list((root / "ensembles").glob("*/complete.json"))), 2)
            for manifest in list(output.glob("*/complete.json")) + list(
                (root / "ensembles").glob("*/complete.json")
            ):
                verify_bundle(manifest.parent)
                self.assertTrue((manifest.parent / "submission.csv").is_file())
            for model, job in zip(("mbert", "mdeberta"), status["jobs"]):
                log = Path(job["log"]).read_text()
                self.assertIn("Verified ensemble exported", log)
                self.assertNotIn("fixture-only", log)
                with self.assertRaises(ProcessLookupError):
                    os.kill(int((output / f"{model}.pid").read_text()), 0)

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
