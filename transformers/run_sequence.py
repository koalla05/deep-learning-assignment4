"""Run the existing training notebooks sequentially, one GPU process at a time."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile


NOTEBOOKS = {
    "mbert": "05_mbert.ipynb",
    "mdeberta": "04_mdeberta.ipynb",
    "xlmr": "03_xlmr_large.ipynb",
}
ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def queue_lock(output):
    """One writer per output root within this runtime; the OS releases interrupted locks."""
    identity = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    path = Path(tempfile.gettempdir()) / f"ner-sequence-{identity}.lock"
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A training queue is already using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute_notebook(path, output, project):
    """Execute training cells after the launcher's shared setup, without mounting again."""
    notebook = json.loads(Path(path).read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    setup = "".join(cells[0]["source"])
    if "from ner.experiment import setup" not in setup or "REPO_REF =" not in setup:
        raise ValueError("Expected the standalone notebook's shared setup cell first")
    namespace = {"OUT": Path(output), "WANDB_PROJECT": project, "__name__": "__main__"}
    for index, cell in enumerate(cells[1:], start=1):
        source = "".join(cell["source"])
        print(f"Running {Path(path).name}, training cell {index}", flush=True)
        exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
    return namespace


def run_sequence(models, output, project, runner=subprocess.run):
    """Stop on failure; reruns rely on the notebooks' verified cache and recovery."""
    if not models or len(models) != len(set(models)) or any(m not in NOTEBOOKS for m in models):
        raise ValueError("Select distinct models from mbert, mdeberta, xlmr")
    directory = Path(output).parent / (
        "sequence_smoke" if Path(output).name == "runs_smoke" else "sequence"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status_path = directory / f"{stamp}.json"
    status = {"models": models, "status": "running", "jobs": []}

    def save():
        temporary = status_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(status, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(status_path)

    save()
    for model in models:
        log = directory / f"{stamp}-{model}.log"
        job = {"model": model, "status": "running", "log": str(log)}
        status["jobs"].append(job)
        save()
        print(f"Starting {model}; live log: {log}", flush=True)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            model,
            "--output",
            str(output),
        ]
        if project:
            command += ["--project", project]
        environment = {**os.environ, "MPLBACKEND": "Agg", "TOKENIZERS_PARALLELISM": "false"}
        try:
            with log.open("w") as stream:
                runner(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except BaseException:
            job["status"] = status["status"] = "interrupted_or_failed"
            save()
            raise
        job["status"] = "passed"
        save()
        print(f"Finished {model}", flush=True)
    status["status"] = "passed"
    save()
    print(f"Sequence complete: {status_path}", flush=True)
    return status_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=NOTEBOOKS, default=list(NOTEBOOKS))
    parser.add_argument("--worker", choices=NOTEBOOKS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project")
    args = parser.parse_args()

    # A notebook cancellation terminates the supervisor. Unwind subprocess.run so it also
    # kills and reaps the active worker, and let the queue persist its failed status.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if args.worker:
        context = execute_notebook(
            ROOT / "transformers" / NOTEBOOKS[args.worker], args.output, args.project
        )
        print(f"Verified ensemble exported to {context['ENSEMBLE_DIR']}", flush=True)
    else:
        with queue_lock(args.output):
            run_sequence(args.models, args.output, args.project)


if __name__ == "__main__":
    main()
