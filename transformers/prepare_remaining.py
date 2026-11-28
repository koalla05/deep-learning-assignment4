"""Seed a writable training directory from an immutable, checksummed rescue."""

import argparse
import ast
import dataclasses
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ner.artifacts import latest_checkpoint, provenance, sha256, verify_bundle, write_json  # noqa: E402


def _compatible(directory, current):
    saved = json.loads((directory / "provenance.json").read_text())
    if any(saved[key] != current[key] for key in ("inputs", "source", "packages")):
        raise ValueError(
            f"{directory.name}: inputs, training source or packages differ from the rescue"
        )


def stage_runs(backup, working, current):
    """Copy completed mDeBERTa trials and the newest valid checkpoint; never edit the rescue."""
    backup, working = Path(backup), Path(working)
    if backup.resolve() == working.resolve() or backup.resolve() in working.resolve().parents:
        raise ValueError("The working directory must be outside the immutable rescue")
    checksums = json.loads((backup / "migration-complete.json").read_text())["files"]
    staged = []
    for source in sorted((backup / "runs").glob("04-mdeberta-*")):
        if not source.is_dir():
            continue
        cfg = json.loads((source / "config.json").read_text())
        _compatible(source, current)
        destination = working / "runs" / source.name
        if destination.exists():
            if json.loads((destination / "config.json").read_text()) != cfg:
                raise ValueError("Existing working run has a different configuration")
            _compatible(destination, current)
        else:
            if (source / "complete.json").exists():
                verify_bundle(source)
                files = [path for path in source.rglob("*") if path.is_file()]
            else:
                files = [path for path in source.iterdir() if path.is_file()]
                for stage in sorted((source / "work").glob("stage-*")):
                    if (stage / "complete.json").exists():
                        verify_bundle(stage)
                        files.extend(path for path in stage.rglob("*") if path.is_file())
                    else:
                        checkpoint = latest_checkpoint(stage / "checkpoints")
                        if checkpoint:
                            files.extend(
                                path for path in Path(checkpoint).rglob("*") if path.is_file()
                            )
            temporary = working / ".seed-staging" / source.name
            temporary.mkdir(parents=True, exist_ok=True)
            for path in files:
                expected = checksums[str(path.relative_to(backup))]
                target = temporary / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or sha256(target) != expected:
                    shutil.copyfile(path, target)
                if sha256(target) != expected:
                    raise ValueError(f"Copy checksum failed: {target}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.rename(destination)
        if (destination / "complete.json").exists():
            verify_bundle(destination)
        checkpoints = []
        for stage in sorted((destination / "work").glob("stage-*")):
            checkpoint = latest_checkpoint(stage / "checkpoints")
            if checkpoint:
                checkpoints.append(str(Path(checkpoint).relative_to(working)))
        staged.append({"run": source.name, "config": cfg, "checkpoints": checkpoints})
        print("Prepared", source.name, checkpoints or "completed run", flush=True)
    if not staged:
        raise ValueError("No rescued mDeBERTa runs found")
    return staged


def prepare_notebooks(source, destination, revisions):
    """Keep the reviewed training cells and pin their pretrained model commits explicitly."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for model, filename in {"mdeberta": "04_mdeberta.ipynb", "xlmr": "03_xlmr_large.ipynb"}.items():
        notebook = json.loads((source / filename).read_text())
        replacements = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            needle = "revision=freeze_model_revision(MODEL)"
            replacements += code.count(needle)
            code = code.replace(needle, f"revision={revisions[model]!r}")
            ast.parse(code)
            cell["source"] = code.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        if replacements != 1:
            raise ValueError(f"Expected one model revision setting in {filename}")
        target = destination / filename
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        hashes[filename] = sha256(target)
    return hashes


def validate_resume_recipe(notebook, staged, output):
    """Construct actual notebook run configs without loading a model or starting training."""
    cells = [
        cell
        for cell in json.loads(Path(notebook).read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    module = ast.parse("".join(cells[1]["source"]))
    final = module.body.pop()
    if not (
        isinstance(final, ast.Expr)
        and isinstance(final.value, ast.Call)
        and isinstance(final.value.func, ast.Name)
        and final.value.func.id == "memory_check"
    ):
        raise ValueError("Expected memory_check at the end of the notebook recipe cell")
    namespace = {"OUT": Path(output), "WANDB_PROJECT": None}
    exec(compile(module, str(notebook), "exec"), namespace)
    namespace["run_or_load"] = lambda cfg, *_: json.loads(json.dumps(dataclasses.asdict(cfg)))
    for item in staged:
        cfg = item["config"]
        constructed = namespace["train"](cfg["learning_rate"], cfg["fold"])
        if constructed != cfg:
            raise ValueError(f"Notebook would change the saved configuration of {item['run']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--working", required=True, type=Path)
    args = parser.parse_args()
    from run_sequence import queue_lock

    with queue_lock(args.working / "runs"):
        current = provenance()
        staged = stage_runs(args.backup, args.working, current)
        revisions = {
            "mdeberta": staged[0]["config"]["revision"],
            "xlmr": json.loads(
                (
                    args.backup / "verification/t4-20261128/smoke-final/runs/xlmr-large/config.json"
                ).read_text()
            )["revision"],
        }
        prepared = args.working / "prepared-notebooks"
        hashes = prepare_notebooks(ROOT / "transformers", prepared, revisions)
        validate_resume_recipe(prepared / "04_mdeberta.ipynb", staged, args.working / "runs")
        write_json(
            args.working / "launch-plan.json",
            {
                "models": ["mdeberta", "xlmr"],
                "backup": str(args.backup),
                "working": str(args.working),
                "staged_runs": staged,
                "model_revisions": revisions,
                "notebook_sha256": hashes,
                "provenance": current,
            },
        )
        print("REMAINING_TRAINING_PREPARED", args.working, flush=True)


if __name__ == "__main__":
    main()
