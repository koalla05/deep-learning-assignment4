"""Durable run bundles: publish completion last, verify before reuse."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from .data import REPO

INPUT_FILES = (
    "data/cv_rows.parquet",
    "data/test.csv",
    "data/labeling_sample.csv",
    "pseudo_label/output/test_pseudo.parquet",
    "synthetic/output/full1_clean.parquet",
    "synthetic/output/full1_adversarial.parquet",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a JSON file only once its contents have been flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def provenance():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "pandas",
        "pyarrow",
        "matplotlib",
        "wandb",
        "sentencepiece",
        "protobuf",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "git_revision": revision,
        "python": platform.python_version(),
        "packages": packages,
        "inputs": {name: sha256(REPO / name) for name in INPUT_FILES},
        "source": {
            str(p.relative_to(REPO)): sha256(p) for p in sorted((REPO / "ner").glob("*.py"))
        },
    }


def publish_bundle(directory, files):
    """The manifest is the last write; it covers every required file, including weights."""
    directory = Path(directory)
    files = sorted(set(str(Path(p).relative_to(directory)) for p in files))
    if not files:
        raise ValueError("cannot publish an empty bundle")
    write_json(
        directory / "complete.json",
        {
            "version": 1,
            "files": {name: sha256(directory / name) for name in files},
        },
    )


def verify_bundle(directory):
    directory = Path(directory)
    manifest = directory / "complete.json"
    if not manifest.exists():
        raise ValueError(f"{directory}: export is incomplete (no complete.json)")
    files = json.loads(manifest.read_text())["files"]
    if not files:
        raise ValueError(f"{directory}: empty completion manifest")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{directory}: missing or damaged artifact {name}")


def row_metadata(sets):
    """Content identities are independent of tokenizer and saved alongside row order."""
    return {
        name: [
            {
                "row": i,
                "row_id": f"{name}:{i}",
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "locations": truth,
            }
            for i, (text, truth) in enumerate(zip(texts, labels))
        ]
        for name, (texts, labels) in sets.items()
    }


def latest_checkpoint(directory):
    """Skip half-written newer checkpoints in favor of an older verified one."""
    candidates = sorted(
        Path(directory).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True
    )
    for candidate in candidates:
        try:
            verify_bundle(candidate)
        except (ValueError, OSError, KeyError):
            continue
        return str(candidate)
    return None
