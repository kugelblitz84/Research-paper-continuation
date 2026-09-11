"""Append-only registry events and immutable checkpoint generations."""

import csv
import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch

from src.config import ROOT
from src.reproducibility import sha256


def utc_now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(f"pid={os.getpid()} time={utc_now()}")
    try:
        yield
    finally:
        path.unlink()


def append_registry(row):
    path = ROOT / "experiments/experiment_registry.csv"
    with exclusive_lock(path.with_suffix(".lock")):
        with path.open(newline="", encoding="utf-8") as f:
            columns = next(csv.reader(f))
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns, lineterminator="\n").writerow({c: row.get(c, "") for c in columns})
            f.flush()
            os.fsync(f.fileno())


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create means existing expensive artifacts cannot be overwritten.
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256(path),
    }


def load_checkpoint(reference):
    path = Path(reference["path"])
    if not path.is_absolute():
        path = ROOT / path
    if sha256(path) != reference["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    # Only load this repository's trusted, locally generated training state.
    return torch.load(path, map_location="cpu", weights_only=False)
