"""Helpers for resolving valid training run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_RUN_FILES = ("hyperparameters.json", "best.pth")


def is_valid_run(run_dir: str | Path) -> bool:
    """Return True when a directory contains a config and checkpoint."""
    path = Path(run_dir)
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_RUN_FILES)


def find_latest_run(artifacts: str | Path = "artifacts") -> Path | None:
    """Return the newest complete run directory, skipping incomplete runs."""
    root = Path(artifacts)
    if not root.is_dir():
        return None

    candidates = [item for item in root.iterdir() if item.is_dir() and is_valid_run(item)]
    if not candidates:
        return None

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_run(
    run: str | Path | None = None,
    artifacts: str | Path = "artifacts",
    fallback: str | Path | None = None,
) -> Path:
    """Resolve an explicit run, the newest complete run, or a fallback."""
    if run:
        path = Path(run).expanduser()
        if not is_valid_run(path):
            raise FileNotFoundError(
                f"{path} is not a valid run directory. Expected {' and '.join(REQUIRED_RUN_FILES)}."
            )
        return path

    latest = find_latest_run(artifacts)
    if latest is not None:
        return latest

    if fallback:
        path = Path(fallback).expanduser()
        if is_valid_run(path):
            return path

    raise FileNotFoundError(
        f"No complete run found in {Path(artifacts).resolve()}. "
        "Pass --run, set CHEST_XAI_RUN, or train a model first."
    )


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def read_hyperparameters(run_dir: str | Path) -> dict[str, Any]:
    """Read a run's hyperparameters.json."""
    return read_json(Path(run_dir) / "hyperparameters.json")
