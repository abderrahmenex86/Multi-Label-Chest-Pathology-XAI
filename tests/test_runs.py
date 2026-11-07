from __future__ import annotations

import json
from pathlib import Path

from src.runs import find_latest_run, is_valid_run, resolve_run


def make_run(directory: Path, with_checkpoint: bool = True) -> Path:
    directory.mkdir(parents=True)
    (directory / "hyperparameters.json").write_text(json.dumps({"classes": 14}))
    if with_checkpoint:
        (directory / "best.pth").write_bytes(b"not-a-real-checkpoint")
    return directory


def test_is_valid_run_requires_config_and_checkpoint(tmp_path: Path) -> None:
    complete = make_run(tmp_path / "complete")
    incomplete = make_run(tmp_path / "incomplete", with_checkpoint=False)

    assert is_valid_run(complete)
    assert not is_valid_run(incomplete)


def test_find_latest_run_skips_incomplete_directories(tmp_path: Path) -> None:
    make_run(tmp_path / "older")
    latest = make_run(tmp_path / "newer")
    make_run(tmp_path / "broken", with_checkpoint=False)

    assert find_latest_run(tmp_path) == latest


def test_resolve_run_uses_fallback_when_artifacts_missing(tmp_path: Path) -> None:
    fallback = make_run(tmp_path / "checkpoint")

    resolved = resolve_run(None, artifacts=tmp_path / "missing", fallback=fallback)

    assert resolved == fallback
