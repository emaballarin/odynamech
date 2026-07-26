"""Shared fixtures. The suite is data-free by default; corpus tests opt in via an env var."""

import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items) -> None:
    """Deselect network tests unless explicitly requested with `-m network`."""
    if "network" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="reaches the network; run with '-m network'")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def corpus_path() -> Path:
    """A built corpus, or skip.

    Set `ODYNAMECH_TEST_CORPUS` to a pack file or an intermediates directory. The
    data is third-party and non-commercially licensed, so it is never downloaded
    by the test suite and never present in CI.
    """
    raw = os.environ.get("ODYNAMECH_TEST_CORPUS")
    if not raw:
        pytest.skip("set ODYNAMECH_TEST_CORPUS to a built corpus to run these")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"ODYNAMECH_TEST_CORPUS points at a missing path: {path}")
    return path


@pytest.fixture(scope="session")
def raw_root() -> Path | None:
    """Raw upstream CSVs, if the caller has them; `None` otherwise."""
    raw = os.environ.get("ODYNAMECH_TEST_RAW")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


@pytest.fixture(scope="session")
def obp(corpus_path: Path):
    """An open corpus handle."""
    from odynamech import OBP

    return OBP(corpus_path)
