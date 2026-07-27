"""The URL-default policy, which is the package's central promise about the network."""

import pytest

from odynamech import DrivelineRelease
from odynamech import PackagedCorpus
from odynamech import RawMirror
from odynamech.config import DRIVELINE_RELEASE_BASE
from odynamech.config import GESTURE_ASSETS
from odynamech.sources import _is_archive


def test_only_driveline_has_a_default_url() -> None:
    """Exactly one source may be constructed with no arguments."""
    assert DrivelineRelease().base_url == DRIVELINE_RELEASE_BASE
    with pytest.raises(TypeError):
        RawMirror()
    with pytest.raises(TypeError):
        PackagedCorpus()


@pytest.mark.parametrize("empty", ["", None])
def test_empty_url_is_rejected_rather_than_defaulted(empty) -> None:
    """An empty URL must not silently fall back to upstream."""
    with pytest.raises(ValueError, match="no default"):
        RawMirror(base_url=empty)
    with pytest.raises(ValueError, match="no default"):
        PackagedCorpus(url=empty)


def test_driveline_covers_every_published_table() -> None:
    """Every signal table plus both scalar tables, per gesture, and nothing else."""
    assets = list(DrivelineRelease().assets())
    expected = sum(len(v) for v in GESTURE_ASSETS.values()) + 2 * len(GESTURE_ASSETS)
    assert len(assets) == expected
    assert all(a.url.startswith(DRIVELINE_RELEASE_BASE) for a in assets if a.archive)


def test_c3d_archives_are_never_requested() -> None:
    """Raw marker trajectories are redundant with `landmarks` and are excluded."""
    for source in (DrivelineRelease(), RawMirror(base_url="https://example.invalid/m")):
        assert not any("c3d" in a.url.lower() for a in source.assets())


def test_gestures_never_share_a_destination() -> None:
    """Upstream reuses table names across gestures, so the paths must not collide."""
    for source in (DrivelineRelease(), RawMirror(base_url="https://example.invalid/m")):
        paths = [a.relpath for a in source.assets()]
        assert len(paths) == len(set(paths))
        assert all("/" in p for p in paths)


def test_raw_mirror_honours_a_separate_scalar_base() -> None:
    """Scalar tables may live somewhere other than the signal archives."""
    source = RawMirror(base_url="https://sig.invalid", scalar_base_url="https://scalars.invalid")
    csvs = [a for a in source.assets() if not a.archive]
    assert csvs and all(a.url.startswith("https://scalars.invalid") for a in csvs)


@pytest.mark.parametrize(
    "base", ["https://m.invalid/mirror", "https://m.invalid/mirror/", "https://m.invalid/mirror///"]
)
def test_raw_mirror_never_emits_a_doubled_slash(base: str) -> None:
    """A trailing slash on the base must not survive into the fetched URLs.

    Hosts that normalise paths hide this; an S3-style host treats `//` as a
    literal key separator and answers 404 instead.
    """
    source = RawMirror(base_url=base)
    assert source.base_url == "https://m.invalid/mirror"
    assert all(a.url.startswith("https://m.invalid/mirror/") for a in source.assets())
    assert not any("//" in a.url.removeprefix("https://") for a in source.assets())


def test_every_joinable_base_is_normalised() -> None:
    """Not just `RawMirror.base_url`: the scalar base and both release bases too."""
    mirror = RawMirror(base_url="https://sig.invalid/s/", scalar_base_url="https://scalars.invalid/c/")
    release = DrivelineRelease(base_url="https://rel.invalid/r/", raw_base_url="https://raw.invalid/g/")
    for source in (mirror, release):
        assert not any("//" in a.url.removeprefix("https://") for a in source.assets())


@pytest.mark.parametrize("degenerate", ["https://", "/", "///"])
def test_a_base_that_is_only_slashes_is_rejected(degenerate: str) -> None:
    """Stripping must not turn a useless base into one that merely looks valid."""
    with pytest.raises(ValueError, match="no host or path"):
        RawMirror(base_url=degenerate)
    with pytest.raises(ValueError, match="no host or path"):
        DrivelineRelease(base_url=degenerate)


def test_raw_mirror_passes_checksums_through() -> None:
    """A mirror can be verified as well as trusted."""
    rel = "pitching/joint_angles.zip"
    source = RawMirror(base_url="https://m.invalid", checksums={rel: "ab" * 32})
    assert next(a for a in source.assets() if a.relpath == rel).sha256 == "ab" * 32


def test_packaged_corpus_is_prebuilt_and_single_asset() -> None:
    """The prebuilt path fetches one thing and builds nothing."""
    source = PackagedCorpus(url="https://x.invalid/corpus.tar.gz", sha256="cd" * 32)
    assets = list(source.assets())
    assert source.prebuilt is True
    assert len(assets) == 1
    assert assets[0].archive is True
    assert assets[0].sha256 == "cd" * 32


def test_packaged_corpus_accepts_a_bare_pack() -> None:
    """A plain `.safetensors` is not an archive and must not be run through extract."""
    asset = next(iter(PackagedCorpus(url="https://x.invalid/c.safetensors").assets()))
    assert asset.archive is False
    assert asset.relpath == "c.safetensors"


def test_raw_sources_are_not_prebuilt() -> None:
    """Raw paths must be built; only `PackagedCorpus` arrives finished."""
    assert DrivelineRelease().prebuilt is False
    assert RawMirror(base_url="https://x.invalid").prebuilt is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("c.tar.gz", True),
        ("c.tgz", True),
        ("c.tar.zst", True),
        ("c.zip", True),
        ("c.safetensors", False),
        ("c.csv", False),
    ],
)
def test_archive_detection(name: str, expected: bool) -> None:
    """Archive detection drives whether extraction is attempted at all."""
    assert _is_archive(name) is expected


def test_sources_are_frozen() -> None:
    """Sources are values, so a fetch cannot mutate the thing that described it."""
    source = DrivelineRelease()
    with pytest.raises((AttributeError, TypeError)):
        source.base_url = "https://elsewhere.invalid"
