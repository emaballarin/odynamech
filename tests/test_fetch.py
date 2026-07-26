"""Acquisition machinery, exercised end-to-end against local files — no network."""

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from odynamech.config import DATA_NOTICE
from odynamech.fetch import acquire
from odynamech.fetch import download
from odynamech.fetch import extract
from odynamech.fetch import FetchError
from odynamech.fetch import NOTICE_FILENAME
from odynamech.fetch import write_notice
from odynamech.sources import PackagedCorpus
from odynamech.sources import RawMirror


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    """A zip archive holding `members`."""
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


def _write_tar(path: Path, members: dict[str, bytes]) -> Path:
    """A gzipped tar archive holding `members`."""
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return path


# ------------------------------------------------------------------- download


def test_download_copies_a_local_file(tmp_path: Path) -> None:
    """A `file://` source is copied, so a local mirror costs nothing."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    out = download(src.as_uri(), tmp_path / "out" / "dst.bin")
    assert out.read_bytes() == b"payload"


def test_download_verifies_a_checksum(tmp_path: Path) -> None:
    """A wrong digest must fail loudly and leave nothing behind."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "dst.bin"
    with pytest.raises(FetchError, match="sha256 mismatch"):
        download(src.as_uri(), dest, sha256="00" * 32)
    assert not dest.exists()


def test_download_accepts_a_correct_checksum(tmp_path: Path) -> None:
    """The happy path keeps the file."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    assert download(src.as_uri(), tmp_path / "dst.bin", sha256=digest).exists()


def test_download_skips_a_file_already_present(tmp_path: Path) -> None:
    """Re-running an acquisition must not re-transfer what is already correct."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "dst.bin"
    download(src.as_uri(), dest)
    dest.write_bytes(b"sentinel")  # would be clobbered if the skip were broken
    download(src.as_uri(), dest)
    assert dest.read_bytes() == b"sentinel"


def test_download_refetches_when_a_cached_file_fails_its_checksum(tmp_path: Path) -> None:
    """A corrupted cache entry is replaced rather than trusted."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "dst.bin"
    dest.write_bytes(b"corrupt")
    download(src.as_uri(), dest, sha256=hashlib.sha256(b"payload").hexdigest())
    assert dest.read_bytes() == b"payload"


def test_download_reports_a_missing_local_source(tmp_path: Path) -> None:
    """A missing mirror path is an error, not an empty file."""
    with pytest.raises(FetchError, match="does not exist"):
        download((tmp_path / "nope.bin").as_uri(), tmp_path / "out.bin")


# -------------------------------------------------------------------- extract


def test_extract_zip_and_tar(tmp_path: Path) -> None:
    """Both archive families unpack, and the archive is removed afterwards."""
    for maker, name in ((_write_zip, "a.zip"), (_write_tar, "a.tar.gz")):
        archive = maker(tmp_path / name, {"inner.csv": b"a,b\n1,2\n"})
        out = tmp_path / name.replace(".", "_")
        written = extract(archive, out)
        assert [p.name for p in written] == ["inner.csv"]
        assert (out / "inner.csv").read_bytes() == b"a,b\n1,2\n"
        assert not archive.exists()


def test_extract_rejects_a_traversal_entry(tmp_path: Path) -> None:
    """An entry escaping the destination aborts rather than being sanitised quietly."""
    archive = _write_zip(tmp_path / "evil.zip", {"../escaped.csv": b"x"})
    with pytest.raises(FetchError, match="escapes"):
        extract(archive, tmp_path / "out")


def test_extract_rejects_a_non_archive(tmp_path: Path) -> None:
    """Something that is not an archive must not be silently ignored."""
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"not an archive")
    with pytest.raises(FetchError, match="not a zip or tar"):
        extract(plain, tmp_path / "out")


def test_extract_skips_macosx_cruft(tmp_path: Path) -> None:
    """`__MACOSX` sidecars in upstream zips must not reach the data directory."""
    archive = _write_zip(tmp_path / "a.zip", {"__MACOSX/._x": b"junk", "real.csv": b"ok"})
    written = extract(archive, tmp_path / "out")
    assert [p.name for p in written] == ["real.csv"]


# --------------------------------------------------------------------- notice


def test_notice_is_written_beside_the_data(tmp_path: Path) -> None:
    """Data outlives the shell that fetched it; the terms must travel with it."""
    path = write_notice(tmp_path)
    assert path.name == NOTICE_FILENAME
    body = path.read_text()
    assert DATA_NOTICE in body
    for phrase in ("CC BY-NC-SA", "NonCommercial", "ShareAlike", "professional sports", "financial-analysis"):
        assert phrase in body


# --------------------------------------------------------------------- acquire


def test_acquire_raw_mirror_lands_gestures_separately(tmp_path: Path) -> None:
    """Upstream reuses table names across gestures; the two must not collide."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    from odynamech.config import GESTURE_ASSETS

    for gesture, tables in GESTURE_ASSETS.items():
        for table in tables:
            _write_zip(mirror / f"{gesture}_{table}.zip", {f"{table}.csv": f"{gesture}/{table}".encode()})
        for name in ("metadata.csv", "poi_metrics.csv"):
            (mirror / f"{gesture}_{name}").write_bytes(f"{gesture}/{name}".encode())

    dest = tmp_path / "raw"
    result = acquire(RawMirror(base_url=mirror.as_uri()), dest)

    assert not result.prebuilt
    assert (dest / "pitching" / "joint_angles.csv").read_bytes() == b"pitching/joint_angles"
    assert (dest / "hitting" / "joint_angles.csv").read_bytes() == b"hitting/joint_angles"
    assert (dest / "pitching" / "metadata.csv").read_bytes() == b"pitching/metadata.csv"
    assert (dest / NOTICE_FILENAME).is_file()


def test_acquire_normalises_a_differently_named_member(tmp_path: Path) -> None:
    """The single-CSV-named-after-the-table convention is checked, not assumed."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    from odynamech.config import GESTURE_ASSETS

    for gesture, tables in GESTURE_ASSETS.items():
        for table in tables:
            inner = "SURPRISE.csv" if (gesture, table) == ("pitching", "landmarks") else f"{table}.csv"
            _write_zip(mirror / f"{gesture}_{table}.zip", {inner: b"x"})
        for name in ("metadata.csv", "poi_metrics.csv"):
            (mirror / f"{gesture}_{name}").write_bytes(b"x")

    dest = tmp_path / "raw"
    acquire(RawMirror(base_url=mirror.as_uri()), dest)
    assert (dest / "pitching" / "landmarks.csv").is_file()
    assert not (dest / "pitching" / "SURPRISE.csv").exists()


def test_acquire_packaged_corpus_extracts_a_tarball(tmp_path: Path) -> None:
    """The prebuilt path unpacks and builds nothing."""
    tar = _write_tar(tmp_path / "corpus.tar.gz", {"odynamech_corpus.safetensors": b"PACK", "BUILD_REPORT.md": b"# r"})
    dest = tmp_path / "tensors"
    result = acquire(PackagedCorpus(url=tar.as_uri()), dest)

    assert result.prebuilt
    assert (dest / "odynamech_corpus.safetensors").read_bytes() == b"PACK"
    assert (dest / "BUILD_REPORT.md").is_file()


def test_acquire_packaged_corpus_checks_its_digest(tmp_path: Path) -> None:
    """The prebuilt path is where a wrong digest matters most."""
    tar = _write_tar(tmp_path / "corpus.tar.gz", {"x": b"y"})
    with pytest.raises(FetchError, match="sha256 mismatch"):
        acquire(PackagedCorpus(url=tar.as_uri(), sha256="00" * 32), tmp_path / "out")


def test_acquire_packaged_corpus_accepts_a_bare_pack(tmp_path: Path) -> None:
    """A plain `.safetensors` is copied through without an extraction attempt."""
    pack = tmp_path / "c.safetensors"
    pack.write_bytes(b"PACK")
    dest = tmp_path / "tensors"
    acquire(PackagedCorpus(url=pack.as_uri()), dest)
    assert (dest / "c.safetensors").read_bytes() == b"PACK"
