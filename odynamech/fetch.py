"""Download and unpack whatever a `Source` names, with resume, retry and verification.

Uses only the standard library, so installing `odynamech` never drags in an HTTP
stack. Downloads resume via HTTP `Range` when a partial file is present, which
matters because the raw upstream tables are ~0.9 GB.
"""

import hashlib
import logging
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import ATTRIBUTION
from .config import DATA_NOTICE
from .config import raw_dir
from .sources import Source

log = logging.getLogger(__name__)

USER_AGENT = "odynamech (+https://github.com/emaballarin/odynamech)"
CHUNK = 1 << 20
NOTICE_FILENAME = "DATA-LICENCE-NOTICE.txt"

ProgressFn = Callable[[str, int, int | None], None]


class FetchError(RuntimeError):
    """A download or extraction failed, or arrived with the wrong bytes."""


@dataclass(slots=True)
class FetchResult:
    """What an acquisition produced."""

    root: Path
    files: list[Path]
    source: str
    prebuilt: bool

    def __repr__(self) -> str:
        """Compact summary."""
        return f"<FetchResult {self.source} -> {self.root} ({len(self.files)} files)>"


def _sha256_file(path: Path) -> str:
    """Content hash of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _open(url: str, offset: int = 0, timeout: float = 60.0):
    """Open a URL, requesting a byte range when resuming."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(request, timeout=timeout)


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    retries: int = 3,
    timeout: float = 60.0,
    progress: ProgressFn | None = None,
    overwrite: bool = False,
) -> Path:
    """Fetch `url` to `dest`, resuming a partial download and verifying if asked.

    A `file://` URL is copied rather than streamed, so a local mirror costs
    nothing. Returns `dest`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        if sha256 is None or _sha256_file(dest) == sha256:
            log.debug("already present, skipping: %s", dest.name)
            return dest
        log.warning("%s failed its checksum; re-fetching", dest.name)
        dest.unlink()

    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        src = Path(parsed.path if parsed.scheme == "file" else url).expanduser()
        if not src.is_file():
            raise FetchError(f"local source does not exist: {src}")
        shutil.copyfile(src, dest)
    else:
        _download_http(url, dest, retries=retries, timeout=timeout, progress=progress)

    if sha256 is not None:
        got = _sha256_file(dest)
        if got != sha256:
            dest.unlink(missing_ok=True)
            raise FetchError(f"{url}: sha256 mismatch — expected {sha256[:16]}…, got {got[:16]}…")
    return dest


def _download_http(url: str, dest: Path, *, retries: int, timeout: float, progress: ProgressFn | None) -> None:
    """Stream a URL to disk, resuming from a `.part` file across retries."""
    partial = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with _open(url, offset=offset, timeout=timeout) as response:
                # A server that ignores Range replies 200; restart rather than
                # append, or the file would be silently corrupted.
                resuming = offset > 0 and response.status == 206
                if offset and not resuming:
                    offset = 0
                    partial.unlink(missing_ok=True)

                declared = response.headers.get("Content-Length")
                total = (int(declared) + offset) if declared is not None else None
                done = offset
                with partial.open("ab" if resuming else "wb") as handle:
                    while chunk := response.read(CHUNK):
                        handle.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(dest.name, done, total)
            partial.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            if attempt == retries:
                break
            backoff = 2.0 ** (attempt - 1)
            log.warning("%s failed (attempt %d/%d): %s — retrying in %.0fs", url, attempt, retries, exc, backoff)
            time.sleep(backoff)

    raise FetchError(f"{url}: giving up after {retries} attempts ({last})") from last


def extract(archive: Path, dest: Path, *, remove: bool = True) -> list[Path]:
    """Unpack a zip or tar into `dest`, refusing entries that escape it.

    Returns the extracted paths. Traversal-safe: an entry resolving outside `dest`
    aborts the extraction rather than being sanitised silently.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            names = [n for n in bundle.namelist() if not n.endswith("/") and not n.startswith("__MACOSX")]
            for name in names:
                _guard(dest, dest / name, archive)
            for name in names:
                bundle.extract(name, dest)
                written.append(dest / name)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            members = [m for m in bundle.getmembers() if m.isfile()]
            for member in members:
                _guard(dest, dest / member.name, archive)
            # `filter="data"` also strips device nodes, setuid bits and absolute paths.
            bundle.extractall(dest, members=members, filter="data")
            written.extend(dest / m.name for m in members)
    else:
        raise FetchError(f"{archive.name}: not a zip or tar archive")

    if remove:
        archive.unlink(missing_ok=True)
    return written


def _guard(root: Path, target: Path, archive: Path) -> None:
    """Refuse an archive entry that would land outside the extraction root."""
    resolved_root = root.resolve()
    if not str(Path(os.path.normpath(target)).resolve()).startswith(str(resolved_root)):
        raise FetchError(f"{archive.name}: entry escapes the extraction directory ({target})")


def write_notice(root: Path) -> Path:
    """Drop the licence and attribution notice beside the data, and log it once.

    The data outlives the shell it was fetched in. Writing the terms next to the
    bytes means a directory found later still says what it is and what may be
    done with it.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / NOTICE_FILENAME
    first_time = not path.exists()
    path.write_text(DATA_NOTICE + "\n" + ATTRIBUTION + "\n", encoding="utf-8")
    if first_time:
        log.warning(
            "Fetching third-party, non-commercially licensed data. Terms written to %s\n%s",
            path,
            DATA_NOTICE,
        )
    return path


def acquire(
    source: Source,
    dest: Path | None = None,
    *,
    progress: ProgressFn | None = None,
    overwrite: bool = False,
    retries: int = 3,
) -> FetchResult:
    """Fetch everything `source` names into `dest`, unpacking archives.

    For a raw source `dest` defaults to the cache's `raw/`; for a prebuilt corpus
    it defaults to the cache's `tensors/`. Archive members are normalised so each
    gesture's tables land in their own directory — upstream reuses table names
    across gestures, so flattening them would have `hitting/joint_angles.csv`
    overwrite `pitching/joint_angles.csv`.
    """
    from .config import tensors_dir

    root = Path(dest) if dest is not None else (tensors_dir() if source.prebuilt else raw_dir())
    root.mkdir(parents=True, exist_ok=True)
    write_notice(root)

    log.info("acquiring from %s into %s", source.describe(), root)
    files: list[Path] = []

    for asset in source.assets():
        target = root / asset.relpath
        if not asset.archive:
            files.append(
                download(
                    asset.url, target, sha256=asset.sha256, retries=retries, progress=progress, overwrite=overwrite
                )
            )
            continue

        download(asset.url, target, sha256=asset.sha256, retries=retries, progress=progress, overwrite=overwrite)
        produced = extract(target, target.parent, remove=True)
        files.extend(_normalise(produced, target))

    log.info("acquired %d files into %s", len(files), root)
    return FetchResult(root=root, files=files, source=source.describe(), prebuilt=source.prebuilt)


def _normalise(produced: list[Path], archive: Path) -> list[Path]:
    """Rename an extracted member to the archive's own stem when they disagree.

    Upstream archives happen to hold a single CSV named after the table, but that
    is an observed convention rather than a promise, so it is checked and
    corrected instead of assumed.
    """
    if len(produced) != 1:
        return produced
    got = produced[0]
    want = archive.parent / (archive.stem + got.suffix)
    if got == want:
        return produced
    log.info("%s contained %r; normalising to %r", archive.name, got.name, want.name)
    got.replace(want)
    # Drop a now-empty directory the archive may have introduced.
    if got.parent != archive.parent and not any(got.parent.iterdir()):
        got.parent.rmdir()
    return [want]
