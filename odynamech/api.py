"""The short path: name a source, get a corpus.

`load()` is the one entry point most users need. It resolves a `Source` to a
built corpus on disk — fetching and building only what is missing — and returns
an open handle.
"""

import logging
from pathlib import Path

from .build import build_gesture
from .build import write_intermediates
from .config import cache_root
from .config import GESTURES
from .config import PACK_NAME
from .config import raw_dir
from .config import tensors_dir
from .fetch import acquire
from .fetch import FetchResult
from .fetch import ProgressFn
from .fetch import write_notice
from .pack import pack
from .sources import DrivelineRelease
from .sources import Source
from .view import OBP

log = logging.getLogger(__name__)


def _find_pack(where: Path) -> Path | None:
    """Locate a single-file pack in a directory, whatever it was named upstream."""
    exact = where / PACK_NAME
    if exact.is_file():
        return exact
    candidates = sorted(
        p for p in where.glob("*.safetensors") if not p.name.endswith(("_ragged.safetensors", "_aligned.safetensors"))
    )
    return candidates[0] if candidates else None


def corpus_path(root: Path | None = None) -> Path | None:
    """The built corpus in `root`'s cache, or `None` if there is not one yet."""
    return _find_pack(tensors_dir(root))


def fetch(
    source: Source | None = None,
    *,
    dest: Path | None = None,
    progress: ProgressFn | None = None,
    overwrite: bool = False,
) -> FetchResult:
    """Acquire data from `source`, defaulting to the original Driveline release.

    `DrivelineRelease` is the only source with a default URL; `RawMirror` and
    `PackagedCorpus` must both be constructed with an explicit one.
    """
    return acquire(source or DrivelineRelease(), dest, progress=progress, overwrite=overwrite)


def build(
    raw: Path | None = None,
    out: Path | None = None,
    *,
    gestures: tuple[str, ...] = GESTURES,
    do_pack: bool = True,
) -> Path:
    """Convert fetched CSVs into the tensor stores, then the single-file pack.

    Returns the path to the pack (or to the intermediates directory when
    `do_pack` is false). Aborts on any failed invariant rather than writing a
    corpus that is quietly wrong.
    """
    raw = Path(raw) if raw is not None else raw_dir()
    out = Path(out) if out is not None else tensors_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_notice(out)

    for gesture in gestures:
        log.info("building %s", gesture)
        result = build_gesture(raw, gesture)
        write_intermediates(result, out)
        for line in result.report:
            log.debug("%s", line)

    if not do_pack:
        return out
    path = pack(out)
    log.info("packed %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load(
    source: Source | None = None,
    *,
    root: Path | None = None,
    mmap: bool = True,
    download: bool = True,
    rebuild: bool = False,
    progress: ProgressFn | None = None,
) -> OBP:
    """Open the corpus, acquiring and building it first if that is needed.

    Resolution order:

    1. a pack already in the cache is opened as-is, unless `rebuild`;
    2. a prebuilt `PackagedCorpus` source is fetched and opened, with no build;
    3. otherwise the raw tables are fetched if absent and the corpus is built.

    Set `download=False` to forbid any network access and fail instead.
    """
    root = Path(root) if root is not None else cache_root()
    tensors = tensors_dir(root)

    existing = _find_pack(tensors)
    if existing is not None and not rebuild:
        log.info("using cached corpus at %s", existing)
        return OBP(existing, mmap=mmap)

    source = source or DrivelineRelease()

    if source.prebuilt:
        if not download:
            raise FileNotFoundError(
                f"no corpus in {tensors} and download=False; nothing to open. "
                f"Call load(..., download=True) or point `root` at an existing cache."
            )
        result = acquire(source, tensors, progress=progress)
        found = _find_pack(tensors)
        if found is None:
            raise FileNotFoundError(
                f"{source.describe()} produced no single-file pack in {tensors}. "
                f"Extracted: {[p.name for p in result.files][:8]}"
            )
        log.info("using fetched corpus at %s", found)
        return OBP(found, mmap=mmap)

    raw = raw_dir(root)
    if not _raw_complete(raw):
        if not download:
            raise FileNotFoundError(
                f"raw tables are missing under {raw} and download=False. "
                f"Run fetch() first, or point `root` at a populated cache."
            )
        acquire(source, raw, progress=progress)

    return OBP(build(raw, tensors), mmap=mmap)


def _raw_complete(raw: Path, gestures: tuple[str, ...] = GESTURES) -> bool:
    """Whether every CSV the build needs is already on disk."""
    from .config import GESTURE_ASSETS

    for gesture in gestures:
        wanted = [f"{t}.csv" for t in GESTURE_ASSETS[gesture]] + ["metadata.csv", "poi_metrics.csv"]
        if any(not (raw / gesture / name).is_file() for name in wanted):
            return False
    return True
