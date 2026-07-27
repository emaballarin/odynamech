"""The three ways to obtain the data, as explicit, inspectable source objects.

Exactly one of them carries a default URL — `DrivelineRelease`, pointing at the
upstream release assets. `RawMirror` and `PackagedCorpus` both *require* a URL
from the caller: there is no fallback, no environment-variable default and no
"helpful" guess, so a mirror is never contacted unless it was named.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from urllib.parse import urlsplit

from .config import DRIVELINE_RAW_BASE
from .config import DRIVELINE_RELEASE_BASE
from .config import DRIVELINE_RELEASE_TAG
from .config import GESTURE_ASSETS
from .config import GESTURE_SCALARS
from .config import GESTURES


@dataclass(frozen=True, slots=True)
class Asset:
    """One file to fetch: where it comes from, and where it lands in the cache."""

    url: str
    relpath: str
    archive: bool = False
    #: Expected `sha256` of the downloaded bytes, when the caller knows it.
    sha256: str | None = None


class Source:
    """Base class for an acquisition path."""

    #: Whether this source yields a prebuilt corpus rather than raw upstream tables.
    prebuilt: bool = False

    def assets(self) -> Iterator[Asset]:
        """The files this source provides."""
        raise NotImplementedError

    def describe(self) -> str:
        """One line naming this source, for logs and provenance records."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DrivelineRelease(Source):
    """Raw upstream tables from the original Driveline release assets.

    The **only** source with a default URL. Override `base_url` / `raw_base_url`
    to point at an identically-laid-out mirror; pass a bare `RawMirror` instead
    if the layout differs.
    """

    base_url: str = DRIVELINE_RELEASE_BASE
    raw_base_url: str = DRIVELINE_RAW_BASE
    tag: str = DRIVELINE_RELEASE_TAG
    gestures: tuple[str, ...] = GESTURES

    def __post_init__(self) -> None:
        """Normalise the two overridable bases; the defaults already comply."""
        object.__setattr__(self, "base_url", _strip_trailing_slash(self.base_url, "DrivelineRelease base_url"))
        object.__setattr__(
            self, "raw_base_url", _strip_trailing_slash(self.raw_base_url, "DrivelineRelease raw_base_url")
        )

    def assets(self) -> Iterator[Asset]:
        """Per-gesture signal archives, plus the scalar tables from the git tree."""
        for gesture in self.gestures:
            for table in GESTURE_ASSETS[gesture]:
                yield Asset(
                    url=f"{self.base_url}/{gesture}_{table}.zip",
                    relpath=f"{gesture}/{table}.zip",
                    archive=True,
                )
            module = GESTURE_SCALARS[gesture]
            yield Asset(url=f"{self.raw_base_url}/{module}/data/metadata.csv", relpath=f"{gesture}/metadata.csv")
            yield Asset(
                url=f"{self.raw_base_url}/{module}/data/poi/poi_metrics.csv",
                relpath=f"{gesture}/poi_metrics.csv",
            )

    def describe(self) -> str:
        """Identify this source."""
        return f"DrivelineRelease(tag={self.tag!r}, base_url={self.base_url!r})"


@dataclass(frozen=True, slots=True)
class RawMirror(Source):
    """Raw upstream tables from a caller-supplied location. No default URL.

    `base_url` is required and is treated as a directory laid out exactly like the
    upstream release: `{base_url}/{gesture}_{table}.zip`. If the scalar tables live
    somewhere else, give `scalar_base_url`; by default they are looked for under
    `{base_url}` too, so a single mirrored directory is enough.

    Point this at an institutional mirror, a lab NAS, or a `file://` path holding
    a previously-fetched copy.
    """

    base_url: str
    scalar_base_url: str | None = None
    gestures: tuple[str, ...] = GESTURES
    #: Optional `sha256` per relpath, so a mirror can be checked as well as trusted.
    checksums: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an empty URL loudly rather than silently reaching upstream."""
        if not self.base_url:
            raise ValueError(
                "RawMirror requires an explicit base_url; it has no default. "
                "Use DrivelineRelease() if you want the original upstream source."
            )
        object.__setattr__(self, "base_url", _strip_trailing_slash(self.base_url, "RawMirror base_url"))
        if self.scalar_base_url:
            object.__setattr__(
                self,
                "scalar_base_url",
                _strip_trailing_slash(self.scalar_base_url, "RawMirror scalar_base_url"),
            )

    def assets(self) -> Iterator[Asset]:
        """Signal archives and scalar tables, all beneath the supplied base."""
        scalars = self.scalar_base_url or self.base_url
        for gesture in self.gestures:
            for table in GESTURE_ASSETS[gesture]:
                rel = f"{gesture}/{table}.zip"
                yield Asset(
                    url=f"{self.base_url}/{gesture}_{table}.zip",
                    relpath=rel,
                    archive=True,
                    sha256=self.checksums.get(rel),
                )
            for name in ("metadata.csv", "poi_metrics.csv"):
                rel = f"{gesture}/{name}"
                yield Asset(
                    url=f"{scalars}/{gesture}_{name}",
                    relpath=rel,
                    sha256=self.checksums.get(rel),
                )

    def describe(self) -> str:
        """Identify this source."""
        return f"RawMirror(base_url={self.base_url!r})"


@dataclass(frozen=True, slots=True)
class PackagedCorpus(Source):
    """An already-repackaged corpus from a caller-supplied URL. No default URL.

    `url` is required and must point at a tarball (`.tar`, `.tar.gz`, `.tar.zst`
    …) or at a bare `.safetensors` pack. A tarball is expected to contain the
    pack, and may also carry the per-gesture intermediates, the build report and
    the licence notice; everything it contains is extracted as-is.

    Nothing is built from source on this path — the corpus arrives finished — so
    it is the fast route, and the one where `sha256` matters most.
    """

    url: str
    sha256: str | None = None
    prebuilt: bool = True

    def __post_init__(self) -> None:
        """Reject an empty URL loudly; there is deliberately nothing to fall back to."""
        if not self.url:
            raise ValueError(
                "PackagedCorpus requires an explicit url; it has no default. "
                "There is no canonical repackaged corpus to fall back to."
            )

    def assets(self) -> Iterator[Asset]:
        """The single archive or pack file."""
        name = self.url.rstrip("/").rsplit("/", 1)[-1] or "corpus.tar.gz"
        yield Asset(url=self.url, relpath=name, archive=_is_archive(name), sha256=self.sha256)

    def describe(self) -> str:
        """Identify this source."""
        return f"PackagedCorpus(url={self.url!r})"


_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
)


def _is_archive(name: str) -> bool:
    """Whether a filename looks like an archive this package will unpack."""
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def _strip_trailing_slash(url: str, field: str) -> str:
    """Drop trailing slashes from a base that will be joined with `/`.

    A doubled separator is invisible on hosts that normalise paths, but an
    S3-style host treats `//` as a literal key separator and answers 404, so
    the mistake surfaces as a missing file rather than as a malformed URL.

    Stripping can eat the whole locator — `https://` becomes `https:` and `/`
    becomes empty — so what remains is parsed rather than pattern-matched, and a
    base left with neither host nor path is refused instead of being handed on
    as something that merely looks usable.
    """
    stripped = url.rstrip("/")
    parts = urlsplit(stripped)
    if not parts.netloc and not parts.path:
        raise ValueError(f"{field} {url!r} has no host or path to fetch from")
    return stripped
