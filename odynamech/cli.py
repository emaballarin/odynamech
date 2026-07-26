"""Command-line entry point: `odynamech fetch | build | verify | info | licence`."""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import cache_root
from .config import DATA_NOTICE
from .config import raw_dir
from .config import tensors_dir
from .sources import DrivelineRelease
from .sources import PackagedCorpus
from .sources import RawMirror
from .sources import Source


def _progress(name: str, done: int, total: int | None) -> None:
    """Single-line download progress on a terminal; silent when piped."""
    if not sys.stderr.isatty():
        return
    if total:
        pct = 100.0 * done / total
        sys.stderr.write(f"\r  {name:<28s} {done / 1e6:8.1f} / {total / 1e6:8.1f} MB  {pct:5.1f}%")
    else:
        sys.stderr.write(f"\r  {name:<28s} {done / 1e6:8.1f} MB")
    if total and done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _source_from(args: argparse.Namespace) -> Source:
    """Build the source the flags describe, refusing ambiguous combinations."""
    chosen = [bool(getattr(args, "raw_url", None)), bool(getattr(args, "corpus_url", None))]
    if all(chosen):
        raise SystemExit("--raw-url and --corpus-url are mutually exclusive: pick one source")
    if getattr(args, "corpus_url", None):
        return PackagedCorpus(url=args.corpus_url, sha256=getattr(args, "sha256", None))
    if getattr(args, "raw_url", None):
        return RawMirror(base_url=args.raw_url)
    return DrivelineRelease()


def _add_source_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the three acquisition paths. Only the default one has a URL baked in."""
    parser.add_argument(
        "--raw-url",
        metavar="URL",
        help="fetch the RAW upstream tables from this location instead of Driveline (no default)",
    )
    parser.add_argument(
        "--corpus-url",
        metavar="URL",
        help="fetch an already-repackaged corpus (tarball or .safetensors) from this location (no default)",
    )
    parser.add_argument("--sha256", metavar="HEX", help="expected sha256 of a --corpus-url download")
    parser.add_argument("--root", type=Path, help=f"cache root (default: {cache_root()})")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(
        prog="odynamech",
        description=(
            "Open biomechanics data as a dynamical-systems and machine-learning testbed. "
            "Ships no data: it fetches from a source you nominate."
        ),
    )
    parser.add_argument("--version", action="version", version=f"odynamech {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download data without building")
    _add_source_flags(p_fetch)
    p_fetch.add_argument("--overwrite", action="store_true", help="re-download files already present")

    p_build = sub.add_parser("build", help="build the corpus from already-fetched raw tables")
    p_build.add_argument("--root", type=Path, help=f"cache root (default: {cache_root()})")
    p_build.add_argument("--no-pack", action="store_true", help="stop after the per-gesture intermediates")

    p_load = sub.add_parser("load", help="fetch and build as needed, then report what was produced")
    _add_source_flags(p_load)
    p_load.add_argument("--rebuild", action="store_true", help="rebuild even if a corpus is cached")

    p_verify = sub.add_parser("verify", help="run the structural gates against a built corpus")
    p_verify.add_argument("--root", type=Path, help=f"cache root (default: {cache_root()})")
    p_verify.add_argument("--source", type=Path, help="pack file or intermediates dir (default: the cache)")

    p_info = sub.add_parser("info", help="show cache locations and what is present")
    p_info.add_argument("--root", type=Path, help=f"cache root (default: {cache_root()})")

    sub.add_parser("licence", help="print the third-party data licence notice")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "licence":
        print(DATA_NOTICE)
        return 0

    from . import api

    root = getattr(args, "root", None)

    if args.command == "fetch":
        result = api.fetch(_source_from(args), progress=_progress, overwrite=args.overwrite)
        print(f"\nfetched {len(result.files)} files from {result.source} into {result.root}")
        return 0

    if args.command == "build":
        path = api.build(raw_dir(root), tensors_dir(root), do_pack=not args.no_pack)
        print(f"built {path}")
        return 0

    if args.command == "load":
        obp = api.load(_source_from(args), root=root, rebuild=args.rebuild, progress=_progress)
        print(f"\n{obp}")
        for gesture in obp.gestures:
            view = obp[gesture]
            print(f"  {gesture:9s} {view.shape}  anchor={view.anchor_event}  rate={view.rate_hz:g} Hz")
        return 0

    if args.command == "verify":
        source = args.source or api.corpus_path(root)
        if source is None:
            print(f"no corpus found under {tensors_dir(root)}; run `odynamech load` first", file=sys.stderr)
            return 2
        from .verify import verify_corpus

        report = verify_corpus(source, raw=raw_dir(root), tensors=tensors_dir(root))
        print(f"\n{report.summary()}")
        for failure in report.failed:
            print(f"  FAILED: {failure}", file=sys.stderr)
        return 0 if report.ok else 1

    if args.command == "info":
        print(f"cache root : {cache_root() if root is None else root}")
        print(f"raw        : {raw_dir(root)}  ({'present' if raw_dir(root).is_dir() else 'absent'})")
        print(f"tensors    : {tensors_dir(root)}  ({'present' if tensors_dir(root).is_dir() else 'absent'})")
        found = api.corpus_path(root)
        print(f"corpus     : {found if found else 'not built'}")
        if found:
            print(f"             {found.stat().st_size / 1e6:.1f} MB")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
