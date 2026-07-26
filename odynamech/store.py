"""Safetensors I/O: opens either the single-file pack or the intermediate directory."""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from .schema import channel_hash

FORMAT_VERSION = "1"


class ChannelHashMismatch(RuntimeError):
    """The stored channel hash disagrees with the schema table it is meant to describe."""


def bytes_to_tensor(payload: bytes) -> torch.Tensor:
    """Wrap raw bytes as a 1-D uint8 tensor for storage inside a pack."""
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Recover the raw bytes a `uint8` tensor carries."""
    return bytes(tensor.contiguous().numpy().tobytes())


def read_parquet_bytes(payload: bytes) -> pd.DataFrame:
    """Reconstruct a DataFrame from stored Parquet bytes."""
    return pd.read_parquet(io.BytesIO(payload))


def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    """Reconstruct the channel schema from stored CSV bytes."""
    frame = pd.read_csv(io.BytesIO(payload), keep_default_na=False, na_values=[])
    for col in ("subtype", "axis", "ref_segment", "unit", "site"):
        if col in frame.columns:
            frame[col] = frame[col].astype("str")
    if "constant_zero" in frame.columns:
        frame["constant_zero"] = frame["constant_zero"].astype(str).str.lower().eq("true")
    return frame


class Store:
    """A read handle over one corpus, whichever on-disk shape it takes."""

    def __init__(self, source: str | Path, mmap: bool = True) -> None:
        """`source` is a single-file `.safetensors` pack or the directory of intermediates."""
        self.source = Path(source)
        self.mmap = mmap
        self.is_pack = self.source.is_file()
        self._files: dict[str, Path] = {}
        self._meta: dict[str, dict[str, str]] = {}
        self._cache: dict[tuple[str, str], torch.Tensor] = {}
        self.read_path = "mmap+slice" if mmap else "eager"

        if self.is_pack:
            self._files["__pack__"] = self.source
            with safe_open(self.source, framework="pt") as handle:
                meta = handle.metadata() or {}
                self._keys = set(handle.keys())
            self._pack_meta = meta
            self.gestures: tuple[str, ...] = tuple(json.loads(meta.get("gestures", "[]")))
        else:
            found = sorted(self.source.glob("*_ragged.safetensors"))
            self.gestures = tuple(p.name.removesuffix("_ragged.safetensors") for p in found)
            self._pack_meta = {"format_version": FORMAT_VERSION}
            for gesture in self.gestures:
                for view in ("ragged", "aligned"):
                    path = self.source / f"{gesture}_{view}.safetensors"
                    self._files[f"{gesture}/{view}"] = path
                    with safe_open(path, framework="pt") as handle:
                        self._meta[f"{gesture}/{view}"] = handle.metadata() or {}
        if not self.gestures:
            raise FileNotFoundError(f"no gesture stores found under {self.source}")

    # ------------------------------------------------------------------ metadata

    def meta(self, gesture: str, view: str) -> dict[str, str]:
        """`__metadata__` for one (gesture, view) store."""
        if self.is_pack:
            raw = self._pack_meta.get(f"{gesture}/{view}")
            return json.loads(raw) if raw else {}
        return self._meta[f"{gesture}/{view}"]

    @property
    def pack_meta(self) -> dict[str, str]:
        """Top-level metadata: format version, gestures, harmonisation map."""
        return self._pack_meta

    # ------------------------------------------------------------------- tensors

    def _locate(self, gesture: str, view: str, name: str) -> tuple[Path, str]:
        """Resolve a logical tensor to its file and in-file key."""
        if self.is_pack:
            return self.source, f"{gesture}/{view}/{name}"
        return self._files[f"{gesture}/{view}"], name

    def has(self, gesture: str, view: str, name: str) -> bool:
        """Whether a tensor is present."""
        path, key = self._locate(gesture, view, name)
        if self.is_pack:
            return key in self._keys
        with safe_open(path, framework="pt") as handle:
            return key in set(handle.keys())

    def tensor(self, gesture: str, view: str, name: str) -> torch.Tensor:
        """Whole tensor, cached. Uses the lazily-mapped path when `mmap` is set."""
        cache_key = (f"{gesture}/{view}", name)
        if cache_key not in self._cache:
            path, key = self._locate(gesture, view, name)
            with safe_open(path, framework="pt") as handle:
                self._cache[cache_key] = handle.get_tensor(key)
        return self._cache[cache_key]

    def rows(self, gesture: str, view: str, name: str, rows: np.ndarray) -> torch.Tensor:
        """Gather rows along the leading axis.

        A contiguous leading-axis slice is a genuine partial read — probed at
        2.3 MiB versus 149 MiB for a full materialisation — so it is worth taking
        when the selection allows. Any other pattern falls back to reading the
        whole tensor, which is affordable here because the corpus fits in RAM.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if self.mmap and len(rows) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
            path, key = self._locate(gesture, view, name)
            with safe_open(path, framework="pt") as handle:
                return handle.get_slice(key)[int(rows[0]) : int(rows[-1]) + 1]
        return self.tensor(gesture, view, name)[torch.from_numpy(rows)]

    # -------------------------------------------------------------------- tables

    def channels(self, gesture: str) -> pd.DataFrame:
        """The channel schema for one gesture, with its hash checked."""
        if self.is_pack:
            with safe_open(self.source, framework="pt") as handle:
                payload = tensor_to_bytes(handle.get_tensor(f"{gesture}/channels_csv"))
            frame = read_csv_bytes(payload)
        else:
            frame = read_csv_bytes((self.source / f"{gesture}_channels.csv").read_bytes())
        stored = self.meta(gesture, "ragged").get("channel_hash", "")
        got = channel_hash(frame["name"].tolist())
        if stored and got != stored:
            raise ChannelHashMismatch(
                f"{gesture}: channel schema hashes to {got[:16]} but the store records {stored[:16]}. "
                f"The channel axis and the schema have drifted apart; rebuild the corpus."
            )
        return frame

    def index(self, gesture: str) -> pd.DataFrame:
        """The per-trial index for one gesture."""
        if self.is_pack:
            with safe_open(self.source, framework="pt") as handle:
                payload = tensor_to_bytes(handle.get_tensor(f"{gesture}/index_parquet"))
            return read_parquet_bytes(payload)
        return pd.read_parquet(self.source / f"{gesture}_index.parquet")

    def json_meta(self, gesture: str, view: str, key: str, default: Any = None) -> Any:
        """A JSON-encoded metadata field."""
        raw = self.meta(gesture, view).get(key)
        return default if raw is None else json.loads(raw)
