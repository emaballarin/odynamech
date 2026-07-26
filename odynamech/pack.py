"""Pack every intermediate into one portable single-file corpus, and verify it."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import PACK_NAME
from .harmonise import harmonise
from .store import bytes_to_tensor
from .store import FORMAT_VERSION
from .store import read_csv_bytes
from .store import Store
from .store import tensor_to_bytes


class PackError(RuntimeError):
    """The single-file pack failed its round-trip check and is not trustworthy."""


def _sha256(tensor: torch.Tensor) -> str:
    """Content hash of a tensor's raw bytes."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def pack(tensors_dir: Path, out: Path | None = None) -> Path:
    """Fold both gestures, all views and both tables into one file.

    Numeric tensors are namespaced (`pitching/aligned/values`). The Parquet index
    and the schema CSV are stored as 1-D `uint8` byte tensors rather than in
    `__metadata__`, which keeps the pack genuinely portable regardless of table size.
    """
    tensors_dir = Path(tensors_dir)
    out = out or tensors_dir / PACK_NAME
    src = Store(tensors_dir, mmap=False)

    payload: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {"format_version": FORMAT_VERSION, "gestures": json.dumps(list(src.gestures))}
    digests: dict[str, str] = {}
    channels: dict[str, pd.DataFrame] = {}

    for gesture in src.gestures:
        for view in ("ragged", "aligned"):
            path = tensors_dir / f"{gesture}_{view}.safetensors"
            with safe_open(path, framework="pt") as handle:
                view_meta = handle.metadata() or {}
                for name in list(handle.keys()):
                    tensor = handle.get_tensor(name)
                    key = f"{gesture}/{view}/{name}"
                    payload[key] = tensor
                    digests[key] = _sha256(tensor)
            meta[f"{gesture}/{view}"] = json.dumps(view_meta)

        csv_bytes = (tensors_dir / f"{gesture}_channels.csv").read_bytes()
        pq_bytes = (tensors_dir / f"{gesture}_index.parquet").read_bytes()
        payload[f"{gesture}/channels_csv"] = bytes_to_tensor(csv_bytes)
        payload[f"{gesture}/index_parquet"] = bytes_to_tensor(pq_bytes)
        digests[f"{gesture}/channels_csv"] = hashlib.sha256(csv_bytes).hexdigest()
        digests[f"{gesture}/index_parquet"] = hashlib.sha256(pq_bytes).hexdigest()
        channels[gesture] = read_csv_bytes(csv_bytes)

    # The harmonisation map is small, so it lives in metadata as decision 12 asks.
    if len(channels) > 1:
        common, excluded = harmonise(channels)
        meta["harmonised"] = common.to_json(orient="records")
        meta["harmonised_excluded_sites"] = json.dumps(excluded)
        meta["harmonised_n"] = str(len(common))

    meta["tensor_sha256"] = json.dumps(digests)
    save_file(payload, str(out), metadata=meta)
    verify_pack(out, tensors_dir)
    return out


def verify_pack(pack_path: Path, tensors_dir: Path) -> None:
    """Pre-mortem gate 8: the pack must reproduce the intermediates exactly."""
    packed = Store(pack_path, mmap=False)
    loose = Store(tensors_dir, mmap=False)

    if set(packed.gestures) != set(loose.gestures):
        raise PackError(f"pack holds gestures {packed.gestures}, intermediates hold {loose.gestures}")

    digests = json.loads(packed.pack_meta.get("tensor_sha256", "{}"))
    with safe_open(pack_path, framework="pt") as handle:
        keys = list(handle.keys())
        for key in keys:
            got = _sha256(handle.get_tensor(key))
            if key in digests and got != digests[key]:
                raise PackError(f"{key}: sha256 {got[:16]} does not match the recorded {digests[key][:16]}")

    for gesture in loose.gestures:
        want_index = loose.index(gesture)
        got_index = packed.index(gesture)
        try:
            pd.testing.assert_frame_equal(got_index, want_index)
        except AssertionError as exc:
            raise PackError(f"{gesture}: index DataFrame did not survive the byte-tensor round-trip\n{exc}") from exc

        want_ch = loose.channels(gesture)
        got_ch = packed.channels(gesture)
        try:
            pd.testing.assert_frame_equal(got_ch, want_ch)
        except AssertionError as exc:
            raise PackError(f"{gesture}: channel schema did not survive the round-trip\n{exc}") from exc

        for view in ("ragged", "aligned"):
            with safe_open(tensors_dir / f"{gesture}_{view}.safetensors", framework="pt") as handle:
                for name in list(handle.keys()):
                    a = handle.get_tensor(name)
                    b = packed.tensor(gesture, view, name)
                    if not _tensor_equal(a, b):
                        raise PackError(f"{gesture}/{view}/{name}: tensor differs between pack and intermediate")


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact equality, treating NaN as equal to NaN in the same position."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if not a.is_floating_point():
        return bool(torch.equal(a, b))
    na, nb = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(na, nb) and torch.equal(a[~na], b[~nb]))


def unpack_tables(pack_path: Path, out_dir: Path) -> None:
    """Write the Parquet index and schema CSV back out of a pack, byte-for-byte."""
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(pack_path, mmap=False)
    with safe_open(pack_path, framework="pt") as handle:
        for gesture in store.gestures:
            (out_dir / f"{gesture}_channels.csv").write_bytes(
                tensor_to_bytes(handle.get_tensor(f"{gesture}/channels_csv"))
            )
            (out_dir / f"{gesture}_index.parquet").write_bytes(
                tensor_to_bytes(handle.get_tensor(f"{gesture}/index_parquet"))
            )
