"""Lazy, immutable views over a built corpus: `OBP`, `GestureView`, `HarmonisedView`.

Handedness note: the release's angle and moment conventions are already
handedness-adjusted, but `landmarks` are global-frame positions and force-plate
`y` is a lab-frame lateral axis, so left-handers are geometrically mirrored in
`y` for those groups. `mirror_lefties=True` negates `y` for the `landmark` and
`grf` groups only. It is **off by default** and never applied at build time.
"""

import json
import logging
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .harmonise import harmonise
from .harmonise import native_indices
from .harmonise import SEMANTIC_KEY
from .store import Store

log = logging.getLogger(__name__)

NanPolicy = str  # "keep" | "interpolate" | "drop"
NAN_POLICIES: tuple[str, ...] = ("keep", "interpolate", "drop")

MIRRORED_GROUPS: frozenset[str] = frozenset({"landmark", "grf"})
LEFT_CODES: frozenset[str] = frozenset({"L", "l", "Left", "left", "LHP"})


class EmptySelection(ValueError):
    """A selection collapsed an axis to zero length."""


# ---------------------------------------------------------------- NaN policies


def _interpolate_nans(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Linearly interpolate NaNs bracketed by known values, per trial and channel.

    Returns the filled array, the number of cells filled, and the number left
    untouched. Leading and trailing gaps have no bracketing values — a derivative
    edge is structurally undefined, not missing — so they are retained and counted.
    """
    out = values.copy()
    n_trials, T, _ = out.shape
    filled = 0
    grid = np.arange(T, dtype=np.float64)
    for i in range(n_trials):
        block = out[i]
        bad = np.isnan(block)
        cols = np.flatnonzero(bad.any(axis=0))
        for c in cols:
            mask = bad[:, c]
            known = np.flatnonzero(~mask)
            if len(known) < 2:
                continue
            lo, hi = known[0], known[-1]
            inner = np.flatnonzero(mask & (grid > lo) & (grid < hi))
            if not len(inner):
                continue
            block[inner, c] = np.interp(grid[inner], grid[known], block[known, c])
            filled += len(inner)
    remaining = int(np.isnan(out).sum())
    return out, filled, remaining


def _drop_cost(values: np.ndarray, athletes: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, dict[str, int]]:
    """Choose between dropping NaN-bearing athletes and dropping NaN-bearing channels.

    Whichever removes fewer cells wins; both costs are reported so the choice is
    inspectable rather than implicit.
    """
    bad = np.isnan(values)
    n, T, C = values.shape
    bad_trials = bad.any(axis=(1, 2))
    bad_athletes = np.unique(athletes[bad_trials])
    drop_rows = np.isin(athletes, bad_athletes)
    bad_chans = bad.any(axis=(0, 1))

    cost_athletes = int(drop_rows.sum()) * T * C
    cost_channels = n * T * int(bad_chans.sum())
    stats = {
        "athletes_removed": len(bad_athletes),
        "trials_removed": int(drop_rows.sum()),
        "channels_removed": int(bad_chans.sum()),
        "cost_by_athlete": cost_athletes,
        "cost_by_channel": cost_channels,
    }
    if cost_athletes <= cost_channels:
        return ~drop_rows, np.ones(C, dtype=bool), "athlete", stats
    return np.ones(n, dtype=bool), ~bad_chans, "channel", stats


# ------------------------------------------------------------------ gesture view


@dataclass(frozen=True, slots=True)
class GestureView:
    """Lazy, immutable selection over one gesture's aligned tensor store.

    Every selector returns a new view holding only index arrays; nothing is read
    until `.torch()`, `.numpy()` or iteration. Views compose in any order.
    """

    store: Store
    gesture: str
    _rows: np.ndarray
    _cols: np.ndarray
    _frames: np.ndarray
    _nan_policy: NanPolicy = "keep"
    _mirror: bool = False

    # ------------------------------------------------------------ construction

    @classmethod
    def open(cls, store: Store, gesture: str) -> GestureView:
        """A view over every trial, channel and frame of one gesture."""
        index = store.index(gesture)
        channels = store.channels(gesture)
        T = int(store.meta(gesture, "aligned")["T"])
        return cls(store, gesture, np.arange(len(index)), np.arange(len(channels)), np.arange(T))

    # ------------------------------------------------------------ introspection

    @property
    def channels(self) -> pd.DataFrame:
        """The current channel subset, carrying every schema column."""
        return self.store.channels(self.gesture).iloc[self._cols].reset_index(drop=True)

    @property
    def trials(self) -> pd.DataFrame:
        """The current trial subset, carrying every index column."""
        return self.store.index(self.gesture).iloc[self._rows].reset_index(drop=True)

    @property
    def shape(self) -> tuple[int, int, int]:
        """`(N, T, C)` of what `.torch()` would return."""
        return (len(self._rows), len(self._frames), len(self._cols))

    @property
    def rate_hz(self) -> float:
        """Marker sample rate."""
        return float(self.store.meta(self.gesture, "aligned")["rate_hz"])

    @property
    def pre_frames(self) -> int:
        """Frames stored before the anchor, in the full window."""
        return int(self.store.meta(self.gesture, "aligned")["pre_frames"])

    @property
    def anchor_event(self) -> str:
        """The event the time axis is zeroed on."""
        return self.store.meta(self.gesture, "aligned")["anchor_event"]

    @property
    def event_names(self) -> list[str]:
        """Event column names, in the order the `events` tensor stores them."""
        return self.store.json_meta(self.gesture, "aligned", "event_names", [])

    @property
    def time(self) -> torch.Tensor:
        """`(T,)` seconds relative to the anchor; negative before it."""
        return torch.from_numpy((self._frames - self.pre_frames) / self.rate_hz).to(torch.float32)

    def __repr__(self) -> str:
        """Compact summary of the current selection."""
        n, t, c = self.shape
        return f"<GestureView {self.gesture} N={n} T={t} C={c} nan={self._nan_policy}>"

    # -------------------------------------------------- trial / athlete selection

    def athlete(self, athlete_id: str | Sequence[str]) -> GestureView:
        """Restrict to one or more athletes."""
        want = {athlete_id} if isinstance(athlete_id, str) else set(athlete_id)
        index = self.store.index(self.gesture)
        keep = index["athlete_id"].isin(want).to_numpy()[self._rows]
        return replace(self, _rows=self._rows[keep])

    def trial(self, trial_id: str | Sequence[str]) -> GestureView:
        """Restrict to one or more trials."""
        want = {trial_id} if isinstance(trial_id, str) else set(trial_id)
        index = self.store.index(self.gesture)
        keep = index["trial_id"].isin(want).to_numpy()[self._rows]
        return replace(self, _rows=self._rows[keep])

    def where(self, expr: str | Callable[[pd.DataFrame], pd.Series]) -> GestureView:
        """Filter trials by a pandas query string or a predicate on the index frame."""
        frame = self.trials
        mask = frame.eval(expr).to_numpy() if isinstance(expr, str) else np.asarray(expr(frame))
        return replace(self, _rows=self._rows[mask.astype(bool)])

    # ------------------------------------------------------------ channel selection

    def select(
        self,
        *,
        site: str | Sequence[str] | None = None,
        group: str | Sequence[str] | None = None,
        axis: str | Sequence[str] | None = None,
        table: str | Sequence[str] | None = None,
        subtype: str | Sequence[str] | None = None,
        name: str | Sequence[str] | None = None,
        regex: str | None = None,
        drop_dead: bool = True,
        drop_grf_missing: bool = False,
    ) -> GestureView:
        """Restrict the channel axis. Every criterion is ANDed; `None` means no filter."""
        frame = self.store.channels(self.gesture).iloc[self._cols]
        mask = np.ones(len(frame), dtype=bool)

        for column, value in (
            ("site", site),
            ("group", group),
            ("axis", axis),
            ("table", table),
            ("subtype", subtype),
            ("name", name),
        ):
            if value is None:
                continue
            want = {value} if isinstance(value, str) else set(value)
            mask &= frame[column].isin(want).to_numpy()
        if regex is not None:
            mask &= frame["name"].str.contains(regex, regex=True).to_numpy()
        if drop_dead:
            mask &= ~frame["constant_zero"].to_numpy().astype(bool)

        view = replace(self, _cols=self._cols[mask])
        if drop_grf_missing:
            view = view.where(lambda f: f["has_grf"].astype(bool))
        return view

    # --------------------------------------------------------------- time selection

    def thin(self, step: int) -> GestureView:
        """Decimate in time by an integer stride, keeping the anchor frame on grid."""
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        offset = int(np.searchsorted(self._frames, self.pre_frames))
        offset = offset % step if offset < len(self._frames) else 0
        return replace(self, _frames=self._frames[offset::step])

    def window(self, start: float | None = None, stop: float | None = None) -> GestureView:
        """Crop to `[start, stop]` seconds relative to the anchor, inclusive."""
        secs = (self._frames - self.pre_frames) / self.rate_hz
        mask = np.ones(len(secs), dtype=bool)
        if start is not None:
            mask &= secs >= start - 1e-9
        if stop is not None:
            mask &= secs <= stop + 1e-9
        return replace(self, _frames=self._frames[mask])

    def frames(self, sl: slice) -> GestureView:
        """Crop the time axis by a slice over the current frame selection."""
        return replace(self, _frames=self._frames[sl])

    # ------------------------------------------------------------------ policies

    def nans(self, policy: NanPolicy) -> GestureView:
        """Choose how NaNs are treated at materialisation: keep, interpolate or drop.

        `keep` edits nothing. `interpolate` fills gaps that are bracketed by known
        values. `drop` removes whole athletes or whole channels, whichever removes
        fewer cells. Storage is never altered by any of them.

        Caveat: `drop` is the one selector whose effect cannot be known without
        reading the values, so `.shape` still reports the pre-drop shape and
        `.torch()` may return something smaller. `.complete()` has the same
        property but resolves it eagerly, so prefer it when the shape must be
        known up front; `dataset()` raises rather than let the labels desynchronise.
        """
        if policy not in NAN_POLICIES:
            raise ValueError(f"nan policy must be one of {NAN_POLICIES}, got {policy!r}")
        return replace(self, _nan_policy=policy)

    def mirror(self, on: bool = True) -> GestureView:
        """Toggle the opt-in left-hander mirroring of `y` for landmarks and GRF."""
        return replace(self, _mirror=on)

    def complete(self, drop_dead: bool = True) -> GestureView:
        """Restrict to a dense, NaN-free, unpadded rectangular block; raises if empty.

        Drops trials that are NaN on any selected channel, drops dead channels, and
        crops the time axis to the span over which every surviving trial is valid.
        """
        view = self.select(drop_dead=True) if drop_dead else self
        valid = view._raw_mask()
        if not len(view._rows):
            raise EmptySelection("complete(): no trials selected before completeness filtering")

        # Crop time to the frames every remaining trial marks valid.
        frame_ok = valid.all(axis=0)
        if not frame_ok.any():
            raise EmptySelection(
                "complete(): the time axis collapsed to zero — no frame is unpadded for every "
                f"selected trial (N={len(view._rows)}). Narrow the trial set or the window first."
            )
        first, last = int(np.argmax(frame_ok)), len(frame_ok) - int(np.argmax(frame_ok[::-1]))
        if not frame_ok[first:last].all():
            interior = np.flatnonzero(~frame_ok[first:last])
            raise EmptySelection(
                f"complete(): validity is not contiguous — {len(interior)} interior frames are "
                "padded for some trial. This should not happen for an edge-padded store."
            )
        view = replace(view, _frames=view._frames[first:last])

        values = view._gather()
        bad_trials = np.isnan(values).any(axis=(1, 2))
        kept = view._rows[~bad_trials]
        if not len(kept):
            raise EmptySelection(
                "complete(): the trial axis collapsed to zero — every selected trial is NaN on at "
                f"least one of the {len(view._cols)} selected channels. Deselect GRF or the "
                "force-derived energy channels, or use nans('interpolate')."
            )
        dropped = int(bad_trials.sum())
        if dropped:
            log.info(
                "complete(): dropped %d/%d trials with NaNs, cropped time to %d frames, %d channels",
                dropped,
                len(view._rows),
                len(view._frames),
                len(view._cols),
            )
        out = replace(view, _rows=kept)
        if min(out.shape) == 0:
            axis = ("N", "T", "C")[int(np.argmin(out.shape))]
            raise EmptySelection(f"complete(): axis {axis} collapsed to zero")
        return out

    # ------------------------------------------------------------- materialisation

    def _gather(self) -> np.ndarray:
        """`(N, T, C)` float32, before any NaN policy or mirroring."""
        block = self.store.rows(self.gesture, "aligned", "values", self._rows).numpy()
        return block[:, self._frames][:, :, self._cols]

    def _raw_mask(self) -> np.ndarray:
        """`(N, T)` bool validity mask over the current selection."""
        block = self.store.rows(self.gesture, "aligned", "valid", self._rows).numpy()
        return block[:, self._frames]

    def _apply_mirror(self, values: np.ndarray) -> np.ndarray:
        """Negate `y` for landmark and GRF channels on left-handed athletes."""
        channels = self.channels
        y_cols = np.flatnonzero(
            (channels["axis"].to_numpy() == "y") & channels["group"].isin(MIRRORED_GROUPS).to_numpy()
        )
        if not len(y_cols):
            return values
        lefty = self.trials["handedness"].isin(LEFT_CODES).to_numpy()
        if not lefty.any():
            return values
        out = values.copy()
        out[np.ix_(np.flatnonzero(lefty), np.arange(values.shape[1]), y_cols)] *= -1.0
        return out

    def torch(
        self,
        device: torch.device | str | None = None,
        mirror_lefties: bool | None = None,
        complete: bool = False,
    ) -> torch.Tensor:
        """`(N, T, C)` float32. `complete=True` is sugar for `.complete().torch()`."""
        if complete:
            return self.complete().torch(device=device, mirror_lefties=mirror_lefties)
        view = self if mirror_lefties is None else self.mirror(mirror_lefties)
        values = view._gather()

        if view._nan_policy == "interpolate":
            values, filled, remaining = _interpolate_nans(values, view._raw_mask())
            log.info("nans('interpolate'): filled %d cells, %d remain unbracketed", filled, remaining)
        elif view._nan_policy == "drop":
            keep_rows, keep_cols, how, stats = _drop_cost(values, view.trials["athlete_id"].to_numpy())
            log.info("nans('drop'): removed by %s; %s", how, stats)
            values = values[keep_rows][:, :, keep_cols]
            if min(values.shape) == 0:
                raise EmptySelection(f"nans('drop') emptied an axis: resulting shape {values.shape}")

        if view._mirror:
            values = view._apply_mirror(values)
        out = torch.from_numpy(np.ascontiguousarray(values))
        return out.to(device) if device is not None else out

    def mask(self) -> torch.Tensor:
        """`(N, T)` bool validity mask, aligned with `.torch()`."""
        return torch.from_numpy(self._raw_mask())

    def numpy(self) -> np.ndarray:
        """`(N, T, C)` float32 as a NumPy array."""
        return self.torch().numpy()

    def events(self) -> torch.Tensor:
        """`(N, E)` int32 event frames relative to the stored window; `-1` if absent."""
        return self.store.rows(self.gesture, "aligned", "events", self._rows)

    def value(self, trial_id: str, frame: float, channel: str) -> float:
        """A single measurement. `frame` is an int index or a float time in seconds."""
        view = self.trial(trial_id).select(name=channel, drop_dead=False)
        if view.shape[0] != 1:
            raise KeyError(f"trial {trial_id!r} is not in this view")
        if view.shape[2] != 1:
            raise KeyError(f"channel {channel!r} is not in this view")
        if isinstance(frame, float):
            target = round(frame * self.rate_hz) + self.pre_frames
        else:
            target = int(frame)
        pos = np.flatnonzero(view._frames == target)
        if not len(pos):
            raise IndexError(f"frame {frame!r} (absolute {target}) is outside this view's time axis")
        return float(view._gather()[0, int(pos[0]), 0])

    # ----------------------------------------------------- ragged / full-rate escapes

    def ragged(self, trial_id: str) -> torch.Tensor:
        """`(T_i, C)` unaligned, unpadded, full native length, current channel subset."""
        index = self.store.index(self.gesture)
        row = index.index[index["trial_id"] == trial_id]
        if not len(row):
            raise KeyError(f"unknown trial {trial_id!r}")
        offsets = self.store.tensor(self.gesture, "ragged", "offsets").numpy()
        i = int(row[0])
        block = self.store.tensor(self.gesture, "ragged", "values")[int(offsets[i]) : int(offsets[i + 1])]
        return block[:, torch.from_numpy(self._cols)]

    def fp_highrate(self, trial_id: str) -> torch.Tensor:
        """`(T_fp, 6)` untouched full-rate force plate; raises if the trial has no GRF."""
        fp_ids = self.store.json_meta(self.gesture, "ragged", "fp_trial_ids", [])
        if trial_id not in fp_ids:
            raise KeyError(f"trial {trial_id!r} has no force-plate data (has_grf is False)")
        i = fp_ids.index(trial_id)
        offsets = self.store.tensor(self.gesture, "ragged", "fp_offsets").numpy()
        return self.store.tensor(self.gesture, "ragged", "fp_values")[int(offsets[i]) : int(offsets[i + 1])]

    def fp_rate_hz(self, trial_id: str) -> float:
        """The native force-plate rate for one trial; not uniform across hitting."""
        index = self.store.index(self.gesture)
        row = index.loc[index["trial_id"] == trial_id]
        if not len(row):
            raise KeyError(f"unknown trial {trial_id!r}")
        return self.rate_hz * float(row["fp_ratio"].iloc[0])

    @property
    def fp_channels(self) -> list[str]:
        """Channel order of the full-rate force-plate store."""
        return self.store.json_meta(self.gesture, "ragged", "fp_channels", [])

    # -------------------------------------------------------------- torch interop

    def dataset(
        self, *, return_mask: bool = True, targets: Sequence[str] = (), complete: bool = False
    ) -> torch.utils.data.Dataset:
        """A `Dataset` yielding one trial per item."""
        from .torchio import TrialDataset

        return TrialDataset(self.complete() if complete else self, return_mask=return_mask, targets=targets)

    def dataloader(self, batch_size: int = 32, **kw) -> torch.utils.data.DataLoader:
        """A `DataLoader` over `.dataset()`; extra kwargs go to both."""
        from .torchio import make_dataloader

        return make_dataloader(self, batch_size=batch_size, **kw)

    def split_by_athlete(
        self, fractions: tuple[float, ...] = (0.8, 0.1, 0.1), seed: int = 0
    ) -> tuple[GestureView, ...]:
        """Athlete-disjoint splits; asserts empty athlete intersections."""
        from .torchio import split_by_athlete

        return split_by_athlete(self, fractions=fractions, seed=seed)

    # ------------------------------------------------------------------- sugar

    def __getitem__(self, key) -> GestureView:
        """`g["1031_2"]` selects a trial; `g[:, ::4, :]` thins time."""
        if isinstance(key, str):
            return self.trial(key)
        if isinstance(key, tuple) and len(key) == 3:
            rows, frames, cols = key
            view = self
            if not (isinstance(rows, slice) and rows == slice(None)):
                view = replace(view, _rows=view._rows[rows])
            if not (isinstance(frames, slice) and frames == slice(None)):
                view = replace(view, _frames=view._frames[frames])
            if not (isinstance(cols, slice) and cols == slice(None)):
                view = replace(view, _cols=view._cols[cols])
            return view
        raise TypeError("index with a trial id, or a 3-tuple over (trials, frames, channels)")


# --------------------------------------------------------------- harmonised view


@dataclass(frozen=True, slots=True)
class HarmonisedView:
    """Both gestures restricted to their shared semantic channel set; reversible."""

    _views: dict[str, GestureView]
    channels: pd.DataFrame
    excluded: dict[str, list[str]]

    @classmethod
    def build(cls, views: dict[str, GestureView], resolve_handedness: bool = False) -> HarmonisedView:
        """Intersect the gestures' channel schemas on the semantic tuple."""
        per = {g: v.store.channels(g) for g, v in views.items()}
        common, excluded = harmonise(per, resolve_handedness=resolve_handedness)
        restricted = {
            g: replace(v, _cols=np.asarray(native_indices(common, g), dtype=np.int64)) for g, v in views.items()
        }
        return cls(restricted, common, excluded)

    def __getitem__(self, gesture: str) -> GestureView:
        """A native `GestureView` pre-restricted to the common channels — maps back."""
        return self._views[gesture]

    @property
    def gestures(self) -> tuple[str, ...]:
        """Gestures present in this view."""
        return tuple(self._views)

    def select(self, **kw) -> HarmonisedView:
        """Channel selection on the common set; applies to both gestures at once."""
        probe = next(iter(self._views.values())).select(**kw)
        keep_names = set(probe.channels["name"])
        first = next(iter(self._views))
        mask = self.channels[f"native_name_{first}"].isin(keep_names).to_numpy()
        common = self.channels[mask].reset_index(drop=True)
        views = {
            g: replace(v, _cols=np.asarray(native_indices(common, g), dtype=np.int64)) for g, v in self._views.items()
        }
        return HarmonisedView(views, common, self.excluded)

    def concat_gestures(
        self, *, window: tuple[float, float], step: int = 1, targets: Sequence[str] = ()
    ) -> tuple[torch.Tensor, list[str], pd.DataFrame]:
        """Stack both gestures into one `(N_p+N_h, T, C)` tensor plus per-row labels.

        Harmonisation never touches the time axis — the gestures keep different
        anchors and different `T` — so the caller must fix a common window and
        stride. This asserts the resulting `(T, C)` and the channel set match, and
        raises rather than padding or warping.
        """
        prepared = {g: v.window(*window).thin(step) for g, v in self._views.items()}
        shapes = {g: v.shape[1:] for g, v in prepared.items()}
        if len({s for s in shapes.values()}) != 1:
            raise ValueError(
                f"gestures disagree on (T, C) after window={window} step={step}: {shapes}. "
                "Fix a window both gestures can supply; concat_gestures will not pad or warp."
            )
        names = {
            g: tuple(v.channels[list(SEMANTIC_KEY)].itertuples(index=False, name=None)) for g, v in prepared.items()
        }
        if len(set(names.values())) != 1:
            raise ValueError("gestures disagree on the harmonised channel set after selection")

        blocks, labels, frames = [], [], []
        for gesture, view in prepared.items():
            blocks.append(view.torch())
            labels.extend([gesture] * view.shape[0])
            frame = view.trials[["trial_id", "athlete_id", *targets]].copy()
            frame["gesture"] = gesture
            frame["athlete_key"] = gesture + ":" + frame["athlete_id"].astype("str")
            frames.append(frame)
        return torch.cat(blocks, dim=0), labels, pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------- root


class OBP:
    """Root handle over a built corpus — a single pack file or an intermediate dir."""

    def __init__(self, source: str | Path, mmap: bool = True) -> None:
        """`source` is `obp_corpus.safetensors` or the tensors directory."""
        self.store = Store(source, mmap=mmap)
        self._views = {g: GestureView.open(self.store, g) for g in self.store.gestures}
        log.info(
            "opened %s (%s) via the %s path", source, "pack" if self.store.is_pack else "dir", self.store.read_path
        )

    def __getitem__(self, gesture: str) -> GestureView:
        """The full view over one gesture."""
        if gesture not in self._views:
            raise KeyError(f"unknown gesture {gesture!r}; have {self.gestures}")
        return self._views[gesture]

    @property
    def gestures(self) -> tuple[str, ...]:
        """Gestures present in this corpus."""
        return self.store.gestures

    def harmonised(self, resolve_handedness: bool = False) -> HarmonisedView:
        """Cross-gesture view over the common channel set."""
        return HarmonisedView.build(dict(self._views), resolve_handedness=resolve_handedness)

    def __repr__(self) -> str:
        """Compact summary."""
        sizes = {g: self._views[g].shape for g in self.gestures}
        return f"<OBP {self.store.source.name} {sizes}>"
