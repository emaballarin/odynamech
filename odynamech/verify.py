"""Structural verification of a built corpus: the ways it can be silently wrong.

Each gate targets a failure that produces a corpus which loads fine and is
quietly incorrect — a drifted channel axis, padding read as signal, an athlete
leaking across a split. `verify_corpus()` returns a `Report`; nothing here
raises on a failed check, so a caller sees every failure rather than the first.

The NaN gate deserves a note. Upstream documentation states that measurement
channels contain no NaNs. That holds for `joint_angles` alone: `joint_velos` is
NaN at the first and last frame of every trial and `forces_moments` at the first
and last two — finite-difference edges, structurally undefined rather than
missing — and `energy_flow`'s joint-energy channels are NaN across whole trials
that lack force-plate data, since they are force-derived. Asserting "zero NaNs"
would therefore fail on correct data. This asserts the *structure* instead, which
is strictly stronger: corruption still fails, physics does not.
"""

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import GESTURES
from .harmonise import round_trip_ok
from .schema import channel_hash
from .store import Store
from .view import EmptySelection
from .view import OBP

log = logging.getLogger(__name__)

#: Frames from each end of a trial that a table's differentiation edge occupies.
EDGE_NAN_FRAMES: dict[tuple[str, str], int] = {
    ("pitching", "joint_velos"): 1,
    ("pitching", "forces_moments"): 2,
    ("pitching", "energy_flow"): 2,
    ("hitting", "joint_velos"): 1,
}


@dataclass(slots=True)
class Report:
    """Outcome of a verification run."""

    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return not self.failed

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        """Record one assertion."""
        text = f"{label}{f' — {detail}' if detail else ''}"
        (self.passed if condition else self.failed).append(text)
        log.info("[%s] %s", "PASS" if condition else "FAIL", text)

    def note(self, text: str) -> None:
        """Record a computed quantity that has no fixed expectation."""
        self.notes.append(text)
        log.info("[note] %s", text)

    def summary(self) -> str:
        """One-line tally."""
        return f"{len(self.passed)} passed, {len(self.failed)} failed, {len(self.notes)} notes"


def gate_clock(store: Store, report: Report) -> None:
    """Merging on a rounded float clock duplicates or drops rows without raising."""
    for gesture in store.gestures:
        offsets = store.tensor(gesture, "ragged", "offsets").numpy()
        frame = store.tensor(gesture, "ragged", "frame").numpy()
        index = store.index(gesture)
        lengths = np.diff(offsets)
        expected = np.concatenate([np.arange(n) for n in lengths]) if len(lengths) else np.empty(0)

        report.check(np.array_equal(frame, expected), f"{gesture}: frame index contiguous 0..len-1 within every trial")
        report.check(
            np.array_equal(index["n_frames"].to_numpy(), lengths), f"{gesture}: index n_frames matches offsets"
        )
        report.check(
            index["trial_id"].tolist() == store.json_meta(gesture, "ragged", "trial_ids", []),
            f"{gesture}: index row order matches the stored trial ids",
        )
        report.check(
            int(offsets[-1]) == int(store.tensor(gesture, "ragged", "values").shape[0]),
            f"{gesture}: offsets span the whole ragged values tensor",
        )


def gate_channels(store: Store, report: Report) -> None:
    """The stores and the schema must not disagree about which column is which."""
    for gesture in store.gestures:
        channels = store.channels(gesture)  # raises on a drifted hash
        stored = store.meta(gesture, "ragged")["channel_hash"]
        report.check(channel_hash(channels["name"].tolist()) == stored, f"{gesture}: schema hash matches the store")
        report.check(
            store.meta(gesture, "aligned")["channel_hash"] == stored,
            f"{gesture}: ragged and aligned agree on the channel hash",
        )
        report.check(
            list(channels["index"]) == list(range(len(channels))),
            f"{gesture}: channel index is 0-based and contiguous",
        )
        report.check(
            int(store.tensor(gesture, "ragged", "values").shape[1]) == len(channels),
            f"{gesture}: ragged channel axis matches the schema width",
        )
        report.check(
            int(store.tensor(gesture, "aligned", "values").shape[2]) == len(channels),
            f"{gesture}: aligned channel axis matches the schema width",
        )
        report.check(
            not channels.duplicated(subset=["group", "site", "subtype", "axis", "ref_segment"]).any(),
            f"{gesture}: the semantic tuple identifies exactly one channel",
        )


def gate_window(obp: OBP, report: Report) -> None:
    """A fixed window around the anchor can amputate the event it was cut around."""
    for gesture in obp.gestures:
        view = obp[gesture]
        raw_events = obp.store.tensor(gesture, "ragged", "events").numpy()
        rel_events = obp.store.tensor(gesture, "aligned", "events").numpy()
        index = obp.store.index(gesture)

        for j, name in enumerate(view.event_names):
            present = raw_events[:, j] >= 0
            outside = present & (rel_events[:, j] < 0)
            if name == view.anchor_event:
                report.check(
                    bool(present.all()) and not bool(outside.any()),
                    f"{gesture}: anchor {name} present and inside the window for every trial",
                )
            else:
                ids = index.loc[outside, "trial_id"].tolist()
                where = f" — {ids[:6]}{' ...' if len(ids) > 6 else ''}" if ids else ""
                report.note(
                    f"{gesture}: {name} outside the window for {int(outside.sum())}/{int(present.sum())} "
                    f"trials carrying it{where}"
                )
        report.check(
            bool((rel_events[rel_events >= 0] < view.shape[1]).all()),
            f"{gesture}: every in-window event index lies in [0, T)",
        )


def gate_padding(obp: OBP, report: Report) -> None:
    """Edge-padded frames look like a physically still athlete unless masked."""
    for gesture in obp.gestures:
        store = obp.store
        valid = store.tensor(gesture, "aligned", "valid").numpy()
        anchor = store.tensor(gesture, "aligned", "anchor_frame").numpy().astype(np.int64)
        index = store.index(gesture)
        pre = int(store.meta(gesture, "aligned")["pre_frames"])

        want = np.arange(valid.shape[1])[None, :] - pre + anchor[:, None]
        n = index["n_frames"].to_numpy()
        report.check(
            np.array_equal(valid, (want >= 0) & (want < n[:, None])),
            f"{gesture}: valid mask matches the anchor and length arithmetic",
        )
        report.check(
            np.array_equal((~valid).sum(axis=1), index["padded_frames"].to_numpy()),
            f"{gesture}: index padded_frames matches the mask",
        )
        report.check(bool(valid.any(axis=1).all()), f"{gesture}: no trial is entirely padded")
        report.note(
            f"{gesture}: padded fraction {100 * float(1.0 - valid.mean()):.4f} %, "
            f"worst trial {int((~valid).sum(axis=1).max())} frames"
        )


def gate_splits(obp: OBP, report: Report) -> None:
    """Several trials per athlete make a random trial split optimistic."""
    for gesture in obp.gestures:
        parts = obp[gesture].split_by_athlete((0.8, 0.1, 0.1), seed=0)
        sets = [set(p.trials["athlete_id"]) for p in parts]
        report.check(
            not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
            f"{gesture}: split athlete sets are pairwise disjoint",
        )
        report.check(
            sum(p.shape[0] for p in parts) == obp[gesture].shape[0],
            f"{gesture}: splits partition every trial",
        )
        report.note(f"{gesture}: split sizes {[p.shape[0] for p in parts]} trials, {[len(s) for s in sets]} athletes")


def gate_harmonisation(obp: OBP, report: Report) -> None:
    """A loose join key silently pairs channels that are not counterparts."""
    if len(obp.gestures) < 2:
        report.note("only one gesture present; harmonisation not applicable")
        return
    view = obp.harmonised()
    per = {g: obp.store.channels(g) for g in obp.gestures}
    common = view.channels

    for gesture in obp.gestures:
        column = f"native_index_{gesture}"
        report.check(common[column].is_unique, f"{gesture}: each harmonised entry maps to one native channel")
        report.check(
            common[column].between(0, len(per[gesture]) - 1).all(),
            f"{gesture}: every back-pointer is a valid native index",
        )
        report.check(
            view[gesture].channels["name"].tolist() == common[f"native_name_{gesture}"].tolist(),
            f"{gesture}: the restricted view exposes exactly the common channels",
        )
    report.check(round_trip_ok(common, per), "native -> harmonised -> native is the identity")
    report.note(f"harmonised common set: {len(common)} channels, by group {common.groupby('group').size().to_dict()}")
    report.note(f"excluded ambiguous sites: {view.excluded}")


def gate_complete(obp: OBP, report: Report) -> None:
    """Complete-cases can empty an axis and yield a tensor whose mean is NaN anyway."""
    for gesture in obp.gestures:
        done = obp[gesture].select(group=["angle"]).complete()
        report.check(min(done.shape) > 0, f"{gesture}: complete() on angles is non-empty", str(done.shape))
        report.check(not bool(torch.isnan(done.torch()).any()), f"{gesture}: complete() output has no NaNs")
        report.check(bool(done.mask().all()), f"{gesture}: complete() output has no padded frames")

        with_grf = obp[gesture].select(group=["angle", "grf"])
        done_grf = with_grf.complete()
        report.check(min(done_grf.shape) > 0, f"{gesture}: complete() with GRF is non-empty", str(done_grf.shape))
        report.check(not bool(torch.isnan(done_grf.torch()).any()), f"{gesture}: complete() with GRF has no NaNs")
        report.note(
            f"{gesture}: selecting GRF and completing dropped "
            f"{with_grf.shape[0] - done_grf.shape[0]} trials, cropped T to {done_grf.shape[1]}"
        )

        try:
            obp[gesture].trial("__nonexistent__").complete()
        except EmptySelection, ValueError:
            report.check(True, f"{gesture}: complete() raises rather than returning a collapsed axis")
        else:
            report.check(False, f"{gesture}: complete() returned a collapsed axis instead of raising")


def gate_fp_highrate(obp: OBP, report: Report) -> None:
    """The full-rate force plate must decimate back to what the main store holds."""
    for gesture in obp.gestures:
        view = obp[gesture]
        index = view.trials
        grf = index[index["has_grf"]]
        if grf.empty:
            report.note(f"{gesture}: no force-plate trials")
            continue

        # One trial per distinct ratio: hitting mixes 1x and 3x, so a single
        # sample would miss the odd one out entirely.
        for ratio in sorted(grf["fp_ratio"].unique()):
            row = grf[grf["fp_ratio"] == ratio].iloc[0]
            tid = row["trial_id"]
            high = view.fp_highrate(tid)
            report.check(
                high.shape[0] == int(row["n_frames"]) * int(ratio),
                f"{gesture}/{tid}: fp_highrate is exactly {int(ratio)}x the marker frame count",
            )
            report.check(
                view.fp_rate_hz(tid) == view.rate_hz * int(ratio),
                f"{gesture}/{tid}: reported full rate is {view.rate_hz * int(ratio):g} Hz",
            )
            grf_view = view.select(group=["grf"], drop_dead=False)
            perm = [view.fp_channels.index(n) for n in grf_view.channels["name"]]
            report.check(
                torch.allclose(high[:: int(ratio)][:, perm], grf_view.ragged(tid), equal_nan=True),
                f"{gesture}/{tid}: decimated full-rate equals the GRF channels in the main store",
            )

        missing = index.loc[~index["has_grf"], "trial_id"]
        if len(missing):
            tid = missing.iloc[0]
            try:
                view.fp_highrate(tid)
            except KeyError:
                report.check(True, f"{gesture}: fp_highrate raises for a GRF-missing trial ({tid})")
            else:
                report.check(False, f"{gesture}: fp_highrate did not raise for GRF-missing trial {tid}")
            grf_cols = view.select(group=["grf"], drop_dead=False)
            report.check(
                bool(torch.isnan(grf_cols.trial(tid).torch()).all()),
                f"{gesture}: GRF channels of a GRF-missing trial are NaN, not zero-filled",
            )


def gate_nan_structure(raw: Path, obp: OBP, report: Report) -> None:
    """Pin NaNs to the derivative edges and the force-derived trials, or enumerate them."""
    from .schema import GESTURE_KEY
    from .schema import GESTURE_TABLES
    from .schema import measurement_columns

    for gesture in obp.gestures:
        key = GESTURE_KEY[gesture]
        index = obp.store.index(gesture)
        no_grf = set(index.loc[~index["has_grf"], "trial_id"])

        for table in GESTURE_TABLES[gesture]:
            path = raw / gesture / f"{table}.csv"
            if not path.is_file():
                report.note(f"{gesture}/{table}: raw CSV absent, NaN structure not checked")
                continue
            frame = pd.read_csv(path)
            columns = measurement_columns(frame.columns, gesture)
            bad = np.isnan(frame[columns].to_numpy(dtype=np.float64))
            if not bad.any():
                report.check(True, f"{gesture}/{table}: zero NaNs in measurement channels")
                continue

            rows = bad.any(axis=1)
            lengths = frame.groupby(key).size().reindex(frame[key].to_numpy()).to_numpy()
            position = frame.groupby(key).cumcount().to_numpy()
            edge = EDGE_NAN_FRAMES.get((gesture, table))

            if edge is not None:
                explained = (position < edge) | (lengths - 1 - position < edge) | frame[key].isin(no_grf).to_numpy()
                report.check(
                    bool(explained[rows].all()),
                    f"{gesture}/{table}: every NaN row is a derivative edge (<{edge} frames) or a GRF-missing trial",
                    f"{int(rows.sum())} NaN rows, {int((~explained & rows).sum())} unexplained",
                )
            else:
                per_trial = pd.Series(rows, index=frame[key].to_numpy()).groupby(level=0).sum()
                affected = per_trial[per_trial > 0]
                report.note(
                    f"{gesture}/{table}: {int(bad.sum())} NaN cells over {int(rows.sum())} rows in "
                    f"{len(affected)} trials {dict(affected)}"
                )


def verify_corpus(
    source: str | Path,
    *,
    raw: Path | None = None,
    tensors: Path | None = None,
) -> Report:
    """Run every gate against a built corpus. Returns a `Report`; never raises on failure.

    `raw` enables the source-level NaN-structure gate; `tensors` enables the pack
    round-trip gate. Both are skipped, with a note, when not supplied.
    """
    report = Report()
    obp = OBP(source)

    gate_clock(obp.store, report)
    gate_channels(obp.store, report)
    gate_window(obp, report)
    gate_padding(obp, report)
    gate_splits(obp, report)
    gate_harmonisation(obp, report)
    gate_complete(obp, report)
    gate_fp_highrate(obp, report)

    if raw is not None and Path(raw).is_dir():
        gate_nan_structure(Path(raw), obp, report)
    else:
        report.note("raw CSVs not supplied; the source-level NaN-structure gate was skipped")

    if tensors is not None and Path(tensors).is_dir():
        _gate_pack(Path(tensors), report)
    else:
        report.note("intermediates not supplied; the pack round-trip gate was skipped")

    for gesture in obp.gestures:
        meta = obp.store.meta(gesture, "ragged")
        report.note(
            f"{gesture}: {meta['n_trials']} trials, rate {meta['rate_hz']} Hz, "
            f"fp ratios {json.loads(meta.get('fp_ratios', '[]'))}, "
            f"release {meta.get('obp_release_tag', '?')}, built {meta.get('build_utc', '?')}"
        )
    return report


def _gate_pack(tensors: Path, report: Report) -> None:
    """The byte-tensor round-trip for the index and schema can truncate silently."""
    from .pack import PackError
    from .pack import verify_pack

    pack_path = next(
        (
            p
            for p in tensors.glob("*.safetensors")
            if not p.name.endswith(("_ragged.safetensors", "_aligned.safetensors"))
        ),
        None,
    )
    if pack_path is None:
        report.note("no single-file pack alongside the intermediates; round-trip gate skipped")
        return

    # This gate compares a pack against the per-gesture intermediates it was made
    # from. A corpus obtained via `PackagedCorpus` has no intermediates — the pack
    # is self-contained and they would be redundant — so there is simply nothing
    # to compare against. That is the normal state of a fetched corpus, not a
    # failure, and must not be reported as one.
    if not any(tensors.glob("*_ragged.safetensors")):
        report.note("no per-gesture intermediates beside the pack; round-trip gate not applicable")
        return

    try:
        verify_pack(pack_path, tensors)
        report.check(True, "pack reproduces every tensor and both tables exactly")
    except (PackError, FileNotFoundError, KeyError) as exc:
        report.check(False, "pack round-trip", str(exc)[:300])


def expected_gestures() -> tuple[str, ...]:
    """Gestures a complete corpus is expected to carry."""
    return GESTURES
