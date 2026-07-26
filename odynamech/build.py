"""Construct the ragged, aligned and full-rate force-plate stores for one gesture."""

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from .config import DRIVELINE_RELEASE_TAG
from .schema import build_schema
from .schema import channel_hash
from .schema import cross_tab
from .schema import GESTURE_ANCHOR
from .schema import GESTURE_EVENTS
from .schema import GESTURE_KEY
from .schema import GESTURE_TABLES
from .schema import measurement_columns

MARKER_RATE_HZ = 360.0
FP_CHANNELS: tuple[str, ...] = (
    "rear_force_x",
    "rear_force_y",
    "rear_force_z",
    "lead_force_x",
    "lead_force_y",
    "lead_force_z",
)

LB_TO_KG = 0.45359237
IN_TO_M = 0.0254


class BuildError(RuntimeError):
    """A build-time invariant failed; the corpus is not written."""


@dataclass(slots=True)
class GestureBuild:
    """Everything one gesture's build produced, ready to serialise."""

    gesture: str
    rate_hz: float
    channels: pd.DataFrame
    index: pd.DataFrame
    ragged: dict[str, torch.Tensor]
    aligned: dict[str, torch.Tensor]
    ragged_meta: dict[str, str]
    aligned_meta: dict[str, str]
    report: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- IO


def _read_tables(raw: Path, gesture: str) -> dict[str, pd.DataFrame]:
    """Read every full-signal CSV for a gesture."""
    out = {}
    for table in GESTURE_TABLES[gesture]:
        out[table] = pd.read_csv(raw / gesture / f"{table}.csv")
    return out


def _frame_index(times: np.ndarray, rate: float) -> np.ndarray:
    """Reconstruct the integer sample index; never join on the rounded float `time`."""
    return np.rint(times * rate).astype(np.int64)


def _check_contiguous(frames: np.ndarray, trial_ids: np.ndarray, what: str) -> None:
    """Assert the frame index runs exactly 0..len-1 inside every trial."""
    order = np.lexsort((frames, trial_ids))
    sorted_ids, sorted_fr = trial_ids[order], frames[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    lengths = np.diff(np.r_[starts, len(sorted_ids)])
    expected = np.concatenate([np.arange(n) for n in lengths])
    if not np.array_equal(sorted_fr, expected):
        bad = sorted_ids[np.flatnonzero(sorted_fr != expected)[:5]]
        raise BuildError(
            f"{what}: reconstructed frame index is not contiguous 0..len-1 within every trial "
            f"(first offending trials: {bad.tolist()}). Refusing to work around it."
        )


def _detect_rate(df: pd.DataFrame, key: str, candidates: tuple[float, ...]) -> float:
    """Pick the unique rate under which every trial's frame index is exactly 0..len-1."""
    good = []
    ids = df[key].to_numpy()
    for rate in candidates:
        frames = _frame_index(df["time"].to_numpy(), rate)
        try:
            _check_contiguous(frames, ids, "probe")
        except BuildError:
            continue
        good.append(rate)
    if len(good) != 1:
        raise BuildError(f"sample rate is not uniquely determined; candidates satisfying contiguity: {good}")
    return good[0]


# ------------------------------------------------------------------ ragged store


def _build_ragged(
    tables: dict[str, pd.DataFrame], gesture: str, rate: float, report: list[str]
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, list[str], dict[str, object]]:
    """Concatenate every marker-derived table along the channel axis, CSR-style."""
    key = GESTURE_KEY[gesture]
    events = list(GESTURE_EVENTS[gesture])
    marker_tables = [t for t in GESTURE_TABLES[gesture] if t != "force_plate"]

    base = tables[marker_tables[0]]
    base_ids = base[key].to_numpy()
    base_frames = _frame_index(base["time"].to_numpy(), rate)
    _check_contiguous(base_frames, base_ids, f"{gesture}/{marker_tables[0]}")

    # Canonical trial order: sorted trial IDs.
    trial_ids = sorted(base[key].unique().tolist())
    n_per = base.groupby(key).size().reindex(trial_ids).to_numpy()
    offsets = np.zeros(len(trial_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_per)
    total = int(offsets[-1])

    # Row order used everywhere below: (canonical trial rank, time).  The rank must
    # come from the sorted `trial_ids` list, not from order of appearance — the two
    # differ for hitting, and using appearance order silently misaligns the CSR
    # offsets against the row blocks.  Sorting on the float `time` is safe (ordering
    # survives the 4-decimal rounding); only *joining* on it would not be.
    rank_of = {t: i for i, t in enumerate(trial_ids)}

    def _row_order(df: pd.DataFrame) -> np.ndarray:
        rank = df[key].map(rank_of).to_numpy(dtype=np.int64)
        return np.lexsort((df["time"].to_numpy(), rank))

    # Every marker-derived table must agree on trial set and per-trial length.
    for table in marker_tables:
        df = tables[table]
        ids = df[key].to_numpy()
        fr = _frame_index(df["time"].to_numpy(), rate)
        _check_contiguous(fr, ids, f"{gesture}/{table}")
        if df.shape[0] != base.shape[0]:
            raise BuildError(f"{gesture}/{table}: {df.shape[0]} rows, expected {base.shape[0]}")
        if sorted(df[key].unique().tolist()) != trial_ids:
            raise BuildError(f"{gesture}/{table}: trial-ID set differs from {marker_tables[0]}")
        got = df.groupby(key).size().reindex(trial_ids).to_numpy()
        if not np.array_equal(got, n_per):
            raise BuildError(f"{gesture}/{table}: per-trial row counts differ from {marker_tables[0]}")

    # Constant-zero flags, computed from the data rather than assumed.
    zeros: dict[str, bool] = {}
    for table in GESTURE_TABLES[gesture]:
        df = tables[table]
        cols = measurement_columns(df.columns, gesture)
        arr = df[cols].to_numpy(dtype=np.float64)
        for c, z in zip(cols, np.all(arr == 0.0, axis=0), strict=True):
            zeros[c] = bool(z)

    schema = build_schema({t: list(tables[t].columns) for t in GESTURE_TABLES[gesture]}, gesture, constant_zero=zeros)
    names = schema["name"].tolist()

    # Assemble (S, C) in canonical channel order.
    values = np.full((total, len(names)), np.nan, dtype=np.float32)
    col_of = {n: i for i, n in enumerate(names)}
    for table in marker_tables:
        df = tables[table]
        order = _row_order(df)
        cols = measurement_columns(df.columns, gesture)
        block = df[cols].to_numpy(dtype=np.float32)[order]
        for j, c in enumerate(cols):
            values[:, col_of[c]] = block[:, j]

    order0 = _row_order(base)
    frame = base_frames[order0].astype(np.int32)

    # Events: one row per trial, in frames, -1 where the source is NaN.
    ev_src = base.groupby(key)[events].first().reindex(trial_ids)
    ev = np.full((len(trial_ids), len(events)), -1, dtype=np.int32)
    for j, e in enumerate(events):
        col = ev_src[e].to_numpy(dtype=np.float64)
        ok = ~np.isnan(col)
        ev[ok, j] = np.rint(col[ok] * rate).astype(np.int32)

    # ------------------------------------------------------------- force plate
    fp = tables["force_plate"]
    fp_ids_present = set(fp[key].unique().tolist())
    stray = fp_ids_present - set(trial_ids)
    if stray:
        raise BuildError(f"{gesture}/force_plate: trials absent from the marker tables: {sorted(stray)[:5]}")

    fp_n = fp.groupby(key).size()
    has_grf = np.array([t in fp_ids_present for t in trial_ids], dtype=bool)

    # The ratio is per trial, not per gesture: hitting mixes 3x (1080 Hz) with 1x
    # (360 Hz, the whole of athlete 17).  A single global rate fails contiguity
    # outright, so each trial's rate comes from its own row count and is verified
    # against its own clock below.
    fp_ratio = np.zeros(len(trial_ids), dtype=np.int32)
    for i, t in enumerate(trial_ids):
        if not has_grf[i]:
            continue
        num, den = int(fp_n[t]), int(n_per[i])
        if num % den:
            raise BuildError(f"{gesture}/{t}: force-plate rows {num} are not an integer multiple of {den}")
        fp_ratio[i] = num // den

    ratios = sorted({int(r) for r in fp_ratio[has_grf]})
    report.append(
        f"force-plate ratios present: {ratios} (per-trial counts: { {r: int((fp_ratio == r).sum()) for r in ratios} })"
    )

    fp_rate = rate * max(ratios)  # the highest native rate present

    # Full-rate store, written untouched, before any decimation.
    fp_order = _row_order(fp)
    fp_trial_ids = [t for t in trial_ids if t in fp_ids_present]
    fp_lengths = np.array([int(fp_n[t]) for t in fp_trial_ids], dtype=np.int64)
    fp_offsets = np.zeros(len(fp_trial_ids) + 1, dtype=np.int64)
    fp_offsets[1:] = np.cumsum(fp_lengths)
    fp_values = fp[list(FP_CHANNELS)].to_numpy(dtype=np.float32)[fp_order]

    # Frame index within each trial, at that trial's own rate.  Verified per trial
    # rather than assumed, since the rate is not uniform across hitting.
    fp_time = fp["time"].to_numpy()[fp_order]
    fp_frames_all = np.empty(len(fp_time), dtype=np.int64)
    ratio_of = dict(zip(trial_ids, fp_ratio.tolist(), strict=True))
    for i, t in enumerate(fp_trial_ids):
        lo, hi = int(fp_offsets[i]), int(fp_offsets[i + 1])
        local = _frame_index(fp_time[lo:hi], rate * ratio_of[t])
        if not np.array_equal(local, np.arange(hi - lo)):
            raise BuildError(
                f"{gesture}/{t}: force-plate frame index is not contiguous 0..len-1 "
                f"at its own rate {rate * ratio_of[t]:g} Hz"
            )
        fp_frames_all[lo:hi] = local

    # Decimated force plate folded into the main channel axis, phase 0.
    grf_cols = [c for c in names if schema.loc[col_of[c], "group"] == "grf"]
    src_of = dict(zip(FP_CHANNELS, range(len(FP_CHANNELS)), strict=True))
    fp_pos = {t: (int(fp_offsets[i]), int(fp_offsets[i + 1])) for i, t in enumerate(fp_trial_ids)}
    for i, t in enumerate(trial_ids):
        if not has_grf[i]:
            continue  # stays NaN; never zero-filled
        lo, hi = fp_pos[t]
        r = int(fp_ratio[i])
        block = fp_values[lo:hi][::r]
        if len(block) != n_per[i]:
            raise BuildError(f"{gesture}/{t}: decimated force plate has {len(block)} rows, expected {n_per[i]}")
        # Verify the decimation lands on the marker clock rather than assuming it.
        if not np.array_equal(fp_frames_all[lo:hi][::r] // r, np.arange(n_per[i])):
            raise BuildError(f"{gesture}/{t}: phase-0 decimation does not align with the marker clock")
        for c in grf_cols:
            values[offsets[i] : offsets[i + 1], col_of[c]] = block[:, src_of[c]]

    ragged = {
        "values": torch.from_numpy(values),
        "offsets": torch.from_numpy(offsets),
        "frame": torch.from_numpy(frame),
        "events": torch.from_numpy(ev),
        "has_grf": torch.from_numpy(has_grf),
        "fp_ratio": torch.from_numpy(fp_ratio),
        "fp_values": torch.from_numpy(fp_values),
        "fp_offsets": torch.from_numpy(fp_offsets),
        "fp_frame": torch.from_numpy(fp_frames_all.astype(np.int32)),
    }
    extras = {
        "trial_ids": trial_ids,
        "fp_trial_ids": fp_trial_ids,
        "n_per": n_per,
        "events": events,
        "fp_rate": fp_rate,
        "fp_ratios": ratios,
        "fp_ratio": fp_ratio,
        "has_grf": has_grf,
    }
    report.append(
        f"ragged: values {values.shape}, {len(trial_ids)} trials, "
        f"fp_values {fp_values.shape} over {len(fp_trial_ids)} trials at {fp_rate:g} Hz"
    )
    return ragged, schema, names, extras


# ----------------------------------------------------------------- aligned store


def _choose_window(before: np.ndarray, after: np.ndarray, coverage: float = 0.95) -> tuple[int, int]:
    """Largest, most symmetric `[-A, +B]` around the anchor that needs no padding for >= `coverage`."""
    need = int(np.ceil(coverage * len(before)))
    best: tuple[int, int, int] | None = None  # (total, -asymmetry, ...) maximised
    for a in np.unique(before):
        # Largest B for which at least `need` trials satisfy both constraints.
        cand = np.sort(after[before >= a])[::-1]
        if len(cand) < need:
            continue
        b = int(cand[need - 1])
        key = (int(a) + b, -abs(int(a) - b))
        if best is None or key > best[:2]:
            best = (*key, int(a), b)  # type: ignore[assignment]
    if best is None:
        raise BuildError(f"no window covers {coverage:.0%} of trials without padding")
    return best[2], best[3]  # type: ignore[misc]


def _build_aligned(
    ragged: dict[str, torch.Tensor], extras: dict[str, object], gesture: str, rate: float, report: list[str]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Anchor-and-crop in native time; edge-pad and record a mandatory validity mask."""
    events: list[str] = extras["events"]  # type: ignore[assignment]
    anchor_name = GESTURE_ANCHOR[gesture]
    ai = events.index(anchor_name)

    values = ragged["values"].numpy()
    offsets = ragged["offsets"].numpy()
    ev = ragged["events"].numpy()
    n_per: np.ndarray = extras["n_per"]  # type: ignore[assignment]
    n_trials, n_chan = len(n_per), values.shape[1]

    anchor = ev[:, ai].astype(np.int64)
    if (anchor < 0).any():
        bad = np.flatnonzero(anchor < 0)[:5]
        raise BuildError(f"{gesture}: anchor {anchor_name} missing for trials at rows {bad.tolist()}")
    if not ((anchor >= 0) & (anchor < n_per)).all():
        raise BuildError(f"{gesture}: anchor {anchor_name} falls outside the trial for some trials")

    pre, post = _choose_window(anchor, n_per - 1 - anchor, 0.95)
    T = pre + post + 1
    no_pad = ((anchor >= pre) & (n_per - 1 - anchor >= post)).mean()
    report.append(
        f"aligned window: anchor {anchor_name}, pre {pre} post {post} frames, T={T} "
        f"({T / rate:.4f} s); {100 * no_pad:.2f} % of trials need no padding"
    )

    out = np.empty((n_trials, T, n_chan), dtype=np.float32)
    valid = np.zeros((n_trials, T), dtype=bool)
    want = np.arange(-pre, post + 1)
    for i in range(n_trials):
        lo, hi, n = int(offsets[i]), int(offsets[i + 1]), int(n_per[i])
        src = anchor[i] + want
        inside = (src >= 0) & (src < n)
        out[i] = values[lo:hi][np.clip(src, 0, n - 1)]  # edge-pad outside the trial
        valid[i] = inside

    padded_fraction = float(1.0 - valid.mean())
    padded_per_trial = (~valid).sum(axis=1).astype(np.int32)

    # Events relative to the window; -1 when absent or outside it.
    ev_rel = np.full_like(ev, -1)
    for j in range(ev.shape[1]):
        rel = ev[:, j].astype(np.int64) - anchor + pre
        ok = (ev[:, j] >= 0) & (rel >= 0) & (rel < T)
        ev_rel[ok, j] = rel[ok].astype(np.int32)

    outside = [(events[j], int(((ev[:, j] >= 0) & (ev_rel[:, j] < 0)).sum())) for j in range(ev.shape[1])]
    report.append(f"events falling outside the window: {dict(outside)}")
    report.append(f"padded fraction overall: {100 * padded_fraction:.3f} %")

    aligned = {
        "values": torch.from_numpy(out),
        "valid": torch.from_numpy(valid),
        "events": torch.from_numpy(ev_rel),
        "anchor_frame": torch.from_numpy(anchor.astype(np.int32)),
        "has_grf": ragged["has_grf"].clone(),
        "padded_frames": torch.from_numpy(padded_per_trial),
    }
    meta = {
        "anchor_event": anchor_name,
        "pre_frames": str(pre),
        "post_frames": str(post),
        "T": str(T),
        "window_seconds": f"{T / rate:.6f}",
        "padded_fraction": f"{padded_fraction:.8f}",
    }
    return aligned, meta


# ------------------------------------------------------------------ trial index


def _build_index(raw: Path, gesture: str, extras: dict[str, object], aligned: dict[str, torch.Tensor]) -> pd.DataFrame:
    """One row per trial, row order identical to the first axis of both stores."""
    key = GESTURE_KEY[gesture]
    trial_ids: list[str] = extras["trial_ids"]  # type: ignore[assignment]
    meta = pd.read_csv(raw / gesture / "metadata.csv")
    poi = pd.read_csv(raw / gesture / "poi_metrics.csv")

    frame = pd.DataFrame({
        "row": np.arange(len(trial_ids), dtype=np.int64),
        "trial_id": pd.Series(trial_ids, dtype="str"),
    })
    frame["athlete_id"] = frame["trial_id"].str.split("_").str[0]

    meta = meta.drop_duplicates(subset=key).set_index(key)
    joined = frame.join(meta, on="trial_id", rsuffix="_meta")

    # Normalise to SI; hitting ships imperial units, pitching SI.
    if gesture == "pitching":
        joined["mass_kg"] = joined["session_mass_kg"].astype("float64")
        joined["height_m"] = joined["session_height_m"].astype("float64")
        joined["age_yrs"] = joined["age_yrs"].astype("float64")
        joined["playing_level"] = joined["playing_level"].astype("str")
    else:
        joined["mass_kg"] = joined["session_mass_lbs"].astype("float64") * LB_TO_KG
        joined["height_m"] = joined["session_height_in"].astype("float64") * IN_TO_M
        joined["age_yrs"] = joined["athlete_age"].astype("float64")
        joined["playing_level"] = joined["highest_playing_level"].astype("str")

    if "session" in joined.columns:
        joined["session_id"] = joined["session"].astype("str")

    poi = poi.drop_duplicates(subset=key).set_index(key)
    poi = poi.rename(columns={c: f"poi_{c}" for c in poi.columns})
    joined = joined.join(poi, on="trial_id")

    # Handedness lives in different tables per gesture.
    if gesture == "pitching":
        joined["handedness"] = joined["poi_p_throws"].astype("str")
        joined["outcome"] = joined["poi_pitch_speed_mph"].astype("float64")
    else:
        joined["handedness"] = joined["hitter_side"].astype("str")
        joined["outcome"] = joined["exit_velo_mph_x"].astype("float64")

    tail = pd.DataFrame(
        {
            "n_frames": np.asarray(extras["n_per"], dtype=np.int64),
            "has_grf": np.asarray(extras["has_grf"], dtype=bool),
            "fp_ratio": np.asarray(extras["fp_ratio"], dtype=np.int64),
            "padded_frames": aligned["padded_frames"].numpy().astype(np.int64),
            "trial_in_athlete": joined.groupby("athlete_id").cumcount().astype(np.int64).to_numpy(),
        },
        index=joined.index,
    )
    return pd.concat([joined, tail], axis=1).reset_index(drop=True).copy()


# ------------------------------------------------------------------------ driver


def build_gesture(raw: Path, gesture: str) -> GestureBuild:
    """Build every store for one gesture, verifying invariants as it goes."""
    report: list[str] = [f"## {gesture}", ""]
    tables = _read_tables(raw, gesture)
    for t, df in tables.items():
        report.append(f"- `{t}`: {df.shape[0]:,} rows x {df.shape[1]} cols")

    key = GESTURE_KEY[gesture]
    rate = _detect_rate(
        tables[GESTURE_TABLES[gesture][0]], key, (100.0, 120.0, 200.0, 240.0, 250.0, 300.0, 360.0, 480.0, 500.0, 600.0)
    )
    report.append(f"- marker rate: **{rate:g} Hz** (uniquely determined)")

    ragged, schema, names, extras = _build_ragged(tables, gesture, rate, report)
    aligned, aligned_meta = _build_aligned(ragged, extras, gesture, rate, report)

    index = _build_index(raw, gesture, extras, aligned)
    index["fp_ratio"] = ragged["fp_ratio"].numpy().astype(np.int64)

    chash = channel_hash(names)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    common = {
        "gesture": gesture,
        "rate_hz": f"{rate:g}",
        "fp_rate_hz": f"{extras['fp_rate']:g}",
        "fp_ratios": json.dumps(extras["fp_ratios"]),
        "n_trials": str(len(extras["trial_ids"])),  # type: ignore[arg-type]
        "channel_hash": chash,
        "event_names": json.dumps(extras["events"]),
        "trial_ids": json.dumps(extras["trial_ids"]),
        "fp_trial_ids": json.dumps(extras["fp_trial_ids"]),
        "fp_channels": json.dumps(list(FP_CHANNELS)),
        "build_utc": now,
        "obp_release_tag": DRIVELINE_RELEASE_TAG,
    }
    ragged_meta = {**common, "channels": schema.to_json(orient="records")}
    report.append(f"- channels: {len(names)} (hash `{chash[:16]}`)")
    report.append(f"- dead channels: {int(schema['constant_zero'].sum())}")
    report.append("")
    report.append("```")
    report.append(cross_tab(schema))
    report.append("```")

    return GestureBuild(
        gesture=gesture,
        rate_hz=rate,
        channels=schema,
        index=index,
        ragged=ragged,
        aligned=aligned,
        ragged_meta=ragged_meta,
        aligned_meta={**common, **aligned_meta},
        report=report,
    )


def write_intermediates(build: GestureBuild, out: Path) -> None:
    """Write the per-gesture intermediate files the pack step consumes."""
    out.mkdir(parents=True, exist_ok=True)
    g = build.gesture
    save_file(build.ragged, str(out / f"{g}_ragged.safetensors"), metadata=build.ragged_meta)
    save_file(build.aligned, str(out / f"{g}_aligned.safetensors"), metadata=build.aligned_meta)
    build.channels.to_csv(out / f"{g}_channels.csv", index=False)
    build.index.to_parquet(out / f"{g}_index.parquet", index=False)
