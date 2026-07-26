"""Access-layer behaviour against a real corpus. Skipped unless ODYNAMECH_TEST_CORPUS is set."""

from pathlib import Path

import numpy as np
import pytest
import torch

from odynamech import OBP
from odynamech.harmonise import round_trip_ok
from odynamech.harmonise import SEMANTIC_KEY
from odynamech.schema import build_schema
from odynamech.schema import channel_hash
from odynamech.schema import parse_channel
from odynamech.schema import UnclassifiedChannel
from odynamech.store import bytes_to_tensor
from odynamech.store import ChannelHashMismatch
from odynamech.store import Store
from odynamech.store import tensor_to_bytes
from odynamech.view import EmptySelection

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def tensors(corpus_path: Path) -> Path:
    """The directory the corpus lives in, for the pack-vs-directory comparisons."""
    return corpus_path if corpus_path.is_dir() else corpus_path.parent


# ------------------------------------------------------------------ schema


def test_semantic_tuple_is_unique_per_gesture(obp: OBP) -> None:
    """Gate 6 depends on the tuple identifying exactly one channel."""
    for gesture in obp.gestures:
        channels = obp.store.channels(gesture)
        assert not channels.duplicated(subset=list(SEMANTIC_KEY)).any()


# ------------------------------------------------------------------- shapes


def test_shape_matches_materialisation(obp: OBP) -> None:
    """`.shape` must predict what `.torch()` returns, without reading it."""
    view = obp["pitching"].athlete("1031").select(group=["angle"]).thin(3)
    assert view.shape == tuple(view.torch().shape)


def test_selectors_commute(obp: OBP) -> None:
    """Views compose in any order."""
    base = obp["pitching"]
    a = base.select(site="elbow").athlete("1031").thin(2)
    b = base.thin(2).athlete("1031").select(site="elbow")
    assert a.shape == b.shape
    # equal_nan: the elbow carries force/moment/energy channels whose derivative
    # edges are NaN, and torch.equal is False for any tensor containing NaN.
    assert torch.allclose(a.torch(), b.torch(), equal_nan=True)


def test_time_axis_is_zero_at_the_anchor(obp: OBP) -> None:
    """`time` is seconds relative to the anchor, negative before it."""
    for gesture in obp.gestures:
        view = obp[gesture]
        t = view.time
        assert float(t[view.pre_frames]) == pytest.approx(0.0, abs=1e-9)
        assert float(t[0]) < 0.0


def test_window_and_thin_keep_the_anchor_on_grid(obp: OBP) -> None:
    """Thinning must not shift the anchor off the sampled grid."""
    view = obp["pitching"].thin(4)
    assert np.isclose(view.time.numpy(), 0.0).any()


def test_select_site_returns_every_quantity_there(obp: OBP) -> None:
    """A site selects angle, velocity, force, moment and energy together."""
    groups = set(obp["pitching"].select(site="elbow").channels["group"])
    assert {"angle", "velo", "force", "moment", "energy"} <= groups


def test_dead_channels_are_stored_but_filtered_by_default(obp: OBP) -> None:
    """Dead channels stay in the file and are dropped by the access layer."""
    view = obp["pitching"]
    assert int(obp.store.channels("pitching")["constant_zero"].sum()) == 7
    assert not view.select(drop_dead=True).channels["constant_zero"].any()
    assert view.select(drop_dead=False).shape[2] > view.select(drop_dead=True).shape[2]


def test_getitem_sugar(obp: OBP) -> None:
    """`g["id"]` selects a trial; `g[:, ::4, :]` thins time."""
    view = obp["pitching"]
    assert view["1031_2"].shape[0] == 1
    assert view[:, ::4, :].shape[1] == len(range(0, view.shape[1], 4))


# ------------------------------------------------------------------ mask / NaN


def test_mask_is_returned_and_marks_padding(obp: OBP) -> None:
    """The validity mask is mandatory and actually flags padded frames."""
    view = obp["pitching"]
    mask = view.mask()
    assert mask.shape == (view.shape[0], view.shape[1])
    assert mask.dtype == torch.bool
    assert not bool(mask.all())  # some trials are padded
    assert bool(mask.any(dim=1).all())  # none is entirely padded


def test_grf_missing_trials_are_nan_not_zero(obp: OBP) -> None:
    """Absent force plate must read as NaN, never as a silent zero."""
    view = obp["pitching"].select(group=["grf"], drop_dead=False)
    missing = view.where(lambda f: ~f["has_grf"].astype(bool))
    assert missing.shape[0] == 8
    values = missing.torch()
    assert bool(torch.isnan(values).all())


def test_nan_policy_keep_is_the_default_and_edits_nothing(obp: OBP) -> None:
    """`keep` is the faithful record."""
    view = obp["pitching"].select(group=["velo"])
    assert torch.equal(
        torch.nan_to_num(view.torch(), nan=-999.0),
        torch.nan_to_num(view.nans("keep").torch(), nan=-999.0),
    )


def test_nan_policy_interpolate_fills_bracketed_gaps_only(obp: OBP) -> None:
    """Interior dropouts are interpolated; unbracketed derivative edges are not."""
    view = obp["hitting"].trial("440_6").select(group=["landmark"])
    before = int(torch.isnan(view.torch()).sum())
    after = int(torch.isnan(view.nans("interpolate").torch()).sum())
    assert before > 0
    assert after < before


def test_nan_policy_drop_removes_athletes_or_channels(obp: OBP) -> None:
    """`drop` yields a NaN-free block and removes whole athletes or whole channels."""
    view = obp["pitching"].athlete(["1031", "1097", "2919"]).select(group=["angle", "grf"])
    dropped = view.nans("drop").torch()
    assert not bool(torch.isnan(dropped).any())
    assert dropped.shape[0] <= view.shape[0]
    assert dropped.shape[2] <= view.shape[2]


def test_nan_policy_rejects_unknown_modes(obp: OBP) -> None:
    """Exactly three modes exist."""
    with pytest.raises(ValueError, match="nan policy"):
        obp["pitching"].nans("ffill")


# ------------------------------------------------------------------ complete


def test_complete_is_dense_unpadded_and_nan_free(obp: OBP) -> None:
    """Decision 13: a rectangular block with no NaN and no padding."""
    done = obp["pitching"].select(group=["angle"]).complete()
    assert min(done.shape) > 0
    assert not bool(torch.isnan(done.torch()).any())
    assert bool(done.mask().all())


def test_complete_with_grf_drops_exactly_the_grf_missing_trials(obp: OBP) -> None:
    """Selecting GRF and completing must remove the 8 trials without force plate."""
    view = obp["pitching"].select(group=["angle", "grf"])
    done = view.complete()
    assert view.shape[0] - done.shape[0] == 8
    assert not bool(torch.isnan(done.torch()).any())


def test_complete_raises_when_an_axis_collapses(obp: OBP) -> None:
    """Gate 7: a collapsed axis raises with a message, never returns an empty tensor."""
    with pytest.raises((EmptySelection, ValueError)):
        obp["pitching"].trial("__nope__").complete()


def test_torch_complete_kwarg_matches_the_method(obp: OBP) -> None:
    """`complete=True` is sugar for `.complete().torch()`."""
    view = obp["pitching"].select(group=["angle"])
    assert torch.equal(view.torch(complete=True), view.complete().torch())


# ------------------------------------------------------------------ harmonise


def test_harmonised_set_is_the_expected_size(obp: OBP) -> None:
    """27 angles + 21 velos + 12 thorax/CoM landmarks + 6 GRF."""
    common = obp.harmonised().channels
    assert len(common) == 66
    assert common.groupby("group").size().to_dict() == {"angle": 27, "velo": 21, "landmark": 12, "grf": 6}


def test_harmonised_round_trip_is_identity(obp: OBP) -> None:
    """Gate 6: native -> harmonised -> native must be the identity."""
    per = {g: obp.store.channels(g) for g in obp.gestures}
    assert round_trip_ok(obp.harmonised().channels, per)


def test_harmonised_view_maps_back_to_a_native_view(obp: OBP) -> None:
    """`.harmonised()["pitching"]` is a real GestureView restricted to the common set."""
    view = obp.harmonised()
    common = view.channels
    for gesture in obp.gestures:
        native = view[gesture]
        assert native.channels["name"].tolist() == common[f"native_name_{gesture}"].tolist()
        assert native.shape[2] == len(common)


def test_harmonisation_excludes_the_arms(obp: OBP) -> None:
    """The arm correspondence is never guessed."""
    sites = set(obp.harmonised().channels["site"])
    for arm in ("shoulder", "elbow", "wrist", "lead_shoulder", "rear_shoulder", "left_wrist"):
        assert arm not in sites


def test_concat_gestures_asserts_rather_than_padding(obp: OBP) -> None:
    """Mismatched (T, C) must raise; the helper never pads or warps."""
    view = obp.harmonised()
    tensor, labels, frame = view.concat_gestures(window=(-0.5, 0.05))
    assert tensor.shape[0] == len(labels) == len(frame)
    assert tensor.shape[0] == obp["pitching"].shape[0] + obp["hitting"].shape[0]
    assert tensor.shape[2] == 66
    assert set(labels) == set(obp.gestures)
    # Athlete keys are gesture-qualified so a disjoint split stays disjoint.
    assert frame["athlete_key"].str.contains(":").all()


def test_harmonised_select_applies_to_both_gestures(obp: OBP) -> None:
    """Channel selection on the common set restricts both sides together."""
    keep_dead = obp.harmonised().select(group=["angle"], drop_dead=False)
    assert len(keep_dead.channels) == 27
    for gesture in obp.gestures:
        assert keep_dead[gesture].shape[2] == 27

    # Decision 7: dead channels stay in the file but the access layer filters them
    # by default. Four of the 27 common angle channels are model-constrained zeros.
    default = obp.harmonised().select(group=["angle"])
    assert len(default.channels) == 23
    assert set(keep_dead.channels["native_name_pitching"]) - set(default.channels["native_name_pitching"]) == {
        "lead_knee_angle_y",
        "lead_knee_angle_z",
        "rear_knee_angle_y",
        "rear_knee_angle_z",
    }
    for gesture in obp.gestures:
        assert default[gesture].shape[2] == 23


# ---------------------------------------------------------------- store / pack


def test_channel_hash_mismatch_raises(tensors: Path) -> None:
    """Gate 2: a drifted hash must be caught on open, not silently tolerated."""
    store = Store(tensors)
    gesture = store.gestures[0]
    store._meta[f"{gesture}/ragged"]["channel_hash"] = "0" * 64
    with pytest.raises(ChannelHashMismatch):
        store.channels(gesture)


def test_pack_and_directory_agree(obp: OBP, tensors: Path) -> None:
    """The loader behaves identically on the pack and on the intermediates."""
    loose = OBP(tensors)
    for gesture in obp.gestures:
        assert obp[gesture].shape == loose[gesture].shape
        tid = obp[gesture].trials["trial_id"][0]
        assert torch.equal(obp[gesture].trial(tid).torch(), loose[gesture].trial(tid).torch())


# --------------------------------------------------------------- ragged / fp


def test_ragged_is_full_native_length(obp: OBP) -> None:
    """The ragged escape hatch returns the unpadded trial at its true length."""
    view = obp["pitching"]
    row = view.trials.iloc[0]
    assert view.ragged(row["trial_id"]).shape == (int(row["n_frames"]), view.shape[2])


def test_fp_highrate_ratio_and_decimation(obp: OBP) -> None:
    """Full-rate force plate is exactly ratio x, and decimates back to the main store."""
    for gesture in obp.gestures:
        view = obp[gesture]
        index = view.trials
        grf = index[index["has_grf"]]
        for ratio in sorted(grf["fp_ratio"].unique()):
            row = grf[grf["fp_ratio"] == ratio].iloc[0]
            hi = view.fp_highrate(row["trial_id"])
            assert hi.shape[0] == int(row["n_frames"]) * int(ratio)
            assert view.fp_rate_hz(row["trial_id"]) == view.rate_hz * int(ratio)
            grf_view = view.select(group=["grf"], drop_dead=False)
            perm = [view.fp_channels.index(n) for n in grf_view.channels["name"]]
            assert torch.allclose(hi[:: int(ratio)][:, perm], grf_view.ragged(row["trial_id"]), equal_nan=True)


def test_fp_highrate_raises_without_grf(obp: OBP) -> None:
    """No silent empty tensor for a trial that has no force plate."""
    with pytest.raises(KeyError):
        obp["pitching"].fp_highrate("1370_1")


def test_hitting_force_plate_ratio_is_per_trial(obp: OBP) -> None:
    """Hitting genuinely mixes 1x and 3x; the store records it per trial."""
    index = obp["hitting"].trials
    grf = index[index["has_grf"]]
    assert set(grf["fp_ratio"].unique()) == {1, 3}
    assert set(grf.loc[grf["fp_ratio"] == 1, "trial_id"]) == {"17_1", "17_2", "17_3", "17_4", "17_5"}
    assert set(index.loc[~index["has_grf"], "fp_ratio"]) == {0}


# -------------------------------------------------------------- torch interop


def test_dataset_contract(obp: OBP) -> None:
    """The dataset yields the documented keys with the documented shapes."""
    view = obp["pitching"].athlete(["1031", "1097"]).select(group=["angle"])
    dataset = view.dataset(targets=["outcome"])
    item = dataset[0]
    assert item["x"].shape == (view.shape[1], view.shape[2])
    assert item["mask"].shape == (view.shape[1],)
    assert isinstance(item["trial_id"], str)
    assert isinstance(item["athlete_id"], str)
    assert item["events"].shape == (len(view.event_names),)
    assert "outcome" in item


def test_dataloader_collation(obp: OBP) -> None:
    """Tensors stack; string fields stay lists."""
    view = obp["pitching"].athlete(["1031", "1097"]).select(group=["angle"])
    batch = next(iter(view.dataloader(batch_size=3)))
    assert batch["x"].shape[0] == 3
    assert isinstance(batch["trial_id"], list) and len(batch["trial_id"]) == 3


def test_dataloader_shuffle_needs_an_explicit_generator(obp: OBP) -> None:
    """No fixed seed is fabricated inside a signature."""
    view = obp["pitching"].athlete("1031").select(group=["angle"])
    with pytest.raises(ValueError, match="generator"):
        view.dataloader(batch_size=2, shuffle=True)
    gen = torch.Generator()
    gen.manual_seed(12345)
    assert view.dataloader(batch_size=2, shuffle=True, generator=gen) is not None


def test_split_by_athlete_is_disjoint(obp: OBP) -> None:
    """Gate 5: no athlete may appear in two splits."""
    parts = obp["pitching"].split_by_athlete((0.6, 0.2, 0.2), seed=7)
    sets = [set(p.trials["athlete_id"]) for p in parts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(p.shape[0] for p in parts) == obp["pitching"].shape[0]


def test_split_fractions_must_sum_to_one(obp: OBP) -> None:
    """A malformed split is an error, not a silent renormalisation."""
    with pytest.raises(ValueError, match="sum to 1"):
        obp["pitching"].split_by_athlete((0.5, 0.2), seed=0)


# ------------------------------------------------------------------ mirroring


def test_mirroring_is_off_by_default_and_touches_only_y(obp: OBP) -> None:
    """`mirror_lefties` negates y for landmarks and GRF only, and never by default."""
    view = obp["pitching"].select(group=["landmark"])
    plain = view.torch()
    mirrored = view.torch(mirror_lefties=True)
    assert torch.equal(plain, view.torch(mirror_lefties=False))

    lefty = view.trials["handedness"].isin({"L", "l", "Left", "left", "LHP"}).to_numpy()
    assert lefty.any(), "expected at least one left-handed athlete"
    y = np.flatnonzero(view.channels["axis"].to_numpy() == "y")
    x = np.flatnonzero(view.channels["axis"].to_numpy() == "x")
    rows = np.flatnonzero(lefty)
    assert torch.allclose(mirrored[rows][:, :, y], -plain[rows][:, :, y], equal_nan=True)
    assert torch.allclose(mirrored[rows][:, :, x], plain[rows][:, :, x], equal_nan=True)
    assert torch.allclose(mirrored[np.flatnonzero(~lefty)], plain[np.flatnonzero(~lefty)], equal_nan=True)


def test_mirroring_leaves_angles_alone(obp: OBP) -> None:
    """Angle conventions are already handedness-adjusted, so they must not be touched."""
    view = obp["pitching"].select(group=["angle"])
    assert torch.allclose(view.torch(), view.torch(mirror_lefties=True), equal_nan=True)


# ---------------------------------------------------------------- single value


def test_value_accepts_frames_and_seconds(obp: OBP) -> None:
    """`value()` takes an int frame index or a float time in seconds."""
    view = obp["pitching"]
    at_anchor = view.value("1031_2", 0.0, "shoulder_angle_x")
    same = view.value("1031_2", view.pre_frames, "shoulder_angle_x")
    assert at_anchor == pytest.approx(same)


def test_value_rejects_unknown_keys(obp: OBP) -> None:
    """Wrong trial or channel is an error, not a NaN."""
    view = obp["pitching"]
    with pytest.raises(KeyError):
        view.value("__nope__", 0.0, "shoulder_angle_x")
    with pytest.raises(KeyError):
        view.value("1031_2", 0.0, "__nope__")


# ------------------------------------------------------ probe-verified constants


#: Measured directly from the `dataset-v1` release on 24/07/2026. These pin the
#: corpus against the upstream data rather than against the code that built it,
#: so a silently-changed release or a regressed builder both show up here.
PITCHING_EXPECTED = {
    "trials": 411,
    "athletes": 100,
    "channels": 286,
    "dead_channels": 7,
    "grf_trials": 403,
    "rate_hz": 360.0,
}
PITCHING_NO_GRF = {"1370_1", "2857_4", "2919_2", "2919_3", "2919_4", "2919_5", "2923_1", "2923_2"}
HITTING_EXPECTED = {"trials": 677, "athletes": 98, "channels": 162, "dead_channels": 7, "rate_hz": 360.0}


def test_pitching_matches_the_probed_release(obp: OBP) -> None:
    """Upstream pitching numbers, measured from `dataset-v1`."""
    view = obp["pitching"]
    index, channels = view.trials, obp.store.channels("pitching")
    assert view.shape[0] == PITCHING_EXPECTED["trials"]
    assert index["athlete_id"].nunique() == PITCHING_EXPECTED["athletes"]
    assert len(channels) == PITCHING_EXPECTED["channels"]
    assert int(channels["constant_zero"].sum()) == PITCHING_EXPECTED["dead_channels"]
    assert int(index["has_grf"].sum()) == PITCHING_EXPECTED["grf_trials"]
    assert view.rate_hz == PITCHING_EXPECTED["rate_hz"]
    assert set(index.loc[~index["has_grf"], "trial_id"]) == PITCHING_NO_GRF
    assert view.anchor_event == "BR_time"


def test_hitting_matches_the_probed_release(obp: OBP) -> None:
    """Upstream hitting numbers. The marker rate is 360 Hz, same as pitching."""
    view = obp["hitting"]
    index, channels = view.trials, obp.store.channels("hitting")
    assert view.shape[0] == HITTING_EXPECTED["trials"]
    assert index["athlete_id"].nunique() == HITTING_EXPECTED["athletes"]
    assert len(channels) == HITTING_EXPECTED["channels"]
    assert int(channels["constant_zero"].sum()) == HITTING_EXPECTED["dead_channels"]
    assert view.rate_hz == HITTING_EXPECTED["rate_hz"]
    assert view.anchor_event == "contact_time"


def test_every_gate_passes(obp: OBP, corpus_path: Path, raw_root: Path | None) -> None:
    """The structural verifier must report a clean corpus."""
    from odynamech import verify_corpus

    tensors = corpus_path if corpus_path.is_dir() else corpus_path.parent
    report = verify_corpus(corpus_path, raw=raw_root, tensors=tensors)
    assert report.ok, "failed gates:\n" + "\n".join(report.failed)
    assert len(report.passed) > 50
