"""Cross-gesture channel harmonisation: semantic-tuple intersection, fully reversible."""

import logging
from collections.abc import Mapping
from collections.abc import Sequence

import pandas as pd

log = logging.getLogger(__name__)

SEMANTIC_KEY: tuple[str, ...] = ("group", "site", "subtype", "axis", "ref_segment")

# --------------------------------------------------------------------------------
# Site correspondences that are NOT decidable from the column names alone.
#
# Pitching frames the upper body by role relative to the throwing arm (throwing vs
# glove); hitting frames it by role relative to the pitcher (lead vs rear).  In
# `landmarks`, hitting additionally switches to anatomical left/right.  Whether a
# pitching `shoulder` corresponds to a hitting `lead_shoulder` or `rear_shoulder`
# — and whether a hitting `left_knee` is the lead or the rear knee — is a
# handedness-and-role question, not a string question.
#
# Both tables below are therefore data, not code: a human can read them, and
# nothing in the build consults them unless explicitly asked.  Default: EXCLUDED.
# --------------------------------------------------------------------------------

# Arms. Deliberately left empty: the throwing/glove <-> lead/rear correspondence is
# not derivable from any column in the release, and guessing it would silently pair
# unrelated channels.  Populate only with external justification.
ARM_SITE_MAP: dict[str, str] = {}

# Landmarks. Hitting uses anatomical sides; the mapping to pitching's roles is
# determined per trial by `hitter_side` / `p_throws`, not globally.  A right-handed
# hitter stands with the left side toward the pitcher, so left == lead; a
# right-handed pitcher strides onto the left leg, so left == lead there too.
# Applying this is opt-in because it is a per-trial channel permutation, not the
# channel-axis-only restriction the corpus otherwise guarantees.
HANDED_SIDE_SITES: tuple[str, ...] = (
    "hip",
    "knee",
    "ankle",
    "shoulder",
    "elbow",
    "wrist",
)


def _semantic_frame(channels: pd.DataFrame) -> pd.DataFrame:
    """Index a channel table by its semantic tuple."""
    frame = channels.copy()
    for col in SEMANTIC_KEY:
        frame[col] = frame[col].fillna("").astype("str")
    return frame


def common_channels(per_gesture: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Intersect two or more gestures on the semantic tuple, keeping back-pointers.

    The result carries `native_index_{gesture}` and `native_name_{gesture}` for every
    gesture, so a harmonised selection maps back to either native channel axis
    losslessly. Nothing is dropped from storage and nothing is imputed.
    """
    gestures = list(per_gesture)
    frames = {g: _semantic_frame(df) for g, df in per_gesture.items()}

    for gesture, frame in frames.items():
        dup = frame.duplicated(subset=list(SEMANTIC_KEY), keep=False)
        if bool(dup.any()):
            raise ValueError(
                f"{gesture}: semantic tuple is not unique, so a harmonised entry could not map "
                f"to exactly one native channel:\n{frame.loc[dup, ['name', *SEMANTIC_KEY]]}"
            )

    merged = frames[gestures[0]][[*SEMANTIC_KEY, "index", "name", "unit", "table", "constant_zero"]].rename(
        columns={"index": f"native_index_{gestures[0]}", "name": f"native_name_{gestures[0]}"}
    )
    for gesture in gestures[1:]:
        right = frames[gesture][[*SEMANTIC_KEY, "index", "name"]].rename(
            columns={"index": f"native_index_{gesture}", "name": f"native_name_{gesture}"}
        )
        merged = merged.merge(right, on=list(SEMANTIC_KEY), how="inner")

    dropped = {g: len(frames[g]) - len(merged) for g in gestures}
    log.info("harmonised %d channels; dropped per gesture: %s", len(merged), dropped)
    return merged.sort_values(list(SEMANTIC_KEY)).reset_index(drop=True)


def ambiguous_sites(per_gesture: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Sites present in one gesture but not the other, i.e. the undecidable ones."""
    sets = {g: set(df["site"].unique()) for g, df in per_gesture.items()}
    shared = set.intersection(*sets.values())
    return {g: sorted(s - shared) for g, s in sets.items()}


def harmonise(
    per_gesture: Mapping[str, pd.DataFrame],
    resolve_handedness: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build the common channel set. Handedness resolution is opt-in and warned about.

    With `resolve_handedness=False` (the default) the arms and the anatomically-sided
    landmarks are excluded, exactly as the site tables above document. With it set,
    the caller is opting into a per-trial channel permutation and must supply the
    per-trial sides at materialisation time.
    """
    common = common_channels(per_gesture)
    excluded = ambiguous_sites(per_gesture)
    if not resolve_handedness:
        log.warning(
            "harmonisation excluded %d ambiguous sites (arms, and left/right landmarks): %s. "
            "These require a handedness-and-role decision that is not derivable from the column "
            "names; see ARM_SITE_MAP and HANDED_SIDE_SITES in odynamech.harmonise.",
            sum(len(v) for v in excluded.values()),
            excluded,
        )
    elif not ARM_SITE_MAP:
        log.warning(
            "resolve_handedness=True, but ARM_SITE_MAP is empty, so the arms stay excluded. "
            "Populate it explicitly rather than letting the code guess."
        )
    return common, excluded


def native_indices(common: pd.DataFrame, gesture: str) -> Sequence[int]:
    """Back-pointers from a harmonised selection to one gesture's native channel axis."""
    return common[f"native_index_{gesture}"].to_numpy().tolist()


def round_trip_ok(common: pd.DataFrame, per_gesture: Mapping[str, pd.DataFrame]) -> bool:
    """Whether native -> harmonised -> native is the identity on the common set."""
    for gesture, channels in per_gesture.items():
        idx = native_indices(common, gesture)
        back = channels.set_index("index").loc[idx]
        if not (back["name"].to_numpy() == common[f"native_name_{gesture}"].to_numpy()).all():
            return False
        for col in SEMANTIC_KEY:
            if not (back[col].fillna("").astype("str").to_numpy() == common[col].to_numpy()).all():
                return False
    return True
