"""Channel schema: deterministic site parsing, canonical ordering, and hashing."""

import hashlib
import re
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical table order.  The channel axis is sorted by (this order, name), so a
# rebuild always produces the same layout.
TABLE_ORDER: tuple[str, ...] = (
    "joint_angles",
    "joint_velos",
    "landmarks",
    "forces_moments",
    "energy_flow",
    "force_plate",
)

# Join key and event columns, per gesture.  Discovered by probe on 24/07/2026;
# hitting carries three events where pitching carries six.
GESTURE_KEY: dict[str, str] = {"pitching": "session_pitch", "hitting": "session_swing"}
GESTURE_EVENTS: dict[str, tuple[str, ...]] = {
    "pitching": ("pkh_time", "fp_10_time", "fp_100_time", "MER_time", "BR_time", "MIR_time"),
    "hitting": ("fp_10_time", "fp_100_time", "contact_time"),
}
GESTURE_ANCHOR: dict[str, str] = {"pitching": "BR_time", "hitting": "contact_time"}

# Tables present per gesture.  Hitting ships no forces_moments and no energy_flow.
GESTURE_TABLES: dict[str, tuple[str, ...]] = {
    "pitching": ("joint_angles", "joint_velos", "landmarks", "forces_moments", "energy_flow", "force_plate"),
    "hitting": ("joint_angles", "joint_velos", "landmarks", "force_plate"),
}

# Units, from the module READMEs.  Empty where genuinely unknown rather than guessed.
GROUP_UNIT: dict[str, str] = {
    "angle": "deg",
    "velo": "deg/s",
    "landmark": "m",
    "force": "N",
    "moment": "Nm",
    "energy": "W",  # segment/joint power; the *_generated channels integrate to J
    "grf": "N",
}

# Segments a shoulder/hip kinetic channel can be resolved against.  Taken from the
# actual forces_moments column names, which side-qualify the thighs and the glove
# upper arm — a wider vocabulary than plan.md §5 anticipates.
REF_SEGMENTS: tuple[str, ...] = (
    "pelvis",
    "thorax",
    "upper_arm",
    "glove_upper_arm",
    "rear_thigh",
    "lead_thigh",
)


class UnclassifiedChannel(ValueError):
    """A source column the site parser refuses to guess at."""


# Columns that are genuinely unclassifiable and are carried with an explicit
# reason rather than silently defaulted.  Empty: every column in dataset-v1
# matches a rule below.
UNCLASSIFIED_ALLOW: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """One scalar signal, resolved to its semantic coordinates."""

    name: str
    table: str
    group: str
    site: str
    subtype: str
    axis: str
    ref_segment: str

    @property
    def semantic(self) -> tuple[str, str, str, str, str]:
        """The tuple harmonisation keys on; unique within a gesture."""
        return (self.group, self.site, self.subtype, self.axis, self.ref_segment)


def _axis_split(name: str) -> tuple[str, str]:
    """Split a trailing `_x`/`_y`/`_z` off a column name."""
    if len(name) > 2 and name[-2] == "_" and name[-1] in "xyz":
        return name[:-2], name[-1]
    return name, ""


# Hitting landmarks use terse anatomical codes: side letter + joint letter + "jc".
_HIT_JOINT = {"h": "hip", "s": "shoulder", "e": "elbow", "k": "knee", "a": "ankle", "w": "wrist"}
_HIT_SIDE = {"l": "left", "r": "right"}


def _parse_angle(stem: str) -> ChannelSpec | None:
    """`{site}_angle` — identical convention in both gestures."""
    if stem.endswith("_angle"):
        return ChannelSpec("", "", "angle", stem[: -len("_angle")], "", "", "")
    return None


def _parse_velo(stem: str) -> ChannelSpec | None:
    """`{site}_velo` (pitching) or `{site}_[global_]angular_velocity` (hitting)."""
    if stem.endswith("_velo"):
        return ChannelSpec("", "", "velo", stem[: -len("_velo")], "", "", "")
    if stem.endswith("_global_angular_velocity"):
        return ChannelSpec("", "", "velo", stem[: -len("_global_angular_velocity")], "global", "", "")
    if stem.endswith("_angular_velocity"):
        return ChannelSpec("", "", "velo", stem[: -len("_angular_velocity")], "", "", "")
    return None


def _parse_landmark(stem: str) -> ChannelSpec | None:
    """Joint centres and tracked points; two different naming conventions."""
    # Shared by both gestures.
    if stem == "centerofmass":
        return ChannelSpec("", "", "landmark", "centerofmass", "", "", "")
    if stem.startswith("thorax_"):
        sub = stem[len("thorax_") :]
        if sub in ("ap", "dist", "prox"):
            return ChannelSpec("", "", "landmark", "thorax", sub, "", "")
        return None

    # Hitting: bat-tracking points, no pitching counterpart.
    if stem in ("blast_hand", "sweet_spot"):
        return ChannelSpec("", "", "landmark", stem, "", "", "")

    # Hitting: `lhjc`, `rajc`, … and the spelled-out `left_hip` / `right_hip`.
    m = re.fullmatch(r"([lr])([hsekaw])jc", stem)
    if m:
        return ChannelSpec("", "", "landmark", f"{_HIT_SIDE[m[1]]}_{_HIT_JOINT[m[2]]}", "", "", "")
    m = re.fullmatch(r"(left|right)_hip", stem)
    if m:
        # The non-`jc` hip landmark: the segment origin, not the joint centre.
        return ChannelSpec("", "", "landmark", f"{m[1]}_hip", "origin", "", "")

    # Pitching: role naming, with or without a `_jc` suffix.
    if stem.endswith("_jc"):
        return ChannelSpec("", "", "landmark", stem[: -len("_jc")], "", "", "")
    if stem in ("rear_hip", "lead_hip"):
        return ChannelSpec("", "", "landmark", stem, "origin", "", "")
    return None


def _parse_kinetic(stem: str) -> ChannelSpec | None:
    """`{site}[_{ref_segment}]_{force|moment}` — the double-resolved convention."""
    for group in ("moment", "force"):
        suffix = f"_{group}"
        if not stem.endswith(suffix):
            continue
        site = stem[: -len(suffix)]
        # Longest reference segment first, so `glove_upper_arm` wins over `upper_arm`.
        for ref in sorted(REF_SEGMENTS, key=len, reverse=True):
            if site.endswith(f"_{ref}"):
                return ChannelSpec("", "", group, site[: -len(ref) - 1], "", "", ref)
        return ChannelSpec("", "", group, site, "", "", "")
    return None


def _parse_energy(stem: str) -> ChannelSpec | None:
    """Joint energy (`_energy_*`) and segment power (`_seg_pwr`) channels."""
    if stem.endswith("_energy_transfer_stp"):
        return ChannelSpec("", "", "energy", stem[: -len("_energy_transfer_stp")], "transfer_stp", "", "")
    if stem.endswith("_energy_transfer_jfp"):
        return ChannelSpec("", "", "energy", stem[: -len("_energy_transfer_jfp")], "transfer_jfp", "", "")
    if stem.endswith("_energy_generated"):
        return ChannelSpec("", "", "energy", stem[: -len("_energy_generated")], "generated", "", "")

    if stem.endswith("_seg_pwr"):
        body = stem[: -len("_seg_pwr")]
        # Pelvis power resolved against a neighbouring segment.
        m = re.fullmatch(r"pelvis_(leadhip|rearhip|thorax)", body)
        if m:
            return ChannelSpec("", "", "energy", "pelvis", "seg_pwr", "", m[1])
        # Thorax distal power on the glove side; must precede the plain dist rule.
        if body == "thorax_dist_glove":
            return ChannelSpec("", "", "energy", "thorax", "dist_glove_seg_pwr", "", "")
        for end in ("dist", "prox"):
            if body.endswith(f"_{end}"):
                return ChannelSpec("", "", "energy", body[: -len(end) - 1], f"{end}_seg_pwr", "", "")
    return None


def _parse_grf(stem: str) -> ChannelSpec | None:
    """`{rear|lead}_force` from the force plate — ground reaction, not joint kinetics."""
    if stem in ("rear_force", "lead_force"):
        # Named for the plate, not a joint: the measurement is the foot-ground
        # interface, and calling it `lead_ankle` would assert anatomy not measured.
        return ChannelSpec("", "", "grf", f"{stem[: -len('_force')]}_force_plate", "", "", "")
    return None


_PARSERS = {
    "joint_angles": (_parse_angle,),
    "joint_velos": (_parse_velo,),
    "landmarks": (_parse_landmark,),
    "forces_moments": (_parse_kinetic,),
    "energy_flow": (_parse_energy,),
    "force_plate": (_parse_grf,),
}


def parse_channel(name: str, table: str) -> ChannelSpec:
    """Resolve one source column to its semantic coordinates, or raise."""
    stem, axis = _axis_split(name)
    for parser in _PARSERS[table]:
        spec = parser(stem)
        if spec is not None:
            return ChannelSpec(name, table, spec.group, spec.site, spec.subtype, axis, spec.ref_segment)
    if name in UNCLASSIFIED_ALLOW:
        return ChannelSpec(name, table, "unclassified", "unknown", "", axis, "")
    raise UnclassifiedChannel(
        f"cannot classify column {name!r} in table {table!r}. "
        f"Add a parsing rule, or add it to UNCLASSIFIED_ALLOW with a reason."
    )


def measurement_columns(columns: Iterable[str], gesture: str) -> list[str]:
    """Source columns that carry a measurement, i.e. neither key nor event."""
    drop = {GESTURE_KEY[gesture], "time", *GESTURE_EVENTS[gesture]}
    return [c for c in columns if c not in drop]


def build_schema(
    table_columns: Mapping[str, Sequence[str]],
    gesture: str,
    constant_zero: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Build the ordered channel table for one gesture.

    `table_columns` maps table name to that table's full column list.  Ordering is
    `(TABLE_ORDER position, name)`, so it is stable across rebuilds.
    """
    specs: list[ChannelSpec] = []
    for table in TABLE_ORDER:
        if table not in table_columns:
            continue
        for name in sorted(measurement_columns(table_columns[table], gesture)):
            specs.append(parse_channel(name, table))

    frame = pd.DataFrame([asdict(s) for s in specs])
    frame.insert(0, "index", np.arange(len(frame), dtype=np.int64))
    frame["unit"] = frame["group"].map(GROUP_UNIT).fillna("")
    zeros = constant_zero or {}
    frame["constant_zero"] = frame["name"].map(lambda n: bool(zeros.get(n, False)))

    dup = frame.duplicated(subset=["group", "site", "subtype", "axis", "ref_segment"], keep=False)
    if bool(dup.any()):
        raise ValueError(
            "semantic tuple (group, site, subtype, axis, ref_segment) is not unique; "
            f"colliding channels:\n{frame.loc[dup, ['name', 'group', 'site', 'subtype', 'axis', 'ref_segment']]}"
        )
    return frame[["index", "name", "table", "group", "site", "subtype", "axis", "ref_segment", "constant_zero", "unit"]]


def channel_hash(names: Sequence[str]) -> str:
    """`sha256` of the newline-joined channel names, in axis order."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def cross_tab(frame: pd.DataFrame) -> str:
    """Site × group cross-tab, for eyeballing the mapping in one screen."""
    tab = pd.crosstab(frame["site"], frame["group"])
    tab["TOTAL"] = tab.sum(axis=1)
    return tab.to_string()
