"""Paths, upstream coordinates, and the data-licence notice this package must surface."""

import os
from pathlib import Path

# ---------------------------------------------------------------------- upstream

#: The only URL in this package that has a default: the upstream release assets.
#: Every other acquisition path takes a caller-supplied URL with no default.
DRIVELINE_RELEASE_BASE: str = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1"

#: Raw git contents, for the scalar metadata and points-of-interest tables, which
#: live in the repository rather than in the release assets.
DRIVELINE_RAW_BASE: str = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main"

DRIVELINE_REPO: str = "https://github.com/drivelineresearch/openbiomechanics"
DRIVELINE_RELEASE_TAG: str = "dataset-v1"

# ------------------------------------------------------------------------ layout

GESTURES: tuple[str, ...] = ("pitching", "hitting")

#: Full-signal tables published per gesture. Hitting ships no `forces_moments`
#: and no `energy_flow`; that asymmetry is in the upstream release, not here.
GESTURE_ASSETS: dict[str, tuple[str, ...]] = {
    "pitching": ("joint_angles", "joint_velos", "landmarks", "forces_moments", "energy_flow", "force_plate"),
    "hitting": ("joint_angles", "joint_velos", "landmarks", "force_plate"),
}

#: Scalar tables fetched from the git tree, per gesture, as (module, filename).
GESTURE_SCALARS: dict[str, str] = {"pitching": "baseball_pitching", "hitting": "baseball_hitting"}

#: The C3D archives are deliberately never fetched: raw marker trajectories,
#: ~0.6 GB, and fully redundant with the `landmarks` tables.
EXCLUDED_ASSETS: tuple[str, ...] = ("pitching_c3d", "hitting_c3d")

PACK_NAME: str = "odynamech_corpus.safetensors"

# ------------------------------------------------------------------------- cache


def cache_root() -> Path:
    """Root of the on-disk cache; `$ODYNAMECH_HOME`, else XDG, else `~/.cache`."""
    override = os.environ.get("ODYNAMECH_HOME")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "odynamech"


def raw_dir(root: Path | None = None) -> Path:
    """Where fetched upstream CSVs land."""
    return (root or cache_root()) / "raw"


def tensors_dir(root: Path | None = None) -> Path:
    """Where built stores and the single-file pack land."""
    return (root or cache_root()) / "tensors"


# ------------------------------------------------------------------------ licence

#: Emitted once per cache directory on first acquisition, and written to disk
#: beside the data. The exclusion in the third paragraph is an addition to plain
#: CC BY-NC-SA 4.0 and is easy to miss, which is precisely why it is restated here.
DATA_NOTICE: str = f"""\
OpenBiomechanics Project (OBP) data — third-party, separately licensed
=====================================================================

Source : {DRIVELINE_REPO}
Release: {DRIVELINE_RELEASE_TAG}
Owner  : Driveline Baseball / Driveline Research & Development

The DATA is licensed CC BY-NC-SA 4.0 — Attribution, NonCommercial, ShareAlike:
<https://creativecommons.org/licenses/by-nc-sa/4.0/>.

Beyond the plain CC terms, the upstream project additionally EXCLUDES use by
anyone affiliated with a professional sports organisation or with a
financial-analysis firm. Confirm your own eligibility against the upstream terms
before use; `odynamech` cannot and does not check this for you.

Consequences that bite in practice:

  * NonCommercial  — no commercial use of the data or of anything derived from it.
  * ShareAlike     — a corpus built by `odynamech` is an ADAPTATION. If you pass it
                     on, it goes out under CC BY-NC-SA 4.0 and your recipients
                     inherit both the rights and these obligations.
  * Attribution    — credit Driveline, link the source, and state that changes were
                     made. `odynamech` makes material changes: it decimates the
                     force plate to the marker clock, anchors and crops trials to a
                     common window, edge-pads, and reorders the channel axis.

`odynamech` itself (the CODE) is MIT-licensed and contains NO DATA. It only
automates acquiring the data from a source you nominate. The two licences are
separate; neither one relaxes the other.

This package exists for dynamical-systems and machine-learning research: the OBP
release is a corpus of many bodies executing the same act with markedly different
parameters and initial conditions, which makes it a useful, non-artificial testbed
for models of dynamical evolution under inter-individual variability.
"""

ATTRIBUTION: str = (
    "OpenBiomechanics Project by Driveline Baseball "
    f"({DRIVELINE_REPO}), release {DRIVELINE_RELEASE_TAG}, "
    "licensed CC BY-NC-SA 4.0 with an additional professional-sports and "
    "financial-analysis exclusion. Adapted by `odynamech`: force plate decimated "
    "to the marker clock, trials anchored and cropped to a common window, "
    "edge-padded, channel axis reordered."
)

DATA_LICENCE_SPDX: str = "CC-BY-NC-SA-4.0"
CODE_LICENCE_SPDX: str = "MIT"
