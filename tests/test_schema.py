"""Channel-schema parsing and byte round-trips. Data-free: these run everywhere."""

import pytest

from odynamech.schema import channel_hash
from odynamech.schema import parse_channel
from odynamech.schema import UnclassifiedChannel
from odynamech.store import bytes_to_tensor
from odynamech.store import tensor_to_bytes


def test_parser_resolves_the_awkward_conventions() -> None:
    """The naming conventions that differ between gestures must all resolve."""
    cases = [
        ("shoulder_upper_arm_moment_z", "forces_moments", ("moment", "shoulder", "", "z", "upper_arm")),
        (
            "glove_shoulder_glove_upper_arm_force_x",
            "forces_moments",
            ("force", "glove_shoulder", "", "x", "glove_upper_arm"),
        ),
        ("rear_hip_pelvis_moment_y", "forces_moments", ("moment", "rear_hip", "", "y", "pelvis")),
        ("elbow_energy_transfer_stp", "energy_flow", ("energy", "elbow", "transfer_stp", "", "")),
        ("elbow_energy_transfer_jfp", "energy_flow", ("energy", "elbow", "transfer_jfp", "", "")),
        ("elbow_energy_generated", "energy_flow", ("energy", "elbow", "generated", "", "")),
        ("thorax_dist_glove_seg_pwr", "energy_flow", ("energy", "thorax", "dist_glove_seg_pwr", "", "")),
        ("thorax_dist_seg_pwr", "energy_flow", ("energy", "thorax", "dist_seg_pwr", "", "")),
        ("pelvis_leadhip_seg_pwr", "energy_flow", ("energy", "pelvis", "seg_pwr", "", "leadhip")),
        ("rear_ankle_jc_x", "landmarks", ("landmark", "rear_ankle", "", "x", "")),
        ("lead_hip_y", "landmarks", ("landmark", "lead_hip", "origin", "y", "")),
        ("thorax_prox_z", "landmarks", ("landmark", "thorax", "prox", "z", "")),
        ("rajc_x", "landmarks", ("landmark", "right_ankle", "", "x", "")),
        ("lhjc_y", "landmarks", ("landmark", "left_hip", "", "y", "")),
        ("left_hip_z", "landmarks", ("landmark", "left_hip", "origin", "z", "")),
        ("sweet_spot_x", "landmarks", ("landmark", "sweet_spot", "", "x", "")),
        ("elbow_velo_x", "joint_velos", ("velo", "elbow", "", "x", "")),
        ("lead_elbow_angular_velocity_x", "joint_velos", ("velo", "lead_elbow", "", "x", "")),
        ("lead_hand_global_angular_velocity_y", "joint_velos", ("velo", "lead_hand", "global", "y", "")),
        ("rear_force_z", "force_plate", ("grf", "rear_force_plate", "", "z", "")),
    ]
    for name, table, expected in cases:
        spec = parse_channel(name, table)
        assert (spec.group, spec.site, spec.subtype, spec.axis, spec.ref_segment) == expected, name


def test_parser_raises_rather_than_guessing() -> None:
    """An unknown column must abort the build, not fall back to site='unknown'."""
    with pytest.raises(UnclassifiedChannel):
        parse_channel("wibble_flange_q", "joint_angles")


def test_channel_hash_is_order_sensitive() -> None:
    """Reordering the channel axis must change the hash."""
    assert channel_hash(["a", "b"]) != channel_hash(["b", "a"])


def test_byte_tensor_round_trip() -> None:
    """Gate 8's primitive: bytes survive the uint8 tensor round-trip exactly."""
    payload = bytes(range(256)) * 37
    assert tensor_to_bytes(bytes_to_tensor(payload)) == payload
