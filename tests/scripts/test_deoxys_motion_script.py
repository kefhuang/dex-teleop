from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "test_deoxys_motion.py"
    spec = importlib.util.spec_from_file_location("test_deoxys_motion_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direction_delta_vectors_are_two_centimeters_by_default():
    module = load_module()

    deltas = module.build_direction_deltas(0.02)

    assert np.allclose(deltas["up"], [0.0, 0.0, 0.02])
    assert np.allclose(deltas["down"], [0.0, 0.0, -0.02])
    assert np.allclose(deltas["left"], [0.0, 0.02, 0.0])
    assert np.allclose(deltas["right"], [0.0, -0.02, 0.0])
    assert np.allclose(deltas["forward"], [0.02, 0.0, 0.0])
    assert np.allclose(deltas["back"], [-0.02, 0.0, 0.0])


def test_build_motion_action_keeps_rotation_and_gripper_slot_zero():
    module = load_module()

    action = module.build_motion_action(np.array([0.01, -0.02, 0.03]))

    assert action == [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.0]


def test_clip_delta_norm_limits_large_vector_without_changing_direction():
    module = load_module()

    clipped = module.clip_delta_norm(np.array([0.03, 0.04, 0.0]), 0.025)

    assert np.allclose(clipped, [0.015, 0.02, 0.0])


def test_clip_delta_norm_leaves_small_vector_unchanged():
    module = load_module()

    clipped = module.clip_delta_norm(np.array([0.003, 0.004, 0.0]), 0.02)

    assert np.allclose(clipped, [0.003, 0.004, 0.0])


def test_direction_sequence_accepts_all_aliases_and_preserves_order():
    module = load_module()

    directions = module.parse_direction_sequence("up,left,forward,back")

    assert directions == ["up", "left", "forward", "back"]


def test_direction_sequence_rejects_unknown_direction():
    module = load_module()

    try:
        module.parse_direction_sequence("up,diagonal")
    except ValueError as exc:
        assert "diagonal" in str(exc)
    else:
        raise AssertionError("expected ValueError")
