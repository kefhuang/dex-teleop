from __future__ import annotations

import math

import numpy as np

from dex_teleop.core.types import ControllerState, DeviceState, Pose
from dex_teleop.policies.bimanual_clutch import BimanualClutchPolicy


def _controller(
    name: str,
    *,
    position=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0, 1.0),
    grip: bool = False,
    trigger: float = 0.0,
) -> ControllerState:
    return ControllerState(
        name=name,
        pose=Pose(position=position, orientation=orientation),
        buttons={"grip": grip},
        axes={"trigger": trigger},
    )


def _state(
    *,
    left_pos=(0.0, 0.0, 0.0),
    left_orientation=(0.0, 0.0, 0.0, 1.0),
    left_grip=False,
    left_trigger=0.0,
    right_present=True,
    right_pos=(0.0, 0.0, 0.0),
    right_orientation=(0.0, 0.0, 0.0, 1.0),
    right_grip=False,
    right_trigger=0.0,
) -> DeviceState:
    controllers = {
        "left": _controller(
            "left",
            position=left_pos,
            orientation=left_orientation,
            grip=left_grip,
            trigger=left_trigger,
        )
    }
    if right_present:
        controllers["right"] = _controller(
            "right",
            position=right_pos,
            orientation=right_orientation,
            grip=right_grip,
            trigger=right_trigger,
        )
    return DeviceState(timestamp=1.0, controllers=controllers)


def _cfg() -> dict:
    return {
        "controls": {"clutch_button": "grip", "gripper_axis": "trigger"},
        "arms": {
            "left": {"controller": "left", "gripper": "left_gripper"},
            "right": {"controller": "right", "gripper": "right_gripper"},
        },
        "mapping": {
            "position_axes": {"x": "+z", "y": "-x", "z": "+y"},
            "rotation_axes": {"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"},
        },
        "scales": {"position": 1.0, "rotation": 1.0},
        "safety": {
            "max_translation_per_cycle": 0.01,
            "max_rotation_per_cycle": 0.05,
            "position_deadzone": 0.0,
            "rotation_deadzone": 0.0,
            "send_zero_when_idle": True,
        },
    }


def _policy_with(**overrides) -> BimanualClutchPolicy:
    cfg = _cfg()
    for section, values in overrides.items():
        cfg[section].update(values)
    return BimanualClutchPolicy.from_config(cfg)


def _policy() -> BimanualClutchPolicy:
    return _policy_with()


def test_grip_released_emits_disabled_motion_and_gripper_command_from_trigger():
    batch = _policy().update(_state(left_grip=False, left_trigger=0.25))

    assert batch.motions["left"].enabled is False
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))
    assert batch.grippers["left_gripper"].normalized == 0.25


def test_first_grip_press_establishes_anchor_and_emits_enabled_zero_delta():
    batch = _policy().update(_state(left_pos=(1.0, 2.0, 3.0), left_grip=True))

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))
    assert np.allclose(batch.motions["left"].delta_rotation, np.zeros(3))


def test_held_grip_outputs_mapped_scaled_limited_delta_from_anchor():
    policy = _policy()
    policy.update(_state(left_pos=(0.0, 0.0, 0.0), left_grip=True))

    batch = policy.update(_state(left_pos=(0.2, 0.3, 0.4), left_grip=True))

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.array([0.01, -0.01, 0.01]))



def test_left_and_right_clutch_anchors_are_independent():
    policy = _policy()
    policy.update(
        _state(
            left_pos=(1.0, 0.0, 0.0),
            left_grip=True,
            right_pos=(0.0, 0.0, 0.0),
            right_grip=False,
        )
    )

    batch = policy.update(
        _state(
            left_pos=(1.0, 0.2, 0.0),
            left_grip=True,
            right_pos=(0.7, -0.4, 0.3),
            right_grip=True,
        )
    )

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.array([0.0, 0.0, 0.01]))
    assert batch.motions["right"].enabled is True
    assert np.allclose(batch.motions["right"].delta_position, np.zeros(3))
    assert np.allclose(batch.motions["right"].delta_rotation, np.zeros(3))


def test_held_grip_outputs_mapped_scaled_limited_rotation_delta():
    policy = _policy_with(
        mapping={
            "position_axes": {"x": "+x", "y": "+y", "z": "+z"},
            "rotation_axes": {"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"},
        },
        scales={"position": 1.0, "rotation": 1.0},
    )
    policy.update(_state(left_orientation=(0.0, 0.0, 0.0, 1.0), left_grip=True))

    batch = policy.update(
        _state(
            left_orientation=(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            left_grip=True,
        )
    )

    assert batch.motions["left"].enabled is True
    assert np.count_nonzero(batch.motions["left"].delta_rotation) == 1
    assert np.allclose(batch.motions["left"].delta_rotation, np.array([0.0, 0.0, 0.05]))


def test_rotation_deadzone_zeroes_small_rotation_after_mapping_and_scale():
    policy = _policy_with(
        mapping={
            "position_axes": {"x": "+x", "y": "+y", "z": "+z"},
            "rotation_axes": {"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"},
        },
        safety={
            "max_translation_per_cycle": 0.01,
            "max_rotation_per_cycle": 0.05,
            "position_deadzone": 0.0,
            "rotation_deadzone": 0.02,
            "send_zero_when_idle": True,
        },
    )
    policy.update(_state(left_orientation=(0.0, 0.0, 0.0, 1.0), left_grip=True))

    small_yaw = 0.01
    batch = policy.update(
        _state(
            left_orientation=(0.0, 0.0, math.sin(small_yaw / 2.0), math.cos(small_yaw / 2.0)),
            left_grip=True,
        )
    )

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_rotation, np.zeros(3))


def test_position_deadzone_zeroes_small_position_delta_after_mapping_and_scale():
    policy = _policy_with(
        mapping={
            "position_axes": {"x": "+x", "y": "+y", "z": "+z"},
            "rotation_axes": {"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"},
        },
        scales={"position": 0.5, "rotation": 1.0},
        safety={
            "max_translation_per_cycle": 0.01,
            "max_rotation_per_cycle": 0.05,
            "position_deadzone": 0.02,
            "rotation_deadzone": 0.0,
            "send_zero_when_idle": True,
        },
    )
    policy.update(_state(left_pos=(0.0, 0.0, 0.0), left_grip=True))

    batch = policy.update(_state(left_pos=(0.03, 0.0, 0.0), left_grip=True))

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))


def test_grip_release_clears_anchor_and_disables_motion():
    policy = _policy()
    policy.update(_state(left_pos=(0.0, 0.0, 0.0), left_grip=True))
    policy.update(_state(left_pos=(0.2, 0.3, 0.4), left_grip=False))

    batch = policy.update(_state(left_pos=(0.2, 0.3, 0.4), left_grip=True))

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))


def test_missing_controller_disables_bound_arm_and_zeroes_gripper():
    policy = _policy()
    policy.update(_state(left_grip=True, right_present=True, right_grip=True, right_trigger=0.8))
    state = DeviceState(timestamp=2.0, controllers={})

    batch = policy.update(state)

    assert batch.motions["left"].enabled is False
    assert batch.motions["right"].enabled is False
    assert batch.grippers["left_gripper"].normalized == 0.0
    assert batch.grippers["right_gripper"].normalized == 0.0


def test_trigger_command_clamps_values_outside_unit_interval():
    batch = _policy().update(_state(left_trigger=1.4, right_trigger=-0.2))

    assert batch.grippers["left_gripper"].normalized == 1.0
    assert batch.grippers["right_gripper"].normalized == 0.0
