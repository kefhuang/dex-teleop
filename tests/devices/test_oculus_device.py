from __future__ import annotations

import math

import numpy as np

from dex_teleop.core.types import DeviceState
from dex_teleop.devices.oculus import OculusDevice
from dex_teleop.policies.bimanual_clutch import BimanualClutchPolicy


def _transform(position=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def test_convert_omits_controllers_when_transforms_are_missing_even_with_grip_true():
    device = OculusDevice()

    controllers = device._convert({}, {"LG": True, "RG": True})

    assert "left" not in controllers
    assert "right" not in controllers


def test_convert_includes_only_controller_with_present_transform():
    device = OculusDevice()

    controllers = device._convert({"l": _transform((1.0, 2.0, 3.0))}, {"LG": True, "RG": True})

    assert set(controllers) == {"left"}
    assert controllers["left"].buttons["grip"] is True
    assert np.allclose(controllers["left"].pose.position, [1.0, 2.0, 3.0])


def test_grip_prefers_left_digital_false_over_small_analog_value():
    device = OculusDevice()

    controllers = device._convert({"l": _transform()}, {"LG": False, "leftGrip": (0.1,)})

    assert controllers["left"].buttons["grip"] is False


def test_grip_prefers_left_digital_true_over_analog_value():
    device = OculusDevice()

    controllers = device._convert({"l": _transform()}, {"LG": True, "leftGrip": (0.0,)})

    assert controllers["left"].buttons["grip"] is True


def test_grip_prefers_right_digital_false_over_small_analog_value():
    device = OculusDevice()

    controllers = device._convert({"r": _transform()}, {"RG": False, "rightGrip": (0.1,)})

    assert controllers["right"].buttons["grip"] is False


def test_grip_prefers_right_digital_true_over_analog_value():
    device = OculusDevice()

    controllers = device._convert({"r": _transform()}, {"RG": True, "rightGrip": (0.0,)})

    assert controllers["right"].buttons["grip"] is True


def test_grip_analog_requires_explicit_threshold_when_digital_missing():
    device = OculusDevice()

    controllers = device._convert(
        {"l": _transform(), "r": _transform()},
        {"leftGrip": (0.49,), "rightGrip": (0.5,)},
    )

    assert controllers["left"].buttons["grip"] is False
    assert controllers["right"].buttons["grip"] is True


def test_pose_from_matrix_converts_rotation_to_xyzw_quaternion():
    device = OculusDevice()
    yaw = math.pi / 2.0
    matrix = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0, 1.0],
            [math.sin(yaw), math.cos(yaw), 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    pose = device._pose_from_matrix(matrix)
    orientation = pose.orientation
    if orientation[3] < 0.0:
        orientation = -orientation

    assert np.allclose(pose.position, [1.0, 2.0, 3.0])
    assert np.allclose(
        orientation,
        [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
    )


def test_policy_disables_active_arm_when_oculus_controller_transform_disappears_with_grip_true():
    device = OculusDevice()
    policy = BimanualClutchPolicy.from_config(
        {
            "controls": {"clutch_button": "grip", "gripper_axis": "trigger"},
            "arms": {"left": {"controller": "left", "gripper": "left_gripper"}},
            "mapping": {
                "position_axes": {"x": "+x", "y": "+y", "z": "+z"},
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
    )
    policy.update(
        DeviceState(
            timestamp=1.0,
            controllers=device._convert({"l": _transform((0.0, 0.0, 0.0))}, {"LG": True}),
        )
    )

    batch = policy.update(
        DeviceState(
            timestamp=2.0,
            controllers=device._convert({}, {"LG": True}),
        )
    )

    assert batch.motions["left"].enabled is False
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))
