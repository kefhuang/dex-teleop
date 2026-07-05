from __future__ import annotations

import math

import numpy as np

from dex_teleop.devices.oculus import OculusDevice


def test_grip_prefers_left_digital_false_over_small_analog_value():
    device = OculusDevice()

    controllers = device._convert({}, {"LG": False, "leftGrip": (0.1,)})

    assert controllers["left"].buttons["grip"] is False


def test_grip_prefers_left_digital_true_over_analog_value():
    device = OculusDevice()

    controllers = device._convert({}, {"LG": True, "leftGrip": (0.0,)})

    assert controllers["left"].buttons["grip"] is True


def test_grip_prefers_right_digital_false_over_small_analog_value():
    device = OculusDevice()

    controllers = device._convert({}, {"RG": False, "rightGrip": (0.1,)})

    assert controllers["right"].buttons["grip"] is False


def test_grip_prefers_right_digital_true_over_analog_value():
    device = OculusDevice()

    controllers = device._convert({}, {"RG": True, "rightGrip": (0.0,)})

    assert controllers["right"].buttons["grip"] is True


def test_grip_analog_requires_explicit_threshold_when_digital_missing():
    device = OculusDevice()

    controllers = device._convert({}, {"leftGrip": (0.49,), "rightGrip": (0.5,)})

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
