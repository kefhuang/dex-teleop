from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dex_teleop.core.types import ControllerAction, DeviceState
from dex_teleop.devices.oculus import OculusDevice


def _transform(position=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _yaw_transform(yaw, position=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = _transform(position)
    matrix[:3, :3] = [
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ]
    return matrix


class BufferedReader:
    def __init__(self, samples):
        self.samples = list(samples)

    def get_samples_since(self, sequence):
        latest = self.samples[-1][0] if self.samples else 0
        return latest, [sample for sample in self.samples if sample[0] > sequence]

    def get_transformations_and_buttons(self):
        if not self.samples:
            return {}, {}
        return self.samples[-1][2], self.samples[-1][3]


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


def test_convert_maps_left_x_and_right_a_to_home_action():
    device = OculusDevice()

    controllers = device._convert(
        {"l": _transform(), "r": _transform()},
        {"X": True, "A": True},
    )

    assert controllers["left"].action(ControllerAction.HOME) is True
    assert controllers["right"].action(ControllerAction.HOME) is True


def test_convert_does_not_cross_map_home_buttons_between_controller_sides():
    device = OculusDevice()

    x_only = device._convert(
        {"l": _transform(), "r": _transform()}, {"X": True}
    )
    a_only = device._convert(
        {"l": _transform(), "r": _transform()}, {"A": True}
    )

    assert x_only["left"].action(ControllerAction.HOME) is True
    assert x_only["right"].action(ControllerAction.HOME) is False
    assert a_only["left"].action(ControllerAction.HOME) is False
    assert a_only["right"].action(ControllerAction.HOME) is True


def test_convert_maps_left_y_and_right_b_to_only_their_calibrate_actions():
    device = OculusDevice()

    y_only = device._convert(
        {"l": _transform(), "r": _transform()}, {"Y": True}
    )
    b_only = device._convert(
        {"l": _transform(), "r": _transform()}, {"B": True}
    )

    assert y_only["left"].action(ControllerAction.CALIBRATE) is True
    assert y_only["right"].action(ControllerAction.CALIBRATE) is False
    assert b_only["left"].action(ControllerAction.CALIBRATE) is False
    assert b_only["right"].action(ControllerAction.CALIBRATE) is True


@pytest.mark.parametrize(
    "invalid_transform",
    [
        np.zeros((4, 4)),
        np.full((4, 4), np.nan),
        np.eye(3),
        np.diag([1.0, 1.0, -1.0, 1.0]),
    ],
)
def test_convert_omits_controller_with_invalid_transform(invalid_transform):
    device = OculusDevice()

    controllers = device._convert({"l": invalid_transform}, {"LG": True})

    assert controllers == {}


@pytest.mark.parametrize("invalid_timeout", [0.0, np.nan, np.inf])
def test_oculus_device_rejects_invalid_stale_timeout(invalid_timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        OculusDevice(stale_timeout=invalid_timeout)


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
    orientation = pose.quaternion_xyzw
    if orientation[3] < 0.0:
        orientation = -orientation

    assert np.allclose(pose.position, [1.0, 2.0, 3.0])
    assert np.allclose(
        orientation,
        [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
    )


@pytest.mark.parametrize("roll", [0.0, 1.2, -2.4])
def test_pointer_direction_recovers_apk_green_ray_regardless_of_roll(roll):
    device = OculusDevice()
    binding_angle = math.pi / 3.0
    pointer_rotation = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    grip_from_pointer = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(binding_angle), -math.sin(binding_angle)],
            [0.0, math.sin(binding_angle), math.cos(binding_angle)],
        ]
    )
    grip = _transform()
    grip[:3, :3] = pointer_rotation @ grip_from_pointer

    direction = device._pointer_direction(grip, {"device_type": 4})

    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0], atol=1e-12)


def test_read_disables_cached_controller_after_source_becomes_stale(monkeypatch):
    device = OculusDevice(stale_timeout=0.15)
    transforms = {"l": _transform((0.1, 0.0, 0.0))}
    buttons = {"LG": True}
    device._reader = SimpleNamespace(
        get_transformations_and_buttons=lambda: (transforms, buttons),
    )

    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.1)
    fresh = device.read()
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.3)
    stale = device.read()

    assert fresh.timestamp == 10.1
    assert fresh.controllers["left"].button("grip") is True
    assert stale.timestamp == 10.3
    assert stale.controllers == {}


def test_one_robot_read_consumes_all_native_rate_frames_and_filters_twist(monkeypatch):
    cutoff = 5.0
    device = OculusDevice(stale_timeout=0.15, velocity_cutoff_hz=cutoff)
    device._reader = BufferedReader([
        (1, 10.00, {"l": _yaw_transform(0.00, (0.00, 0, 0))}, {}),
        (2, 10.01, {"l": _yaw_transform(0.01, (0.01, 0, 0))}, {}),
        (3, 10.02, {"l": _yaw_transform(0.02, (0.02, 0, 0))}, {}),
    ])
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.025)

    state = device.read()

    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff * 0.01)
    expected = 1.0 - (1.0 - alpha) ** 2
    controller = state.controllers["left"]
    assert controller.timestamp == pytest.approx(10.02)
    assert controller.twist is not None
    assert controller.twist.linear[0] == pytest.approx(expected)
    assert controller.twist.angular[2] == pytest.approx(expected)
    np.testing.assert_allclose(controller.twist.linear[1:], 0.0, atol=1e-12)


def test_no_new_source_frame_decays_velocity_instead_of_reusing_it(monkeypatch):
    device = OculusDevice(stale_timeout=0.15, velocity_cutoff_hz=5.0)
    device._reader = BufferedReader([
        (1, 10.00, {"l": _transform((0.00, 0, 0))}, {}),
        (2, 10.01, {"l": _transform((0.01, 0, 0))}, {}),
    ])
    times = iter((10.01, 10.06))
    monkeypatch.setattr(
        "dex_teleop.devices.oculus.time.monotonic", lambda: next(times)
    )

    fresh = device.read().controllers["left"].twist.linear[0]
    decayed = device.read().controllers["left"].twist.linear[0]

    assert 0.0 < decayed < fresh
    assert decayed == pytest.approx(fresh * math.exp(-2.0 * math.pi * 5.0 * 0.05))


def test_tracking_loss_resets_velocity_history_before_a_reappearing_pose(monkeypatch):
    device = OculusDevice(stale_timeout=0.15, velocity_cutoff_hz=5.0)
    device._reader = BufferedReader([
        (1, 10.00, {"l": _transform((0.00, 0, 0))}, {}),
        (2, 10.01, {"l": _transform((0.01, 0, 0))}, {}),
        (3, 10.02, {}, {}),
        (4, 10.03, {"l": _transform((1.00, 0, 0))}, {}),
    ])
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.03)

    controller = device.read().controllers["left"]

    np.testing.assert_allclose(controller.twist.linear, 0.0)
    np.testing.assert_allclose(controller.twist.angular, 0.0)


def _native_tracking(
    *,
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    status=3,
    confidence=1,
    device_type=4,
):
    return {
        "_protocol_version": 2,
        "_frame": "tracking_world",
        "l": {
            "frame": "tracking_world",
            "device_type": device_type,
            "time_in_seconds": 123.0,
            "prediction_in_seconds": 0.01,
            "status": status,
            "confidence": confidence,
            "linear_velocity": np.asarray(linear, dtype=float),
            "angular_velocity": np.asarray(angular, dtype=float),
        }
    }


def test_native_tracking_twist_takes_precedence_over_pose_differencing(monkeypatch):
    device = OculusDevice(stale_timeout=0.15, velocity_cutoff_hz=5.0)
    device._reader = BufferedReader([
        (
            1,
            10.0,
            {"l": _transform((0.0, 0.0, 0.0))},
            {},
            _native_tracking(linear=(1.0, 2.0, 3.0), angular=(4.0, 5.0, 6.0)),
        )
    ])
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.01)

    controller = device.read().controllers["left"]

    np.testing.assert_allclose(controller.twist.linear, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(controller.twist.angular, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(
        controller.pointer_direction,
        [0.0, -math.sqrt(3.0) / 2.0, -0.5],
        atol=1e-12,
    )


def test_non_controller_device_type_falls_back_to_pose_differencing(monkeypatch):
    device = OculusDevice(stale_timeout=0.15)
    device._reader = BufferedReader([
        (
            1,
            10.0,
            {"l": _transform()},
            {},
            _native_tracking(
                linear=(1.0, 2.0, 3.0),
                angular=(4.0, 5.0, 6.0),
                device_type=1 << 5,
            ),
        )
    ])
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.01)

    controller = device.read().controllers["left"]

    np.testing.assert_allclose(controller.twist.linear, 0.0)
    np.testing.assert_allclose(controller.twist.angular, 0.0)
    assert controller.pointer_direction is None


@pytest.mark.parametrize(
    ("status", "confidence"),
    [(0, 1), (1, 1), (2, 1), (3, 0)],
)
def test_invalid_native_tracking_status_or_confidence_omits_controller(
    monkeypatch, status, confidence
):
    device = OculusDevice(stale_timeout=0.15)
    device._reader = BufferedReader([
        (
            1,
            10.0,
            {"l": _transform()},
            {},
            _native_tracking(status=status, confidence=confidence),
        )
    ])
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.01)

    assert device.read().controllers == {}


def test_missing_v2_tracking_record_omits_controller(monkeypatch):
    device = OculusDevice(stale_timeout=0.15)
    device._reader = BufferedReader([
        (
            1,
            10.0,
            {"l": _transform()},
            {},
            {"_protocol_version": 2, "_frame": "tracking_world"},
        )
    ])
    monkeypatch.setattr("dex_teleop.devices.oculus.time.monotonic", lambda: 10.01)

    assert device.read().controllers == {}
