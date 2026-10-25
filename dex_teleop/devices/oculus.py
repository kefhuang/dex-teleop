from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.types import (
    ControllerAction,
    ControllerState,
    DeviceState,
    Pose,
    Twist,
)
from dex_teleop.diagnostics import trace_event
from dex_teleop.utils.transforms import quaternion_delta_rotation_vector


REPO_ROOT = Path(__file__).resolve().parents[2]
OCULUS_READER_ROOT = REPO_ROOT / "dependency" / "oculus_reader"

# The APK renders the green ray from the raw Touch pointer pose along local
# -Z, then applies this +60 degree X rotation to expose the ergonomic grip
# pose sent to the host. Undo it to recover the direction of the visible ray.
_TOUCH_POINTER_FROM_GRIP_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.5, math.sqrt(3.0) / 2.0],
        [0.0, -math.sqrt(3.0) / 2.0, 0.5],
    ],
    dtype=float,
)
_TOUCH_POINTER_FORWARD = np.array([0.0, 0.0, -1.0])


@dataclass
class _PoseVelocityEstimator:
    """Estimate and low-pass controller-frame velocity at source-frame rate."""

    cutoff_hz: float
    max_gap_s: float
    pose: Pose | None = None
    timestamp: float | None = None
    linear: np.ndarray | None = None
    angular: np.ndarray | None = None

    def update(self, pose: Pose, timestamp: float) -> Twist:
        sample_time = float(timestamp)
        if self.pose is None or self.timestamp is None:
            self.pose = pose
            self.timestamp = sample_time
            self.linear = np.zeros(3)
            self.angular = np.zeros(3)
            return Twist(self.linear, self.angular)

        dt = sample_time - self.timestamp
        previous_pose = self.pose
        self.pose = pose
        self.timestamp = sample_time
        if not math.isfinite(dt) or dt <= 1e-4 or dt > self.max_gap_s:
            self.linear = np.zeros(3)
            self.angular = np.zeros(3)
            return Twist(self.linear, self.angular)

        raw_linear = (pose.position - previous_pose.position) / dt
        raw_angular = quaternion_delta_rotation_vector(
            previous_pose.quaternion_xyzw,
            pose.quaternion_xyzw,
        ) / dt
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt)
        previous_linear = np.zeros(3) if self.linear is None else self.linear
        previous_angular = np.zeros(3) if self.angular is None else self.angular
        self.linear = previous_linear + alpha * (raw_linear - previous_linear)
        self.angular = previous_angular + alpha * (raw_angular - previous_angular)
        return Twist(self.linear, self.angular)


class OculusDevice:
    """Thin adapter around the vendored Oculus Reader package."""

    def __init__(
        self,
        *,
        ip: str | None = None,
        port: int = 5555,
        print_fps: bool = False,
        stale_timeout: float = 0.15,
        velocity_cutoff_hz: float = 5.0,
    ) -> None:
        self.ip = ip
        self.port = int(port)
        self.print_fps = bool(print_fps)
        self.stale_timeout = float(stale_timeout)
        self.velocity_cutoff_hz = float(velocity_cutoff_hz)
        if not math.isfinite(self.stale_timeout) or self.stale_timeout <= 0.0:
            raise ValueError("stale_timeout must be finite and positive")
        if not math.isfinite(self.velocity_cutoff_hz) or self.velocity_cutoff_hz <= 0.0:
            raise ValueError("velocity_cutoff_hz must be finite and positive")
        self._reader: Any | None = None
        self._last_source_objects: tuple[Any, Any, Any] | None = None
        self._last_source_received_at: float | None = None
        self._last_reader_sequence = 0
        self._latest_transforms: dict[str, Any] = {}
        self._latest_raw_transforms: dict[str, Any] = {}
        self._latest_buttons: dict[str, Any] = {}
        self._latest_tracking: dict[str, Any] = {}
        self._latest_twists: dict[str, Twist] = {}
        self._latest_pointer_directions: dict[str, np.ndarray] = {}
        self._estimators: dict[str, _PoseVelocityEstimator] = {}

    def start(self) -> None:
        if not OCULUS_READER_ROOT.exists():
            raise RuntimeError(f"Missing Oculus Reader dependency: {OCULUS_READER_ROOT}")
        reader_root = str(OCULUS_READER_ROOT)
        if reader_root not in sys.path:
            sys.path.insert(0, reader_root)

        from oculus_reader.reader import OculusReader  # noqa: PLC0415

        self._reader = OculusReader(
            ip_address=self.ip,
            port=self.port,
            print_FPS=self.print_fps,
            run=True,
        )
        self._last_source_objects = None
        self._last_source_received_at = None
        self._last_reader_sequence = 0
        self._latest_transforms = {}
        self._latest_raw_transforms = {}
        self._latest_buttons = {}
        self._latest_tracking = {}
        self._latest_twists = {}
        self._latest_pointer_directions = {}
        self._estimators = {}

    def read(self) -> DeviceState:
        if self._reader is None:
            raise RuntimeError("OculusDevice.start() must be called before read()")
        now = time.monotonic()
        source_sample_count = 0
        if hasattr(self._reader, "get_samples_since"):
            latest_sequence, samples = self._reader.get_samples_since(
                self._last_reader_sequence
            )
            self._last_reader_sequence = int(latest_sequence)
            for sample in samples:
                if len(sample) == 4:
                    _sequence, received_at, transforms, buttons = sample
                    tracking = {}
                elif len(sample) == 5:
                    _sequence, received_at, transforms, buttons, tracking = sample
                else:
                    continue
                self._consume_source_sample(
                    transforms or {},
                    buttons or {},
                    float(received_at),
                    tracking or {},
                )
                source_sample_count += 1
        if source_sample_count == 0 and self._last_source_received_at is None:
            transforms, buttons = self._reader.get_transformations_and_buttons()
            tracking = (
                self._reader.get_tracking()
                if hasattr(self._reader, "get_tracking")
                else None
            )
            source_changed = self._last_source_objects is None or (
                transforms is not self._last_source_objects[0]
                or buttons is not self._last_source_objects[1]
                or tracking is not self._last_source_objects[2]
            )
            if source_changed:
                self._last_source_objects = (transforms, buttons, tracking)
                self._consume_source_sample(
                    transforms or {}, buttons or {}, now, tracking or {}
                )
                source_sample_count = 1
        elif not hasattr(self._reader, "get_samples_since"):
            transforms, buttons = self._reader.get_transformations_and_buttons()
            tracking = (
                self._reader.get_tracking()
                if hasattr(self._reader, "get_tracking")
                else None
            )
            source_changed = self._last_source_objects is None or (
                transforms is not self._last_source_objects[0]
                or buttons is not self._last_source_objects[1]
                or tracking is not self._last_source_objects[2]
            )
            if source_changed:
                self._last_source_objects = (transforms, buttons, tracking)
                self._consume_source_sample(
                    transforms or {}, buttons or {}, now, tracking or {}
                )
                source_sample_count = 1
        source_timestamp = self._last_source_received_at
        trace_event(
            "oculus.raw_sample",
            transforms=self._latest_raw_transforms,
            buttons=self._latest_buttons,
            tracking=self._latest_tracking,
            source_sample_count=source_sample_count,
            source_changed=source_sample_count > 0,
            source_received_monotonic_s=source_timestamp,
            read_monotonic_s=now,
        )
        if (
            source_timestamp is None
            or not math.isfinite(float(source_timestamp))
            or now - float(source_timestamp) > self.stale_timeout
        ):
            self._estimators = {}
            self._latest_twists = {}
            self._latest_pointer_directions = {}
            state = DeviceState(timestamp=now, controllers={})
            trace_event(
                "oculus.state.result",
                stale=True,
                stale_timeout_s=self.stale_timeout,
                state=state,
            )
            return state
        state = DeviceState(
            timestamp=float(source_timestamp),
            controllers=self._convert(
                self._latest_transforms,
                self._latest_buttons,
                twists=self._twists_for_read(
                    now,
                    source_sample_count > 0,
                ),
                pointer_directions=self._latest_pointer_directions,
                sample_timestamp=float(source_timestamp),
            ),
        )
        trace_event(
            "oculus.state.result",
            stale=False,
            stale_timeout_s=self.stale_timeout,
            state=state,
        )
        return state

    def _twists_for_read(
        self, read_time: float, source_changed: bool
    ) -> dict[str, Twist]:
        if source_changed or self._last_source_received_at is None:
            return self._latest_twists
        age = max(0.0, float(read_time) - self._last_source_received_at)
        decay = math.exp(-2.0 * math.pi * self.velocity_cutoff_hz * age)
        return {
            name: Twist(twist.linear * decay, twist.angular * decay)
            for name, twist in self._latest_twists.items()
        }

    def stop(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        self._last_source_objects = None
        self._last_source_received_at = None
        self._last_reader_sequence = 0
        self._latest_transforms = {}
        self._latest_raw_transforms = {}
        self._latest_buttons = {}
        self._latest_tracking = {}
        self._latest_twists = {}
        self._latest_pointer_directions = {}
        self._estimators = {}

    def _consume_source_sample(
        self,
        transforms: dict[str, Any],
        buttons: dict[str, Any],
        timestamp: float,
        tracking: dict[str, Any] | None = None,
    ) -> None:
        tracking = {} if tracking is None else tracking
        twists: dict[str, Twist] = {}
        pointer_directions: dict[str, np.ndarray] = {}
        valid_transforms: dict[str, Any] = {}
        present: set[str] = set()
        native_tracking_expected = tracking.get("_protocol_version") == 2
        for name in ("left", "right"):
            matrix = self._find_transform(transforms, name)
            if matrix is None or not self._is_valid_transform(matrix):
                self._estimators.pop(name, None)
                continue
            tracking_record = tracking.get(name[0])
            if (
                native_tracking_expected and tracking_record is None
            ) or (
                tracking_record is not None
                and not self._tracking_record_is_valid(tracking_record)
            ):
                self._estimators.pop(name, None)
                continue
            present.add(name)
            valid_transforms[name[0]] = matrix
            pose = self._pose_from_matrix(matrix)
            estimator = self._estimators.get(name)
            if estimator is None:
                estimator = _PoseVelocityEstimator(
                    self.velocity_cutoff_hz,
                    self.stale_timeout,
                )
                self._estimators[name] = estimator
            estimated_twist = estimator.update(pose, timestamp)
            native_twist = self._native_twist(tracking_record)
            twists[name] = estimated_twist if native_twist is None else native_twist
            pointer_direction = self._pointer_direction(matrix, tracking_record)
            if pointer_direction is not None:
                pointer_directions[name] = pointer_direction
        for name in set(self._estimators) - present:
            self._estimators.pop(name, None)
        self._latest_raw_transforms = transforms
        self._latest_transforms = valid_transforms
        self._latest_buttons = buttons
        self._latest_tracking = tracking
        self._latest_twists = twists
        self._latest_pointer_directions = pointer_directions
        self._last_source_received_at = float(timestamp)

    def _tracking_record_is_valid(self, record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        if record.get("frame") != "tracking_world":
            return False
        try:
            status = int(record["status"])
            confidence = int(record["confidence"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        orientation_and_position_tracked = 0x0001 | 0x0002
        return (
            status & orientation_and_position_tracked
        ) == orientation_and_position_tracked and confidence > 0

    def _native_twist(self, record: Any) -> Twist | None:
        if record is None:
            return None
        try:
            tracked_remote_type = 1 << 2
            if int(record["device_type"]) != tracked_remote_type:
                return None
            linear = np.asarray(record["linear_velocity"], dtype=float)
            angular = np.asarray(record["angular_velocity"], dtype=float)
            return Twist(linear=linear, angular=angular)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _pointer_direction(
        self, grip_matrix: Any, tracking_record: Any
    ) -> np.ndarray | None:
        """Recover the APK's visible green Touch ray in tracking-world axes."""
        try:
            tracked_remote_type = 1 << 2
            if (
                not isinstance(tracking_record, dict)
                or int(tracking_record["device_type"]) != tracked_remote_type
            ):
                return None
            grip_rotation = np.asarray(grip_matrix, dtype=float)[:3, :3]
            direction = (
                grip_rotation
                @ _TOUCH_POINTER_FROM_GRIP_ROTATION
                @ _TOUCH_POINTER_FORWARD
            )
            norm = float(np.linalg.norm(direction))
            if not math.isfinite(norm) or norm <= 1e-12:
                return None
            return direction / norm
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _convert(
        self,
        transforms: dict[str, Any],
        buttons: dict[str, Any],
        *,
        twists: dict[str, Twist] | None = None,
        pointer_directions: dict[str, np.ndarray] | None = None,
        sample_timestamp: float | None = None,
    ) -> dict[str, ControllerState]:
        controllers: dict[str, ControllerState] = {}
        for name in ("left", "right"):
            matrix = self._find_transform(transforms, name)
            if matrix is None:
                continue
            if not self._is_valid_transform(matrix):
                continue
            pose = self._pose_from_matrix(matrix)
            home_button = "X" if name == "left" else "A"
            calibrate_button = "Y" if name == "left" else "B"
            actions = set()
            if self._button_bool(buttons, name, home_button):
                actions.add(ControllerAction.HOME)
            if self._button_bool(buttons, name, calibrate_button):
                actions.add(ControllerAction.CALIBRATE)
            controllers[name] = ControllerState(
                name=name,
                pose=pose,
                buttons={"grip": self._button_bool(buttons, name, "grip")},
                axes={"trigger": self._button_float(buttons, name, "trigger")},
                actions=frozenset(actions),
                twist=None if twists is None else twists.get(name),
                timestamp=sample_timestamp,
                pointer_direction=(
                    None
                    if pointer_directions is None
                    else pointer_directions.get(name)
                ),
            )
        return controllers

    def _find_transform(self, transforms: dict[str, Any], side: str) -> Any | None:
        side_key = side[0]
        for key, value in transforms.items():
            lowered = str(key).lower()
            if lowered == side_key or lowered == side:
                return value
            if side in lowered:
                return value
        return None

    def _is_valid_transform(self, matrix: Any) -> bool:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
            return False
        if not np.allclose(arr[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            return False
        rotation = arr[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            return False
        return bool(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3))

    def _pose_from_matrix(self, matrix: Any) -> Pose:
        arr = np.asarray(matrix, dtype=float)
        if not self._is_valid_transform(arr):
            raise ValueError("Oculus controller transform must be a valid 4x4 pose")
        position = arr[:3, 3]
        orientation = self._quaternion_xyzw_from_rotation(arr[:3, :3])
        return Pose(position=position, quaternion_xyzw=orientation)

    def _quaternion_xyzw_from_rotation(self, rotation: Any) -> np.ndarray:
        rot = np.asarray(rotation, dtype=float)
        trace = float(np.trace(rot))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * scale
            x = (rot[2, 1] - rot[1, 2]) / scale
            y = (rot[0, 2] - rot[2, 0]) / scale
            z = (rot[1, 0] - rot[0, 1]) / scale
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / scale
            x = 0.25 * scale
            y = (rot[0, 1] + rot[1, 0]) / scale
            z = (rot[0, 2] + rot[2, 0]) / scale
        elif rot[1, 1] > rot[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / scale
            x = (rot[0, 1] + rot[1, 0]) / scale
            y = 0.25 * scale
            z = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / scale
            x = (rot[0, 2] + rot[2, 0]) / scale
            y = (rot[1, 2] + rot[2, 1]) / scale
            z = 0.25 * scale

        quat = np.array([x, y, z, w], dtype=float)
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return quat / norm

    def _button_bool(self, buttons: dict[str, Any], side: str, name: str) -> bool:
        for candidate in self._button_candidates(side, name):
            if candidate in buttons:
                value = buttons[candidate]
                if isinstance(value, tuple):
                    value = value[0] if value else 0.0
                if name == "grip" and isinstance(value, (int, float, np.integer, np.floating)):
                    return float(value) >= 0.5
                return bool(value)
        return False

    def _button_float(self, buttons: dict[str, Any], side: str, name: str) -> float:
        for candidate in self._button_candidates(side, name):
            if candidate in buttons:
                value = buttons[candidate]
                if isinstance(value, tuple):
                    value = value[0] if value else 0.0
                return float(min(1.0, max(0.0, float(value))))
        return 0.0

    def _button_candidates(self, side: str, name: str) -> list[str]:
        side_initial = side[0]
        oculus_side = "left" if side == "left" else "right"
        oculus_initial = "L" if side == "left" else "R"
        if name == "grip":
            oculus_names = ["Grip", "G"]
        elif name == "trigger":
            oculus_names = ["Trig", "Tr"]
        else:
            oculus_names = [name]
        if name == "grip":
            return [
                f"{oculus_initial}{oculus_names[-1]}",
                f"{side}_{name}",
                f"{side_initial}_{name}",
                f"{side}{name}",
                f"{oculus_side}{oculus_names[0]}",
                name,
            ]
        return [
            f"{side}_{name}",
            f"{side_initial}_{name}",
            f"{side}{name}",
            f"{oculus_side}{oculus_names[0]}",
            f"{oculus_initial}{oculus_names[-1]}",
            name,
        ]
