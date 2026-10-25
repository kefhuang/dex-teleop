from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ControllerAction(str, Enum):
    """Device-independent controller intents consumed by applications."""

    HOME = "HOME"
    CALIBRATE = "CALIBRATE"


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = result.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self):
        position_array = _array(self.position, (3,), "position")
        quaternion = _array(self.quaternion_xyzw, (4,), "quaternion_xyzw")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("quaternion_xyzw must have non-zero norm")
        object.__setattr__(self, "position", position_array)
        object.__setattr__(self, "quaternion_xyzw", quaternion / norm)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


@dataclass(frozen=True)
class Twist:
    linear: np.ndarray
    angular: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear", _array(self.linear, (3,), "linear"))
        object.__setattr__(self, "angular", _array(self.angular, (3,), "angular"))


@dataclass(frozen=True)
class ArmState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pose: Pose
    timestamp: float

    def __post_init__(self) -> None:
        positions = np.asarray(self.joint_positions, dtype=float)
        velocities = np.asarray(self.joint_velocities, dtype=float)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("joint_positions must be a non-empty vector")
        object.__setattr__(self, "joint_positions", _array(positions, positions.shape, "joint_positions"))
        object.__setattr__(self, "joint_velocities", _array(velocities, positions.shape, "joint_velocities"))
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "timestamp", timestamp)

    @property
    def pose(self) -> Pose:
        return self.end_effector_pose


@dataclass(frozen=True)
class EndEffectorState:
    timestamp: float
    position: float | None = None
    joint_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if self.position is not None and not np.isfinite(float(self.position)):
            raise ValueError("position must be finite")
        if self.joint_positions is not None:
            values = np.asarray(self.joint_positions, dtype=float)
            object.__setattr__(self, "joint_positions", _array(values, values.shape, "joint_positions"))


@dataclass
class ControllerState:
    name: str
    pose: Pose
    buttons: dict[str, bool] = field(default_factory=dict)
    axes: dict[str, float] = field(default_factory=dict)
    twist: Twist | None = None
    timestamp: float | None = None
    actions: frozenset[ControllerAction] = field(default_factory=frozenset)
    pointer_direction: np.ndarray | None = None

    def __post_init__(self) -> None:
        actions = frozenset(self.actions)
        if not all(isinstance(action, ControllerAction) for action in actions):
            raise TypeError("actions must contain only ControllerAction values")
        self.actions = actions
        if self.twist is not None and not isinstance(self.twist, Twist):
            raise TypeError("twist must be a Twist or None")
        if self.timestamp is not None and not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite or None")
        if self.pointer_direction is not None:
            direction = _array(self.pointer_direction, (3,), "pointer_direction")
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise ValueError("pointer_direction must have non-zero norm")
            direction = direction / norm
            direction.flags.writeable = False
            self.pointer_direction = direction

    def button(self, name: str) -> bool: return bool(self.buttons.get(name, False))
    def axis(self, name: str) -> float: return float(self.axes.get(name, 0.0))
    def action(self, action: ControllerAction) -> bool:
        if not isinstance(action, ControllerAction):
            raise TypeError("action must be a ControllerAction")
        return action in self.actions


@dataclass
class DeviceState:
    timestamp: float
    controllers: dict[str, ControllerState]
