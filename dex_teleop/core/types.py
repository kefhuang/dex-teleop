from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


@dataclass
class Pose:
    """Cartesian pose with quaternion orientation in xyzw convention."""

    position: np.ndarray | list[float] | tuple[float, float, float]
    orientation: np.ndarray | list[float] | tuple[float, float, float, float]

    def __post_init__(self) -> None:
        self.position = _array(self.position, (3,), "position")
        self.orientation = _array(self.orientation, (4,), "orientation")

    @classmethod
    def identity(cls) -> "Pose":
        return cls(
            position=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )


@dataclass
class ControllerState:
    name: str
    pose: Pose
    buttons: dict[str, bool] = field(default_factory=dict)
    axes: dict[str, float] = field(default_factory=dict)

    def button(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))

    def axis(self, name: str) -> float:
        return float(self.axes.get(name, 0.0))


@dataclass
class DeviceState:
    timestamp: float
    controllers: dict[str, ControllerState]


@dataclass
class ArmState:
    arm_id: str
    pose: Pose


@dataclass
class MotionCommand:
    arm_id: str
    delta_position: np.ndarray | list[float] | tuple[float, float, float]
    delta_rotation: np.ndarray | list[float] | tuple[float, float, float]
    enabled: bool = True

    def __post_init__(self) -> None:
        self.delta_position = _array(self.delta_position, (3,), "delta_position")
        self.delta_rotation = _array(self.delta_rotation, (3,), "delta_rotation")

    @classmethod
    def disabled(cls, arm_id: str) -> "MotionCommand":
        return cls(
            arm_id=arm_id,
            delta_position=np.zeros(3),
            delta_rotation=np.zeros(3),
            enabled=False,
        )


@dataclass
class GripperCommand:
    gripper_id: str
    normalized: float

    def __post_init__(self) -> None:
        self.normalized = _clamp01(self.normalized)


@dataclass
class TeleopCommandBatch:
    motions: dict[str, MotionCommand] = field(default_factory=dict)
    grippers: dict[str, GripperCommand] = field(default_factory=dict)
