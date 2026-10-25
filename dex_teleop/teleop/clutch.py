from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dex_teleop.core.types import Pose, Twist
from dex_teleop.diagnostics import trace_event
from dex_teleop.teleop.calibration import TrackingFrameCalibration
from dex_teleop.utils.transforms import (
    clamp_vector_norm,
    quaternion_delta_rotation_vector,
    quaternion_xyzw_to_matrix,
    rotation_matrix_to_quaternion_xyzw,
    rotation_vector_to_matrix,
)


@dataclass(frozen=True)
class ClutchAnchor:
    controller_pose: Pose
    robot_pose: Pose

    @classmethod
    def snapshot(cls, controller_pose: Pose, robot_pose: Pose) -> ClutchAnchor:
        return cls(
            Pose(controller_pose.position.copy(), controller_pose.quaternion_xyzw.copy()),
            Pose(robot_pose.position.copy(), robot_pose.quaternion_xyzw.copy()),
        )


@dataclass(frozen=True)
class TargetPoseResult:
    pose: Pose
    translation_clamped: bool
    rotation_clamped: bool
    mapped_translation: np.ndarray
    mapped_rotation_vector: np.ndarray


class ClutchMapper:
    """Stateful controller-to-base target generator for one arm."""

    def __init__(
        self,
        *,
        calibration: TrackingFrameCalibration | None = None,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        max_translation_from_anchor_m: float = 0.5,
        max_rotation_from_anchor_rad: float = np.pi,
    ) -> None:
        if calibration is not None and not isinstance(
            calibration, TrackingFrameCalibration
        ):
            raise TypeError("calibration must be a TrackingFrameCalibration")
        self.calibration = calibration or TrackingFrameCalibration()
        self.options = {
            "position_scale": float(position_scale),
            "rotation_scale": float(rotation_scale),
            "max_translation_from_anchor_m": float(
                max_translation_from_anchor_m
            ),
            "max_rotation_from_anchor_rad": float(
                max_rotation_from_anchor_rad
            ),
        }
        self.anchor: ClutchAnchor | None = None
        self._last_target_result: TargetPoseResult | None = None

    def release(self) -> None:
        trace_event("teleop.clutch.anchor_cleared", previous_anchor=self.anchor)
        self.anchor = None
        self._last_target_result = None

    def update(self, controller_pose: Pose, measured_robot_pose: Pose) -> Pose:
        robot_from_vr = self.calibration.robot_from_vr
        if self.anchor is None:
            self.anchor = ClutchAnchor.snapshot(
                controller_pose, measured_robot_pose
            )
            trace_event("teleop.clutch.anchor_created", anchor=self.anchor)
        result = build_clutch_target(
            controller_pose,
            self.anchor,
            robot_from_vr_rotation=robot_from_vr,
            **self.options,
        )
        self._last_target_result = result
        trace_event(
            "teleop.clutch.target_built",
            anchor=self.anchor,
            controller_pose=controller_pose,
            measured_robot_pose=measured_robot_pose,
            result=result,
        )
        return result.pose

    def map_twist(self, controller_twist: Twist) -> Twist:
        """Map a controller-frame Twist into the robot base frame."""
        if not isinstance(controller_twist, Twist):
            raise TypeError("controller_twist must be a Twist")
        linear = self.calibration.map_vector(controller_twist.linear) * self.options[
            "position_scale"
        ]
        angular = self.calibration.map_vector(controller_twist.angular) * self.options[
            "rotation_scale"
        ]
        result = self._last_target_result
        if result is not None:
            if result.translation_clamped:
                linear = _remove_outward_component(
                    linear, result.mapped_translation
                )
            if result.rotation_clamped:
                angular = _remove_outward_component(
                    angular, result.mapped_rotation_vector
                )
        mapped = Twist(linear, angular)
        trace_event(
            "teleop.clutch.twist_mapped",
            controller_twist=controller_twist,
            mapped_twist=mapped,
            target_result=result,
        )
        return mapped


def _remove_outward_component(
    velocity: np.ndarray, radial_offset: np.ndarray
) -> np.ndarray:
    """Preserve tangential motion while suppressing motion beyond a clamp."""
    result = np.asarray(velocity, dtype=float).copy()
    radial = np.asarray(radial_offset, dtype=float)
    squared_norm = float(radial @ radial)
    if squared_norm <= 1e-18:
        return result
    outward = float(result @ radial)
    if outward > 0.0:
        result -= radial * (outward / squared_norm)
    return result


def build_clutch_target(
    controller_pose: Pose,
    anchor: ClutchAnchor,
    *,
    robot_from_vr_rotation: np.ndarray,
    position_scale: float,
    rotation_scale: float,
    max_translation_from_anchor_m: float,
    max_rotation_from_anchor_rad: float,
) -> TargetPoseResult:
    rotation = np.asarray(robot_from_vr_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(
            "robot_from_vr_rotation must be a finite shape-(3, 3) matrix"
        )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("robot_from_vr_rotation must be a proper rotation matrix")
    raw_translation = controller_pose.position - anchor.controller_pose.position
    mapped_translation = rotation @ raw_translation * position_scale
    limited_translation = clamp_vector_norm(
        mapped_translation, max_translation_from_anchor_m
    )
    raw_rotation = quaternion_delta_rotation_vector(
        anchor.controller_pose.quaternion_xyzw, controller_pose.quaternion_xyzw
    )
    mapped_rotation = rotation @ raw_rotation * rotation_scale
    limited_rotation = clamp_vector_norm(
        mapped_rotation, max_rotation_from_anchor_rad
    )
    anchor_rotation = quaternion_xyzw_to_matrix(anchor.robot_pose.quaternion_xyzw)
    target_rotation = rotation_vector_to_matrix(limited_rotation) @ anchor_rotation
    return TargetPoseResult(
        pose=Pose(
            anchor.robot_pose.position + limited_translation,
            rotation_matrix_to_quaternion_xyzw(target_rotation),
        ),
        translation_clamped=not np.allclose(
            mapped_translation, limited_translation
        ),
        rotation_clamped=not np.allclose(mapped_rotation, limited_rotation),
        mapped_translation=limited_translation,
        mapped_rotation_vector=limited_rotation,
    )
