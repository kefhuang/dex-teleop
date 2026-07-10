from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dex_teleop.core.types import Pose
from dex_teleop.utils.transforms import (
    axis_mapping,
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
            Pose(controller_pose.position.copy(), controller_pose.orientation.copy()),
            Pose(robot_pose.position.copy(), robot_pose.orientation.copy()),
        )


@dataclass(frozen=True)
class TargetPoseResult:
    pose: Pose
    translation_clamped: bool
    rotation_clamped: bool
    mapped_translation: np.ndarray
    mapped_rotation_vector: np.ndarray


def build_clutch_target(
    controller_pose: Pose,
    anchor: ClutchAnchor,
    *,
    position_axes: dict[str, str],
    rotation_axes: dict[str, str],
    position_scale: float,
    rotation_scale: float,
    max_translation_from_anchor_m: float,
    max_rotation_from_anchor_rad: float,
) -> TargetPoseResult:
    raw_translation = controller_pose.position - anchor.controller_pose.position
    mapped_translation = (
        axis_mapping(
            raw_translation,
            position_axes,
            ("x", "y", "z"),
            ("x", "y", "z"),
        )
        * position_scale
    )
    limited_translation = clamp_vector_norm(
        mapped_translation, max_translation_from_anchor_m
    )
    raw_rotation = quaternion_delta_rotation_vector(
        anchor.controller_pose.orientation, controller_pose.orientation
    )
    mapped_rotation = (
        axis_mapping(
            raw_rotation,
            rotation_axes,
            ("roll", "pitch", "yaw"),
            ("roll", "pitch", "yaw"),
        )
        * rotation_scale
    )
    limited_rotation = clamp_vector_norm(
        mapped_rotation, max_rotation_from_anchor_rad
    )
    anchor_rotation = quaternion_xyzw_to_matrix(anchor.robot_pose.orientation)
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
