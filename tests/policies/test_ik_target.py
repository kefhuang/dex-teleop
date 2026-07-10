from __future__ import annotations

import math

import numpy as np

from dex_teleop.core.types import Pose
from dex_teleop.policies.ik_target import ClutchAnchor, build_clutch_target
from dex_teleop.utils.transforms import (
    quaternion_xyzw_to_matrix,
    rotation_matrix_to_quaternion_xyzw,
    rotation_vector_to_matrix,
)


IDENTITY_POSITION = {"x": "+x", "y": "+y", "z": "+z"}
IDENTITY_ROTATION = {"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"}


def _quaternion(vector: np.ndarray) -> np.ndarray:
    return rotation_matrix_to_quaternion_xyzw(rotation_vector_to_matrix(vector))


def test_build_clutch_target_maps_full_pose_and_clamps_anchor_norms():
    anchor = ClutchAnchor.snapshot(Pose.identity(), Pose.identity())
    yaw = 1.0
    controller = Pose(
        [0.3, 0.4, 0.0],
        [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
    )
    result = build_clutch_target(
        controller,
        anchor,
        position_axes=IDENTITY_POSITION,
        rotation_axes=IDENTITY_ROTATION,
        position_scale=1.0,
        rotation_scale=1.0,
        max_translation_from_anchor_m=0.1,
        max_rotation_from_anchor_rad=0.5,
    )

    np.testing.assert_allclose(result.pose.position, [0.06, 0.08, 0.0], atol=1e-8)
    assert result.translation_clamped is True
    assert result.rotation_clamped is True


def test_build_clutch_target_applies_signed_axis_mappings_and_scales():
    anchor = ClutchAnchor.snapshot(
        Pose.identity(),
        Pose([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
    )
    yaw = 0.25
    controller = Pose(
        [0.1, 0.2, 0.3],
        [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
    )

    result = build_clutch_target(
        controller,
        anchor,
        position_axes={"x": "+z", "y": "-x", "z": "+y"},
        rotation_axes={"roll": "+yaw", "pitch": "-roll", "yaw": "+pitch"},
        position_scale=2.0,
        rotation_scale=0.5,
        max_translation_from_anchor_m=10.0,
        max_rotation_from_anchor_rad=10.0,
    )

    np.testing.assert_allclose(result.pose.position, [1.6, 0.8, 1.4], atol=1e-8)
    np.testing.assert_allclose(result.mapped_translation, [0.6, -0.2, 0.4], atol=1e-8)
    np.testing.assert_allclose(
        quaternion_xyzw_to_matrix(result.pose.orientation),
        rotation_vector_to_matrix(np.array([0.125, 0.0, 0.0])),
        atol=1e-8,
    )
    assert result.translation_clamped is False
    assert result.rotation_clamped is False


def test_build_clutch_target_composes_rotation_in_robot_base_frame():
    anchor_rotation = rotation_vector_to_matrix(np.array([0.0, 0.0, math.pi / 2.0]))
    anchor = ClutchAnchor.snapshot(
        Pose.identity(),
        Pose(np.zeros(3), rotation_matrix_to_quaternion_xyzw(anchor_rotation)),
    )
    controller = Pose(np.zeros(3), _quaternion(np.array([0.2, 0.0, 0.0])))

    result = build_clutch_target(
        controller,
        anchor,
        position_axes=IDENTITY_POSITION,
        rotation_axes=IDENTITY_ROTATION,
        position_scale=1.0,
        rotation_scale=1.0,
        max_translation_from_anchor_m=1.0,
        max_rotation_from_anchor_rad=1.0,
    )

    expected = rotation_vector_to_matrix(np.array([0.2, 0.0, 0.0])) @ anchor_rotation
    np.testing.assert_allclose(
        quaternion_xyzw_to_matrix(result.pose.orientation), expected, atol=1e-8
    )


def test_clutch_anchor_snapshot_does_not_alias_source_poses():
    controller = Pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    robot = Pose([4.0, 5.0, 6.0], [0.1, 0.2, 0.3, 0.9])
    anchor = ClutchAnchor.snapshot(controller, robot)

    controller.position[:] = -1.0
    controller.orientation[:] = -1.0
    robot.position[:] = -2.0
    robot.orientation[:] = -2.0

    np.testing.assert_allclose(anchor.controller_pose.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(anchor.controller_pose.orientation, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(anchor.robot_pose.position, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(anchor.robot_pose.orientation, [0.1, 0.2, 0.3, 0.9])


def test_build_clutch_target_clamps_vectors_without_changing_direction():
    anchor = ClutchAnchor.snapshot(Pose.identity(), Pose.identity())
    raw_rotation = np.array([0.2, -0.3, 0.4])
    controller = Pose([0.3, -0.4, 0.0], _quaternion(raw_rotation))

    result = build_clutch_target(
        controller,
        anchor,
        position_axes=IDENTITY_POSITION,
        rotation_axes=IDENTITY_ROTATION,
        position_scale=2.0,
        rotation_scale=2.0,
        max_translation_from_anchor_m=0.25,
        max_rotation_from_anchor_rad=0.1,
    )

    scaled_rotation = raw_rotation * 2.0
    expected_rotation = scaled_rotation * (0.1 / np.linalg.norm(scaled_rotation))
    np.testing.assert_allclose(result.mapped_translation, [0.15, -0.2, 0.0], atol=1e-8)
    np.testing.assert_allclose(result.mapped_rotation_vector, expected_rotation, atol=1e-8)
