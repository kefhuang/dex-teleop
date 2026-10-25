import math

import numpy as np
import pytest

from dex_teleop.core.types import Pose, Twist
from dex_teleop.teleop import (
    CalibrationError,
    ClutchMapper,
    TrackingFrameCalibration,
)


def _calibrated_mapper(**kwargs) -> ClutchMapper:
    calibration = TrackingFrameCalibration()
    calibration.calibrate(np.array([0.0, 0.0, 1.0]))
    return ClutchMapper(calibration=calibration, **kwargs)


def test_mapper_anchors_and_maps_base_frame_translation():
    mapper = _calibrated_mapper(
        max_translation_from_anchor_m=1,
        max_rotation_from_anchor_rad=1,
    )
    robot = Pose([0.4, 0.0, 0.2], [0, 0, 0, 1])
    first = mapper.update(Pose.identity(), robot)

    np.testing.assert_allclose(first.position, robot.position)

    target = mapper.update(Pose([0.1, 0.2, 0.3], [0, 0, 0, 1]), robot)

    # With the pointer along VR +Z: VR [+X, +Y, +Z] -> robot [+Y, +Z, +X].
    np.testing.assert_allclose(target.position, [0.7, 0.1, 0.4])


def test_mapper_maps_world_frame_rotation_vector_with_same_calibration():
    mapper = _calibrated_mapper(max_rotation_from_anchor_rad=1)
    mapper.update(Pose.identity(), Pose.identity())
    angle = 0.2
    controller = Pose(
        [0, 0, 0],
        [0, 0, math.sin(angle / 2), math.cos(angle / 2)],
    )

    mapper.update(controller, Pose.identity())

    assert mapper._last_target_result is not None
    np.testing.assert_allclose(
        mapper._last_target_result.mapped_rotation_vector,
        [angle, 0.0, 0.0],
        atol=1e-12,
    )


def test_release_reanchors():
    mapper = _calibrated_mapper()
    pose = Pose.identity()
    mapper.update(pose, pose)

    mapper.release()

    assert mapper.anchor is None


def test_mapper_maps_controller_twist_with_same_rotation_and_scales():
    mapper = _calibrated_mapper(
        position_scale=0.5,
        rotation_scale=0.25,
        max_translation_from_anchor_m=1,
        max_rotation_from_anchor_rad=1,
    )
    mapper.update(Pose.identity(), Pose.identity())

    mapped = mapper.map_twist(Twist([1, 2, 3], [4, 5, 6]))

    np.testing.assert_allclose(mapped.linear, [1.5, 0.5, 1.0])
    np.testing.assert_allclose(mapped.angular, [1.5, 1.0, 1.25])


def test_mapper_removes_only_outward_twist_at_workspace_clamp():
    mapper = _calibrated_mapper(
        max_translation_from_anchor_m=1,
        max_rotation_from_anchor_rad=1,
    )
    mapper.update(Pose.identity(), Pose.identity())
    mapper.update(Pose([0, 0, 2], [0, 0, 0, 1]), Pose.identity())

    outward = mapper.map_twist(Twist([1, 0, 1], [0, 0, 0]))
    inward = mapper.map_twist(Twist([1, 0, -1], [0, 0, 0]))

    np.testing.assert_allclose(outward.linear, [0, 1, 0])
    np.testing.assert_allclose(inward.linear, [-1, 1, 0])


def test_mapper_requires_its_own_calibration_before_teleoperation():
    mapper = ClutchMapper()

    with pytest.raises(CalibrationError, match="has not been calibrated"):
        mapper.update(Pose.identity(), Pose.identity())

    assert mapper.anchor is None


def test_two_mappers_keep_independent_per_arm_calibrations():
    left = ClutchMapper()
    right = ClutchMapper()

    left.calibration.calibrate(np.array([0.0, 0.0, 1.0]))
    right.calibration.calibrate(np.array([0.0, 0.0, -1.0]))

    assert left.calibration is not right.calibration
    assert not np.allclose(
        left.calibration.robot_from_vr,
        right.calibration.robot_from_vr,
    )
    np.testing.assert_allclose(
        left.calibration.map_vector(np.array([0.0, 0.0, 1.0])),
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        right.calibration.map_vector(np.array([0.0, 0.0, -1.0])),
        [1.0, 0.0, 0.0],
    )
