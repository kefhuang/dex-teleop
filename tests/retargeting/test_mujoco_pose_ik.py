from __future__ import annotations

import numpy as np
import pytest

from dex_teleop.core.types import Pose
from dex_teleop.retargeting.mujoco_pose_ik import MujocoPoseIKSolver
from dex_teleop.utils.transforms import (
    quaternion_xyzw_to_matrix,
    rotation_error_angle,
    rotation_matrix_to_quaternion_xyzw,
    rotation_vector_to_matrix,
)


HOME = np.array(
    [0.09162, -0.19826, -0.01990, -2.47323, -0.01307, 2.30397, 0.84809]
)


def test_solver_reaches_relative_translation_and_rotation():
    solver = MujocoPoseIKSolver()
    measured = solver.forward_pose(HOME)
    target_rotation = rotation_vector_to_matrix(
        np.array([0.02, -0.01, 0.03])
    ) @ solver.forward_rotation(HOME)
    target = Pose(
        measured.position + np.array([0.005, -0.004, 0.006]),
        rotation_matrix_to_quaternion_xyzw(target_rotation),
    )

    solution = solver.solve(HOME, measured, target)

    assert solution.position_residual_m <= 0.005
    assert solution.rotation_residual_rad <= 0.05
    assert solution.joint_positions.shape == (7,)
    assert np.all(solution.joint_positions >= solution.lower_limits)
    assert np.all(solution.joint_positions <= solution.upper_limits)


def test_solver_applies_measured_pose_error_to_model_fk():
    solver = MujocoPoseIKSolver()
    model_pose = solver.forward_pose(HOME)
    model_rotation = quaternion_xyzw_to_matrix(model_pose.orientation)
    measured_rotation = rotation_vector_to_matrix(
        np.array([0.08, -0.05, 0.04])
    ) @ model_rotation
    measured = Pose(
        model_pose.position + np.array([0.08, -0.06, 0.04]),
        rotation_matrix_to_quaternion_xyzw(measured_rotation),
    )
    position_delta = np.array([-0.004, 0.003, 0.005])
    rotation_delta = rotation_vector_to_matrix(np.array([-0.015, 0.025, 0.01]))
    target = Pose(
        measured.position + position_delta,
        rotation_matrix_to_quaternion_xyzw(rotation_delta @ measured_rotation),
    )

    solution = solver.solve(HOME, measured, target)
    solved_pose = solver.forward_pose(solution.joint_positions)

    np.testing.assert_allclose(
        solved_pose.position,
        model_pose.position + position_delta,
        atol=0.005,
    )
    assert rotation_error_angle(
        rotation_delta @ model_rotation,
        quaternion_xyzw_to_matrix(solved_pose.orientation),
    ) <= 0.05


def test_forward_pose_is_deterministic_after_other_fk_calls():
    solver = MujocoPoseIKSolver()
    expected = solver.forward_pose(HOME)

    solver.forward_pose(HOME + np.array([0.1, -0.05, 0.08, 0.0, -0.03, 0.04, -0.02]))
    actual = solver.forward_pose(HOME)

    np.testing.assert_allclose(actual.position, expected.position, atol=1e-12)
    np.testing.assert_allclose(
        quaternion_xyzw_to_matrix(actual.orientation),
        quaternion_xyzw_to_matrix(expected.orientation),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "joints",
    [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)],
)
def test_forward_pose_rejects_invalid_joint_positions(joints):
    solver = MujocoPoseIKSolver()

    with pytest.raises(ValueError, match="joint_positions"):
        solver.forward_pose(joints)


def test_solver_exposes_panda_arm_joint_limits_only():
    solver = MujocoPoseIKSolver()

    np.testing.assert_allclose(
        solver.lower_limits,
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    )
    np.testing.assert_allclose(
        solver.upper_limits,
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    )


def test_solver_bounds_unreachable_result_and_reports_its_fk_residual():
    solver = MujocoPoseIKSolver()
    measured = solver.forward_pose(HOME)
    target = Pose(
        measured.position + np.array([0.0, 0.0, 0.5]),
        measured.orientation.copy(),
    )

    solution = solver.solve(HOME, measured, target)
    solved_pose = solver.forward_pose(solution.joint_positions)

    assert np.all(solution.joint_positions >= solution.lower_limits)
    assert np.all(solution.joint_positions <= solution.upper_limits)
    assert solution.position_residual_m == pytest.approx(
        np.linalg.norm(solved_pose.position - target.position)
    )
    assert solution.rotation_residual_rad == pytest.approx(
        rotation_error_angle(
            quaternion_xyzw_to_matrix(target.orientation),
            quaternion_xyzw_to_matrix(solved_pose.orientation),
        )
    )


def test_solution_arrays_are_immutable():
    solver = MujocoPoseIKSolver()
    measured = solver.forward_pose(HOME)
    solution = solver.solve(HOME, measured, measured)

    for values in (
        solution.joint_positions,
        solution.lower_limits,
        solution.upper_limits,
    ):
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0
