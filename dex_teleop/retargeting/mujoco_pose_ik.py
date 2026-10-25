from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from dex_teleop.core.types import Pose
from dex_teleop.utils.transforms import (
    quaternion_xyzw_to_matrix,
    rotation_error_angle,
    rotation_matrix_to_quaternion_xyzw,
)


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _finite_joints(joint_positions: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_positions, dtype=float)
    if joints.shape != (7,):
        raise ValueError(f"joint_positions must have shape (7,), got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("joint_positions must contain only finite values")
    return joints


def _read_only_copy(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PoseIKSolution:
    joint_positions: np.ndarray
    position_residual_m: float
    rotation_residual_rad: float
    lower_limits: np.ndarray
    upper_limits: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "joint_positions", _read_only_copy(self.joint_positions)
        )
        object.__setattr__(self, "lower_limits", _read_only_copy(self.lower_limits))
        object.__setattr__(self, "upper_limits", _read_only_copy(self.upper_limits))


class MujocoPoseIKSolver:
    def __init__(self) -> None:
        from deoxys.utils.ik_utils import IKWrapper

        self.wrapper = IKWrapper()
        self.model = self.wrapper.model
        self.data = self.wrapper.data
        self.site_id = self.model.site("grip_site").id
        self.lower_limits = np.asarray(self.model.jnt_range[:7, 0], dtype=float)
        self.upper_limits = np.asarray(self.model.jnt_range[:7, 1], dtype=float)

    def forward_pose(self, joint_positions: np.ndarray) -> Pose:
        rotation = self.forward_rotation(joint_positions)
        position = np.copy(self.data.site(self.site_id).xpos)
        return Pose(position, rotation_matrix_to_quaternion_xyzw(rotation))

    def forward_rotation(self, joint_positions: np.ndarray) -> np.ndarray:
        joints = _finite_joints(joint_positions)
        self.data.qpos[:] = joints.tolist() + [0.04, 0.04]
        mujoco.mj_fwdPosition(self.model, self.data)
        return np.copy(self.data.site(self.site_id).xmat).reshape(3, 3)

    def solve(
        self,
        seed_joint_positions: np.ndarray,
        measured_robot_pose: Pose,
        target_robot_pose: Pose,
    ) -> PoseIKSolution:
        seed = _finite_joints(seed_joint_positions)
        model_pose = self.forward_pose(seed)
        model_rotation = quaternion_xyzw_to_matrix(model_pose.orientation)
        measured_rotation = quaternion_xyzw_to_matrix(measured_robot_pose.orientation)
        target_rotation = quaternion_xyzw_to_matrix(target_robot_pose.orientation)
        model_target_position = (
            model_pose.position
            + target_robot_pose.position
            - measured_robot_pose.position
        )
        model_target_rotation = target_rotation @ measured_rotation.T @ model_rotation
        solved = _finite_joints(
            self.wrapper.inverse_kinematics(
                self.model,
                self.data,
                model_target_rotation,
                model_target_position,
                seed.tolist(),
            )
        )
        solved = np.clip(solved, self.lower_limits, self.upper_limits)
        solved_pose = self.forward_pose(solved)
        return PoseIKSolution(
            solved.copy(),
            float(np.linalg.norm(solved_pose.position - model_target_position)),
            rotation_error_angle(
                model_target_rotation,
                quaternion_xyzw_to_matrix(solved_pose.orientation),
            ),
            self.lower_limits.copy(),
            self.upper_limits.copy(),
        )
