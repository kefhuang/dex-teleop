from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.types import ArmState, GripperCommand, MotionCommand, Pose

_CONTROLLER_TYPE = "OSC_POSE"
_ZERO_ACTION = [0.0] * 7


def _send_no_control(interface: Any) -> None:
    """Send a Deoxys NO_CONTROL keepalive without running OSC nullspace torque."""

    if hasattr(interface, "send_no_control"):
        interface.send_no_control()
        return

    _ensure_deoxys_on_path()
    from deoxys.proto.franka_interface import franka_controller_pb2

    dummy_msg = franka_controller_pb2.FrankaDummyControllerMessage()
    control_msg = franka_controller_pb2.FrankaControlMessage()
    control_msg.controller_type = (
        franka_controller_pb2.FrankaControlMessage.ControllerType.NO_CONTROL
    )
    control_msg.control_msg.Pack(dummy_msg)
    control_msg.timeout = 0.2
    control_msg.termination = False
    interface._publisher.send(control_msg.SerializeToString())

# Deoxys' generated protobuf modules require the pure-Python protobuf runtime in
# this repository's environment. Keep this as a default so callers may override
# it explicitly before import if needed.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _deoxys_root() -> Path:
    return Path(__file__).resolve().parents[2] / "dependency" / "deoxys_control" / "deoxys"


def _ensure_deoxys_on_path() -> None:
    try:
        __import__("deoxys")
        return
    except ModuleNotFoundError:
        pass

    root = _deoxys_root()
    root_str = str(root)
    if root.exists() and root_str not in sys.path:
        sys.path.insert(0, root_str)


def _create_franka_interface(interface_cfg: str) -> Any:
    _ensure_deoxys_on_path()
    from deoxys.franka_interface import FrankaInterface

    return FrankaInterface(
        interface_cfg,
        has_gripper=False,
        use_visualizer=False,
    )


def _load_controller_config(controller_type: str) -> Any:
    _ensure_deoxys_on_path()
    from deoxys.utils.config_utils import get_default_controller_config

    return get_default_controller_config(controller_type)


def _is_valid_eef_pose(pose: np.ndarray) -> bool:
    if pose.shape != (4, 4):
        return False
    if not np.all(np.isfinite(pose)):
        return False
    if not np.allclose(pose[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        return False

    rotation = pose[:3, :3]
    if np.allclose(rotation, np.zeros((3, 3)), atol=1e-12):
        return False
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        return False
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        return False

    return True

def _quaternion_xyzw_from_rotation(rotation: Any) -> np.ndarray:
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


class FrankaArmController:
    def __init__(
        self,
        arm_id: str,
        interface_cfg: str,
        *,
        interface: Any | None = None,
        controller_cfg: Any | None = None,
    ) -> None:
        self.arm_id = arm_id
        self.interface_cfg = interface_cfg
        self.interface = interface
        self.controller_cfg = controller_cfg
        self.connected = False
        self.closed = False
        self._owns_interface = interface is None

    def connect(self) -> None:
        if self.interface is None:
            self.interface = _create_franka_interface(self.interface_cfg)
            if self.controller_cfg is None:
                self.controller_cfg = _load_controller_config(_CONTROLLER_TYPE)
        self.connected = True

    def get_state(self) -> ArmState:
        if self.interface is None:
            return ArmState(arm_id=self.arm_id, pose=Pose.identity())

        pose = getattr(self.interface, "last_eef_pose", None)
        if pose is None:
            return ArmState(arm_id=self.arm_id, pose=Pose.identity())

        arr = np.asarray(pose, dtype=float)
        if not _is_valid_eef_pose(arr):
            return ArmState(arm_id=self.arm_id, pose=Pose.identity())

        return ArmState(
            arm_id=self.arm_id,
            pose=Pose(
                position=arr[:3, 3],
                orientation=_quaternion_xyzw_from_rotation(arr[:3, :3]),
            ),
        )

    def send_motion(self, command: MotionCommand) -> None:
        if not command.enabled:
            self.stop_motion()
            return

        action = np.concatenate(
            [command.delta_position, command.delta_rotation, np.array([0.0])]
        ).tolist()
        self._control(action, termination=False)

    def stop_motion(self) -> None:
        # Deoxys C++ treats termination=True as a request to terminate the
        # franka-interface process. Do not send OSC zero here either: the OSC
        # controller applies nullspace posture torques even with zero Cartesian
        # delta, so the arm can move while the clutch is released. NO_CONTROL
        # is the safe idle/keepalive command.
        if self.interface is None:
            raise RuntimeError("FrankaArmController must be connected before use")
        _send_no_control(self.interface)

    def close(self) -> None:
        if self._owns_interface and self.interface is not None:
            self.interface.close()
        self.closed = True

    def _control(self, action: list[float], *, termination: bool) -> None:
        if self.interface is None:
            raise RuntimeError("FrankaArmController must be connected before use")
        self.interface.control(
            controller_type=_CONTROLLER_TYPE,
            action=action,
            controller_cfg=self.controller_cfg,
            termination=termination,
        )


class FrankaGripperController:
    def __init__(self, gripper_id: str, *, interface: Any) -> None:
        self.gripper_id = gripper_id
        self.interface = interface
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        if self.interface is None:
            raise RuntimeError("FrankaGripperController requires a Franka interface")
        self.connected = True

    def send_gripper(self, command: GripperCommand) -> None:
        if self.interface is None:
            raise RuntimeError("FrankaGripperController must be connected before use")
        self.interface.gripper_control(self._to_deoxys_action(command.normalized))

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _to_deoxys_action(normalized: float) -> float:
        return (2.0 * float(normalized)) - 1.0
