from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.types import ArmState, GripperCommand, MotionCommand, Pose

_CONTROLLER_TYPE = "OSC_POSE"
_ZERO_ACTION = [0.0] * 7

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
        return ArmState(arm_id=self.arm_id, pose=Pose.identity())

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
        # franka-interface process, not just to stop the current arm motion.
        # Use a zero OSC delta as a hold/keepalive command so releasing the
        # clutch does not kill the backend.
        self._control(list(_ZERO_ACTION), termination=False)

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
