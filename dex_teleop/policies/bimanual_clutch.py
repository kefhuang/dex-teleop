from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dex_teleop.core.types import (
    DeviceState,
    GripperCommand,
    MotionCommand,
    Pose,
    TeleopCommandBatch,
)
from dex_teleop.utils.transforms import (
    apply_deadzone,
    axis_mapping,
    limit_components,
    quaternion_delta_rpy,
)


@dataclass
class _ArmClutchState:
    active: bool = False
    previous_pose: Pose | None = None


def _snapshot_pose(pose: Pose) -> Pose:
    return Pose(
        position=pose.position.copy(),
        orientation=pose.orientation.copy(),
    )


class BimanualClutchPolicy:
    def __init__(
        self,
        *,
        arm_bindings: dict[str, dict[str, Any]],
        clutch_button: str,
        gripper_axis: str,
        position_axes: dict[str, str],
        rotation_axes: dict[str, str],
        position_scale: float,
        rotation_scale: float,
        max_translation_per_cycle: float,
        max_rotation_per_cycle: float,
        position_deadzone: float,
        rotation_deadzone: float,
        send_zero_when_idle: bool,
    ) -> None:
        self.arm_bindings = arm_bindings
        self.clutch_button = clutch_button
        self.gripper_axis = gripper_axis
        self.position_axes = position_axes
        self.rotation_axes = rotation_axes
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.max_translation_per_cycle = float(max_translation_per_cycle)
        self.max_rotation_per_cycle = float(max_rotation_per_cycle)
        self.position_deadzone = float(position_deadzone)
        self.rotation_deadzone = float(rotation_deadzone)
        self.send_zero_when_idle = bool(send_zero_when_idle)
        self._states = {arm_id: _ArmClutchState() for arm_id in arm_bindings}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> BimanualClutchPolicy:
        return cls(
            arm_bindings={
                arm_id: {
                    "controller": arm_cfg["controller"],
                    "gripper": arm_cfg["gripper"],
                }
                for arm_id, arm_cfg in cfg["arms"].items()
            },
            clutch_button=cfg["controls"]["clutch_button"],
            gripper_axis=cfg["controls"]["gripper_axis"],
            position_axes=cfg["mapping"]["position_axes"],
            rotation_axes=cfg["mapping"]["rotation_axes"],
            position_scale=cfg["scales"]["position"],
            rotation_scale=cfg["scales"]["rotation"],
            max_translation_per_cycle=cfg["safety"]["max_translation_per_cycle"],
            max_rotation_per_cycle=cfg["safety"]["max_rotation_per_cycle"],
            position_deadzone=cfg["safety"]["position_deadzone"],
            rotation_deadzone=cfg["safety"]["rotation_deadzone"],
            send_zero_when_idle=cfg["safety"]["send_zero_when_idle"],
        )

    def update(self, state: DeviceState) -> TeleopCommandBatch:
        motions: dict[str, MotionCommand] = {}
        grippers: dict[str, GripperCommand] = {}

        for arm_id, binding in self.arm_bindings.items():
            controller_name = binding["controller"]
            gripper_id = binding["gripper"]
            controller = state.controllers.get(controller_name)

            if controller is None:
                self._states[arm_id] = _ArmClutchState()
                motions[arm_id] = MotionCommand.disabled(arm_id)
                grippers[gripper_id] = GripperCommand(gripper_id, 0.0)
                continue

            grippers[gripper_id] = GripperCommand(
                gripper_id,
                controller.axis(self.gripper_axis),
            )

            clutch_state = self._states[arm_id]
            if not controller.button(self.clutch_button):
                self._states[arm_id] = _ArmClutchState()
                motions[arm_id] = MotionCommand.disabled(arm_id)
                continue

            if not clutch_state.active or clutch_state.previous_pose is None:
                self._states[arm_id] = _ArmClutchState(
                    active=True,
                    previous_pose=_snapshot_pose(controller.pose),
                )
                motions[arm_id] = MotionCommand(
                    arm_id=arm_id,
                    delta_position=np.zeros(3),
                    delta_rotation=np.zeros(3),
                    enabled=True,
                )
                continue

            raw_position_delta = controller.pose.position - clutch_state.previous_pose.position
            raw_rotation_delta = quaternion_delta_rpy(
                clutch_state.previous_pose.orientation,
                controller.pose.orientation,
            )
            mapped_position = axis_mapping(
                raw_position_delta,
                self.position_axes,
                output_axes=("x", "y", "z"),
                input_axes=("x", "y", "z"),
            )
            mapped_rotation = axis_mapping(
                raw_rotation_delta,
                self.rotation_axes,
                output_axes=("roll", "pitch", "yaw"),
                input_axes=("roll", "pitch", "yaw"),
            )

            delta_position = limit_components(
                apply_deadzone(mapped_position * self.position_scale, self.position_deadzone),
                self.max_translation_per_cycle,
            )
            delta_rotation = limit_components(
                apply_deadzone(mapped_rotation * self.rotation_scale, self.rotation_deadzone),
                self.max_rotation_per_cycle,
            )
            motions[arm_id] = MotionCommand(
                arm_id=arm_id,
                delta_position=delta_position,
                delta_rotation=delta_rotation,
                enabled=True,
            )
            self._states[arm_id] = _ArmClutchState(
                active=True,
                previous_pose=_snapshot_pose(controller.pose),
            )

        return TeleopCommandBatch(motions=motions, grippers=grippers)
