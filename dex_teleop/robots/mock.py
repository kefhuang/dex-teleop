from __future__ import annotations

from dex_teleop.core.types import ArmState, GripperCommand, MotionCommand, Pose


class MockArmController:
    def __init__(self, arm_id: str) -> None:
        self.arm_id = arm_id
        self.connected = False
        self.closed = False
        self.motion_commands: list[MotionCommand] = []

    def connect(self) -> None:
        self.connected = True

    def get_state(self) -> ArmState:
        return ArmState(arm_id=self.arm_id, pose=Pose.identity())

    def send_motion(self, command: MotionCommand) -> None:
        self.motion_commands.append(command)

    def stop_motion(self) -> None:
        self.motion_commands.append(MotionCommand.disabled(self.arm_id))

    def close(self) -> None:
        self.closed = True


class MockGripperController:
    def __init__(self, gripper_id: str) -> None:
        self.gripper_id = gripper_id
        self.connected = False
        self.closed = False
        self.gripper_commands: list[GripperCommand] = []

    def connect(self) -> None:
        self.connected = True

    def send_gripper(self, command: GripperCommand) -> None:
        self.gripper_commands.append(command)

    def close(self) -> None:
        self.closed = True
