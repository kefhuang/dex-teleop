from __future__ import annotations

from dex_teleop.core.types import GripperCommand, MotionCommand
from dex_teleop.robots.mock import MockArmController, MockGripperController


def test_mock_arm_records_motion_and_stop():
    arm = MockArmController("left")
    arm.connect()
    arm.send_motion(MotionCommand("left", [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], True))
    arm.stop_motion()

    assert arm.connected is True
    assert len(arm.motion_commands) == 2
    assert arm.motion_commands[0].enabled is True
    assert arm.motion_commands[1].arm_id == "left"
    assert arm.motion_commands[1].enabled is False


def test_mock_gripper_records_commands():
    gripper = MockGripperController("left_gripper")
    gripper.connect()
    command = GripperCommand("left_gripper", 0.75)
    gripper.send_gripper(command)

    assert gripper.connected is True
    assert gripper.gripper_commands[-1] is command
