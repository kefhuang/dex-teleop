from __future__ import annotations

from dex_teleop.core.types import GripperCommand, MotionCommand
from dex_teleop.robots.franka import FrankaArmController, FrankaGripperController


class FakeFrankaInterface:
    def __init__(self):
        self.control_calls = []
        self.gripper_calls = []
        self.closed = False

    def control(self, controller_type, action, controller_cfg=None, termination=False):
        self.control_calls.append((controller_type, list(action), controller_cfg, termination))

    def gripper_control(self, action):
        self.gripper_calls.append(action)

    def close(self):
        self.closed = True


def test_franka_arm_sends_osc_pose_action_with_neutral_gripper_slot():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.send_motion(MotionCommand("left", [0.1, 0.2, 0.3], [0.01, 0.02, 0.03], True))

    assert fake.control_calls[-1] == (
        "OSC_POSE",
        [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.0],
        {},
        False,
    )


def test_franka_arm_stop_sends_zero_termination_action():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.stop_motion()

    assert fake.control_calls[-1] == (
        "OSC_POSE",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        {},
        True,
    )


def test_franka_arm_disabled_motion_sends_zero_termination_action():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.send_motion(MotionCommand.disabled("left"))

    assert fake.control_calls[-1] == (
        "OSC_POSE",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        {},
        True,
    )


def test_franka_arm_close_calls_interface_close():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.close()

    assert fake.closed is True


def test_franka_gripper_converts_normalized_to_deoxys_action():
    fake = FakeFrankaInterface()
    gripper = FrankaGripperController("left_gripper", interface=fake)
    gripper.connect()
    gripper.send_gripper(GripperCommand("left_gripper", 1.0))
    gripper.send_gripper(GripperCommand("left_gripper", 0.0))

    assert fake.gripper_calls == [1.0, -1.0]
