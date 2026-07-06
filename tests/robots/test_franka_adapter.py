from __future__ import annotations

import sys
import types

import numpy as np

from dex_teleop.core.types import GripperCommand, MotionCommand
from dex_teleop.robots.franka import FrankaArmController, FrankaGripperController


class FakeFrankaInterface:
    def __init__(self):
        self.control_calls = []
        self.gripper_calls = []
        self.closed = False
        self.no_control_calls = 0

    def send_no_control(self):
        self.no_control_calls += 1

    def control(self, controller_type, action, controller_cfg=None, termination=False):
        self.control_calls.append((controller_type, list(action), controller_cfg, termination))

    def gripper_control(self, action):
        self.gripper_calls.append(action)

    def close(self):
        self.closed = True


def test_franka_arm_get_state_reads_last_eef_pose_from_interface():
    fake = FakeFrankaInterface()
    fake.last_eef_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})

    state = arm.get_state()

    np.testing.assert_allclose(state.pose.position, [0.4, -0.2, 0.3])
    np.testing.assert_allclose(state.pose.orientation, [0.0, 0.0, 0.0, 1.0])


def test_franka_arm_get_state_returns_identity_when_no_eef_pose_available():
    fake = FakeFrankaInterface()
    fake.last_eef_pose = None
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})

    state = arm.get_state()

    np.testing.assert_allclose(state.pose.position, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.pose.orientation, [0.0, 0.0, 0.0, 1.0])


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


def test_franka_arm_stop_sends_zero_hold_action():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.stop_motion()

    assert fake.no_control_calls == 1
    assert fake.control_calls == []


def test_franka_arm_disabled_motion_sends_zero_hold_action():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.send_motion(MotionCommand.disabled("left"))

    assert fake.no_control_calls == 1
    assert fake.control_calls == []


def test_franka_arm_close_marks_injected_interface_closed_without_closing_interface():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.close()

    assert arm.closed is True
    assert fake.closed is False


def test_franka_arm_connect_constructs_owned_interface_without_gripper(monkeypatch):
    constructor_calls = []

    class ConstructedFakeFrankaInterface(FakeFrankaInterface):
        def __init__(self, *args, **kwargs):
            super().__init__()
            constructor_calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "deoxys", types.ModuleType("deoxys"))
    monkeypatch.setitem(
        sys.modules,
        "deoxys.franka_interface",
        types.SimpleNamespace(FrankaInterface=ConstructedFakeFrankaInterface),
    )
    monkeypatch.setitem(
        sys.modules,
        "deoxys.utils.config_utils",
        types.SimpleNamespace(get_default_controller_config=lambda controller_type: {"type": controller_type}),
    )

    arm = FrankaArmController("left", interface_cfg="robot.yml")
    arm.connect()

    assert constructor_calls == [
        (("robot.yml",), {"has_gripper": False, "use_visualizer": False})
    ]
    assert arm.controller_cfg == {"type": "OSC_POSE"}


def test_franka_arm_close_closes_owned_constructed_interface(monkeypatch):
    constructed = []

    class ConstructedFakeFrankaInterface(FakeFrankaInterface):
        def __init__(self, *args, **kwargs):
            super().__init__()
            constructed.append(self)

    monkeypatch.setitem(sys.modules, "deoxys", types.ModuleType("deoxys"))
    monkeypatch.setitem(
        sys.modules,
        "deoxys.franka_interface",
        types.SimpleNamespace(FrankaInterface=ConstructedFakeFrankaInterface),
    )
    monkeypatch.setitem(
        sys.modules,
        "deoxys.utils.config_utils",
        types.SimpleNamespace(get_default_controller_config=lambda controller_type: {}),
    )

    arm = FrankaArmController("left", interface_cfg="robot.yml")
    arm.connect()
    arm.close()

    assert arm.closed is True
    assert constructed[0].closed is True


def test_franka_gripper_converts_normalized_to_deoxys_action():
    fake = FakeFrankaInterface()
    gripper = FrankaGripperController("left_gripper", interface=fake)
    gripper.connect()
    gripper.send_gripper(GripperCommand("left_gripper", 1.0))
    gripper.send_gripper(GripperCommand("left_gripper", 0.0))

    assert fake.gripper_calls == [1.0, -1.0]

def test_franka_gripper_uses_arm_interface_after_arm_connect_with_injected_fake():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})

    arm.connect()
    gripper = FrankaGripperController("left_gripper", interface=arm.interface)
    gripper.connect()
    gripper.send_gripper(GripperCommand("left_gripper", 0.75))

    assert arm.interface is fake
    assert fake.gripper_calls == [0.5]
