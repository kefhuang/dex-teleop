from __future__ import annotations

import numpy as np

from dex_teleop.core.types import (
    ArmState,
    ControllerState,
    DeviceState,
    GripperCommand,
    MotionCommand,
    Pose,
    TeleopCommandBatch,
)


def test_pose_arrays_are_float_arrays():
    pose = Pose(position=[1, 2, 3], orientation=[0, 0, 0, 1])

    assert pose.position.dtype == float
    assert pose.orientation.dtype == float
    assert pose.position.shape == (3,)
    assert pose.orientation.shape == (4,)


def test_motion_command_defaults_to_disabled_zero_delta():
    command = MotionCommand.disabled("left")

    assert command.arm_id == "left"
    assert command.enabled is False
    assert np.allclose(command.delta_position, np.zeros(3))
    assert np.allclose(command.delta_rotation, np.zeros(3))


def test_gripper_command_clamps_normalized_value():
    low = GripperCommand("left_gripper", -0.2)
    high = GripperCommand("right_gripper", 1.4)

    assert low.normalized == 0.0
    assert high.normalized == 1.0


def test_command_batch_groups_arm_and_gripper_commands():
    batch = TeleopCommandBatch(
        motions={"left": MotionCommand.disabled("left")},
        grippers={"left_gripper": GripperCommand("left_gripper", 0.5)},
    )

    assert set(batch.motions) == {"left"}
    assert set(batch.grippers) == {"left_gripper"}


def test_device_state_contains_controller_states():
    controller = ControllerState(
        name="left",
        pose=Pose.identity(),
        buttons={"grip": True},
        axes={"trigger": 0.25},
    )
    state = DeviceState(timestamp=123.0, controllers={"left": controller})

    assert state.controllers["left"].buttons["grip"] is True
    assert state.controllers["left"].axes["trigger"] == 0.25


def test_arm_state_can_hold_current_pose():
    state = ArmState(arm_id="left", pose=Pose.identity())

    assert state.arm_id == "left"
    assert np.allclose(state.pose.position, np.zeros(3))
