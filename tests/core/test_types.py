from __future__ import annotations

import numpy as np
import pytest

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


def test_pose_identity_uses_zero_position_and_unit_w_quaternion():
    pose = Pose.identity()

    assert np.allclose(pose.position, np.zeros(3))
    assert np.allclose(pose.orientation, np.array([0.0, 0.0, 0.0, 1.0]))


def test_pose_rejects_invalid_position_and_orientation_shapes():
    with pytest.raises(ValueError, match="position must have shape"):
        Pose(position=[1.0, 2.0], orientation=[0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="orientation must have shape"):
        Pose(position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 1.0])


def test_motion_command_defaults_to_disabled_zero_delta():
    command = MotionCommand.disabled("left")

    assert command.arm_id == "left"
    assert command.enabled is False
    assert np.allclose(command.delta_position, np.zeros(3))
    assert np.allclose(command.delta_rotation, np.zeros(3))


def test_controller_state_accessors_return_existing_values_and_defaults():
    controller = ControllerState(
        name="left",
        pose=Pose.identity(),
        buttons={"grip": True, "menu": False},
        axes={"trigger": 0.25},
    )

    assert controller.button("grip") is True
    assert controller.button("menu") is False
    assert controller.button("missing") is False
    assert controller.axis("trigger") == 0.25
    assert controller.axis("missing") == 0.0


def test_motion_command_rejects_invalid_delta_shapes():
    with pytest.raises(ValueError, match="delta_position must have shape"):
        MotionCommand(
            arm_id="left",
            delta_position=[0.0, 0.0],
            delta_rotation=[0.0, 0.0, 0.0],
        )

    with pytest.raises(ValueError, match="delta_rotation must have shape"):
        MotionCommand(
            arm_id="left",
            delta_position=[0.0, 0.0, 0.0],
            delta_rotation=[0.0, 0.0],
        )


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
