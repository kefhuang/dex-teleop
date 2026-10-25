import numpy as np
import pytest
from dex_teleop.core.types import Pose, Twist
from dex_teleop.robots import MockArm, MockGripper, Robot

def test_mock_implements_direct_arm_api_and_rate_shapes():
    arm=MockArm(max_velocity=1,max_acceleration=2); arm.connect()
    arm.joint_position(np.ones(7),dt=.1)
    np.testing.assert_allclose(arm.get_state().joint_positions,.02)
    arm.joint_velocity(np.ones(7),dt=.1)
    assert arm.commands[-1][0]=='joint_velocity'
    arm.joint_impedance(np.zeros(7),dt=.1)
    assert arm.commands[-1][0]=='joint_impedance'

def test_mock_home_is_a_fixed_rate_shaped_action():
    home=np.full(7,.5)
    arm=MockArm(home_positions=home,max_velocity=1,max_acceleration=10)
    arm.connect()

    arm.home(dt=.1)

    assert arm.commands[-1][0]=='HOME'
    np.testing.assert_allclose(arm.get_state().joint_positions,.1)

def test_mock_ik_entry_points():
    arm=MockArm(); arm.connect(); arm.ik_joint_position(Pose([.1,.2,.3],[0,0,0,1]),dt=.1)
    arm.ik_joint_velocity(Twist([.1,0,0],[0,0,0]),dt=.1)
    assert len(arm.commands)==2

def test_mock_validates_dt_and_gripper_range():
    arm=MockArm(); arm.connect()
    with pytest.raises(ValueError,match='dt'): arm.joint_position(np.zeros(7),dt=0)
    grip=MockGripper(); grip.connect()
    with pytest.raises(ValueError): grip.position(1.1)

def test_robot_lifecycle_and_close_is_idempotent():
    arm=MockArm(); grip=MockGripper(); robot=Robot(arm,grip); robot.connect(); robot.close(); robot.close()
    assert arm.stopped and grip.stopped
