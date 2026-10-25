import numpy as np
import pytest
from dex_teleop.core.types import ArmState, ControllerAction, ControllerState, EndEffectorState, Pose, Twist

def test_pose_normalizes_xyzw_and_copies():
    source=np.array([0.,0.,0.,2.]); pose=Pose([1,2,3],source); source[3]=0
    np.testing.assert_allclose(pose.quaternion_xyzw,[0,0,0,1])

def test_pose_rejects_invalid_values():
    with pytest.raises(ValueError): Pose([0,0,np.nan],[0,0,0,1])
    with pytest.raises(ValueError): Pose([0,0,0],[0,0,0,0])

def test_twist_and_arm_state_validate_shapes():
    twist=Twist([1,2,3],[4,5,6]); assert twist.linear.shape==(3,)
    state=ArmState(np.zeros(7),np.ones(7),Pose.identity(),1.0); assert state.joint_velocities.shape==(7,)
    with pytest.raises(ValueError): ArmState(np.zeros(7),np.zeros(6),Pose.identity(),1.0)

def test_end_effector_state_validates_finite():
    assert EndEffectorState(1.0,position=.5).position==.5
    with pytest.raises(ValueError): EndEffectorState(float('nan'))

def test_controller_state_carries_optional_twist_and_source_timestamp():
    twist=Twist([1,2,3],[4,5,6])
    state=ControllerState('right',Pose.identity(),twist=twist,timestamp=1.25)
    assert state.twist is twist and state.timestamp==1.25
    with pytest.raises(ValueError): ControllerState('right',Pose.identity(),timestamp=np.nan)

def test_controller_state_exposes_fixed_actions():
    state=ControllerState(
        'left',Pose.identity(),actions={ControllerAction.HOME,ControllerAction.CALIBRATE}
    )

    assert state.action(ControllerAction.HOME) is True
    assert state.action(ControllerAction.CALIBRATE) is True
    assert isinstance(state.actions,frozenset)
    with pytest.raises(TypeError,match='ControllerAction'):
        ControllerState('left',Pose.identity(),actions={'HOME'})
    with pytest.raises(TypeError,match='ControllerAction'):
        state.action('HOME')

def test_controller_state_normalizes_optional_pointer_direction():
    source=np.array([0.,0.,2.])
    state=ControllerState('right',Pose.identity(),pointer_direction=source)
    source[2]=0.

    np.testing.assert_allclose(state.pointer_direction,[0,0,1])
    assert state.pointer_direction.flags.writeable is False
    with pytest.raises(ValueError,match='non-zero'):
        ControllerState('right',Pose.identity(),pointer_direction=np.zeros(3))
