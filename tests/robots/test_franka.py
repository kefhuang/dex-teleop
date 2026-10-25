import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import dex_teleop.robots.franka as franka_module
from dex_teleop.core.types import Pose, Twist
from dex_teleop.diagnostics import ControlTracer, trace_scope
from dex_teleop.robots.base import RobotStateStale, RobotStateUnavailable
from dex_teleop.robots.arm import ArmMotionState
from dex_teleop.robots.franka import FRANKA_HOME_JOINTS, FrankaArm, FrankaConnection, FrankaGripper, _IKSolution, _PinocchioKinematics, make_franka_robot
from dex_teleop.utils.transforms import rotation_matrix_to_quaternion_xyzw, rotation_vector_to_matrix, quaternion_xyzw_to_matrix, rotation_error_angle

class Fake:
    def __init__(self):
        self.last_state=object(); self.last_q=np.array([0,0,0,-1,0,0,0],dtype=float); self.last_dq=np.zeros(7); self.last_eef_pose=np.eye(4); self.calls=[]; self.closed=0; self.prepared=[]
    def prepare_controller(self,controller_type): self.prepared.append(controller_type)
    def control(self,**kwargs): self.calls.append(kwargs)
    def gripper_control(self,value): self.calls.append(('gripper',value))
    def send_no_control(self): self.calls.append('stop')
    def close(self): self.closed+=1

def arm(fake=None,**kwargs):
    fake=fake or Fake(); result=FrankaArm(FrankaConnection('unused',fake),config_loader=lambda kind:{},**kwargs); result.connect(); return result,fake

def test_created_interface_disables_implicit_gripper_reset(monkeypatch):
    captured = {}

    class FakeInterface:
        def __init__(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs

    package = ModuleType("deoxys")
    package.__path__ = []
    module = ModuleType("deoxys.franka_interface")
    module.FrankaInterface = FakeInterface
    monkeypatch.setitem(sys.modules, "deoxys", package)
    monkeypatch.setitem(sys.modules, "deoxys.franka_interface", module)
    monkeypatch.setattr(franka_module, "_ensure_deoxys_on_path", lambda: None)

    result = franka_module._make_interface("robot.yml", has_gripper=True)

    assert isinstance(result, FakeInterface)
    assert captured == {
        "path": "robot.yml",
        "kwargs": {
            "has_gripper": True,
            "use_visualizer": False,
            "automatic_gripper_reset": False,
        },
    }

def test_deoxys_controller_preparation_is_idempotent():
    franka_module._ensure_deoxys_on_path()
    from deoxys.franka_interface.franka_interface import FrankaInterface

    interface=object.__new__(FrankaInterface)
    interface._last_controller_type="Dummy"
    preprocess_calls=[]
    interface.preprocess=lambda: preprocess_calls.append("preprocess")

    interface.prepare_controller("JOINT_POSITION")
    interface.prepare_controller("JOINT_POSITION")

    assert preprocess_calls==["preprocess"]
    assert interface._last_controller_type=="JOINT_POSITION"

def test_arm_prepares_initial_joint_position_during_connect():
    value,fake=arm()

    assert value.motion_state is ArmMotionState.HOLDING
    assert fake.prepared==["JOINT_POSITION"]
    assert [call for call in fake.calls if isinstance(call,dict)]==[]

def test_bimanual_startup_prepares_both_arms_before_either_control_command():
    events=[]

    class OrderedFake(Fake):
        def __init__(self,name):
            super().__init__(); self.name=name
        def prepare_controller(self,controller_type):
            events.append((self.name,"prepare",controller_type))
            return super().prepare_controller(controller_type)
        def control(self,**kwargs):
            events.append((self.name,"control",kwargs["controller_type"]))
            return super().control(**kwargs)

    left_fake=OrderedFake("left"); right_fake=OrderedFake("right")
    left=FrankaArm(FrankaConnection("left",left_fake),config_loader=lambda kind:{})
    right=FrankaArm(FrankaConnection("right",right_fake),config_loader=lambda kind:{})

    for candidate in (left,right):
        candidate.connect()
    for candidate in (left,right):
        candidate.stop()

    assert events==[
        ("left","prepare","JOINT_POSITION"),
        ("right","prepare","JOINT_POSITION"),
        ("left","control","JOINT_POSITION"),
        ("right","control","JOINT_POSITION"),
    ]

def test_state_and_three_joint_backends():
    value,fake=arm(max_acceleration=100,max_velocity=100)
    state=value.get_state(); assert state.joint_positions.shape==(7,)
    value.joint_position(np.ones(7),dt=.01); fake.last_q=np.zeros(7)
    value.joint_velocity(np.ones(7),dt=.01); value.joint_impedance(np.ones(7),dt=.01)
    arm_calls=[c for c in fake.calls if isinstance(c,dict)]
    assert [c['controller_type'] for c in arm_calls]==['JOINT_POSITION','JOINT_VELOCITY','JOINT_IMPEDANCE']
    assert all(c['control_gripper'] is False for c in arm_calls)

def test_joint_position_arms_with_measured_position_before_moving():
    value,fake=arm(max_acceleration=1,max_velocity=1)
    fake.last_q=np.arange(7,dtype=float)*.01
    fake.last_q[3]=-1.0
    value.joint_position(np.ones(7),dt=.05)
    first=[c for c in fake.calls if isinstance(c,dict)][0]
    np.testing.assert_allclose(first['action'][:7],fake.last_q)
    np.testing.assert_allclose(first['joint_velocity_reference'],0.0)
    value.joint_position(np.ones(7),dt=.05)
    second=[c for c in fake.calls if isinstance(c,dict)][1]
    assert np.max(np.asarray(second['action'][:7])-fake.last_q) <= .0025+1e-12
    np.testing.assert_allclose(second['joint_velocity_reference'],.05)
    np.testing.assert_allclose(
        np.asarray(second['action'][:7]),
        np.asarray(first['action'][:7])
        + .5*np.asarray(second['joint_velocity_reference'])*.05,
    )

def test_joint_position_shaping_advances_from_previous_safe_command_when_state_lags():
    value,fake=arm(max_acceleration=1,max_velocity=1)
    value.joint_position(np.ones(7),dt=.05)
    value.joint_position(np.ones(7),dt=.05)
    value.joint_position(np.ones(7),dt=.05)
    calls=[c for c in fake.calls if isinstance(c,dict)]
    assert calls[1]['action'][0] == pytest.approx(.00125)
    assert calls[2]['action'][0] == pytest.approx(.005)

def test_home_uses_fixed_franka_target_through_joint_position_shaping():
    value,fake=arm(max_acceleration=1,max_velocity=1)

    value.home(dt=.05)

    np.testing.assert_allclose(value._motion_target_position,FRANKA_HOME_JOINTS)
    assert value.motion_state is ArmMotionState.STREAMING
    assert value.is_settled is False
    first=np.asarray(fake.calls[-1]['action'][:7])
    np.testing.assert_allclose(first,fake.last_q)

    value.home(dt=.05)

    second=np.asarray(fake.calls[-1]['action'][:7])
    assert np.linalg.norm(second-FRANKA_HOME_JOINTS) < np.linalg.norm(first-FRANKA_HOME_JOINTS)

def test_home_validates_dt_before_reading_state():
    value,fake=arm()
    fake.last_state=None

    with pytest.raises(ValueError,match='dt'):
        value.home(dt=0)

def test_home_arms_from_measured_joints_after_velocity_control():
    value,fake=arm()
    value.joint_velocity(np.zeros(7),dt=.05)
    fake.last_q=FRANKA_HOME_JOINTS.copy()
    fake.last_dq=np.zeros(7)
    fake.last_q_d=np.full(7,.5)

    value.home(dt=.05)

    call=[item for item in fake.calls if isinstance(item,dict)][-1]
    assert call['controller_type']=='JOINT_POSITION'
    np.testing.assert_allclose(call['action'][:7],FRANKA_HOME_JOINTS)

def test_streaming_joint_position_uses_full_period_linear_interpolation():
    fake=Fake(); configs=[]
    def load(_kind):
        config={"traj_interpolator_cfg":{"traj_interpolator_type":"SMOOTH_JOINT_POSITION","time_fraction":.3}}
        configs.append(config); return config
    value=FrankaArm(FrankaConnection('unused',fake),config_loader=load); value.connect()
    value.joint_position(np.zeros(7),dt=.05)
    assert configs[0]["traj_interpolator_cfg"]=={"traj_interpolator_type":"LINEAR_JOINT_POSITION","time_fraction":1.0}

def test_stop_streams_one_stable_joint_hold_target():
    value,fake=arm(); fake.last_state=SimpleNamespace(dq_d=np.zeros(7)); value.stop(); value.stop()
    calls=[c for c in fake.calls if isinstance(c,dict)]
    assert [c['controller_type'] for c in calls]==['JOINT_POSITION','JOINT_POSITION']
    np.testing.assert_allclose(calls[0]['action'],calls[1]['action'])
    np.testing.assert_allclose(calls[0]['joint_velocity_reference'],0.0)
    assert value.motion_state is ArmMotionState.HOLDING

def test_stop_reports_holding_immediately_while_reference_settles():
    value,fake=arm(max_acceleration=1.0,max_velocity=1.0)
    target=fake.last_q.copy(); target[0]+=.2
    value.joint_position(target,dt=.05)
    value.joint_position(target,dt=.05)
    moving_call=[call for call in fake.calls if isinstance(call,dict)][-1]
    fake.last_q_d=np.asarray(moving_call['action'][:7])
    fake.last_dq=np.asarray(moving_call['joint_velocity_reference'])
    fake.last_state=SimpleNamespace(dq_d=fake.last_dq.copy())

    value.stop()

    assert value.motion_state is ArmMotionState.HOLDING
    assert value.is_settled is False
    np.testing.assert_allclose(value._hold_position,fake.last_q_d)

def test_trace_captures_backend_state_shaping_and_transmitted_action(tmp_path):
    value,_=arm(max_acceleration=1,max_velocity=1)
    path=tmp_path/'franka-trace.jsonl'
    tracer=ControlTracer(path)
    with tracer.span('control_cycle',iteration=0):
        with trace_scope(arm_id='right'):
            value.joint_position(np.ones(7),dt=.05)
            value.joint_position(np.ones(7),dt=.05)
    tracer.close()

    import json
    records=[json.loads(line) for line in path.read_text().splitlines()]
    cycle=next(record for record in records if record['record_type']=='control_cycle')
    events=cycle['events']
    names=[event['name'] for event in events]
    assert 'franka.backend.state.read' in names
    assert 'franka.arm.joint_position.shaped' in names
    assert 'franka.backend.command.sent' in names
    sent=[event for event in events if event['name']=='franka.backend.command.sent'][-1]
    assert sent['scope']['arm_id']=='right'
    assert sent['data']['controller_type']=='JOINT_POSITION'
    assert len(sent['data']['action'])==8

def test_state_errors_are_explicit(monkeypatch):
    value,fake=arm(); fake.last_state=None
    with pytest.raises(RobotStateUnavailable): value.get_state()
    fake.last_state=object(); value.get_state(); monkeypatch.setattr('dex_teleop.robots.franka.time.monotonic',lambda: 10.)
    value._state_time=0
    with pytest.raises(RobotStateStale): value.get_state()

def test_gripper_and_shared_connection_close_once():
    fake=Fake(); robot=make_franka_robot('unused',interface=fake,config_loader=lambda kind:{})
    robot.connect(); robot.end_effector.position(.75); robot.close(); robot.close()
    assert ('gripper',-1.0) in fake.calls and fake.closed==1

def test_arm_hold_does_not_send_an_implicit_gripper_close():
    fake=Fake(); robot=make_franka_robot('unused',interface=fake,config_loader=lambda kind:{})
    robot.connect()

    robot.arm.stop()
    robot.end_effector.position(1.0)

    arm_calls=[call for call in fake.calls if isinstance(call,dict)]
    assert arm_calls and all(call['control_gripper'] is False for call in arm_calls)
    assert [call for call in fake.calls if isinstance(call,tuple)]==[('gripper',-1.0)]

def test_gripper_uses_hysteresis_and_sends_only_state_changes():
    fake=Fake(); grip=FrankaGripper(FrankaConnection('unused',fake)); grip.connect()
    for value in (1.0,.9,.7,.55,.45,.3,.2,.1): grip.position(value)
    assert [call for call in fake.calls if isinstance(call,tuple)]==[('gripper',-1.0),('gripper',1.0)]

def test_differential_ik_uses_damped_jacobian():
    class Solver:
        def jacobian(self,q): return np.c_[np.eye(6),np.zeros(6)]
    value,_=arm(ik_solver=Solver()); result=value._solve_differential_ik(Twist([1,0,0],[0,0,0]),np.zeros(7))
    assert result.shape==(7,) and result[0]>0

def test_pose_and_twist_ik_streams_paired_joint_position_and_velocity_reference():
    class Solver:
        def solve(self,seed,measured,target):
            return _IKSolution(np.asarray(seed).copy(),0.0,0.0)
        def jacobian(self,q):
            return np.c_[np.eye(6),np.zeros(6)]
    value,fake=arm(ik_solver=Solver(),max_acceleration=1.0,max_velocity=1.0)
    target=Pose([.005,0,0],[0,0,0,1])
    twist=Twist([.2,0,0],[0,0,0])

    value.ik_joint_position(target,twist_ref=twist,dt=.05)
    value.ik_joint_position(target,twist_ref=twist,dt=.05)

    calls=[call for call in fake.calls if isinstance(call,dict)]
    np.testing.assert_allclose(calls[0]['joint_velocity_reference'],0.0)
    reference=np.asarray(calls[1]['joint_velocity_reference'])
    assert reference[0] == pytest.approx(.05)
    np.testing.assert_allclose(reference[1:],0.0)
    assert value.last_ik_joint_velocities[0] == pytest.approx(.2/(1.0+.05**2))
    assert value.motion_state is ArmMotionState.STREAMING

    fake.last_dq[0]=.1
    value.ik_joint_position(target,twist_ref=Twist([0,0,0],[0,0,0]),dt=.05)
    assert value.motion_state is ArmMotionState.STREAMING
    assert value.is_settled is False
    third=[call for call in fake.calls if isinstance(call,dict)][-1]
    np.testing.assert_allclose(
        np.asarray(third['action'][:7])-np.asarray(calls[1]['action'][:7]),
        .5*(np.asarray(calls[1]['joint_velocity_reference'])+np.asarray(third['joint_velocity_reference']))*.05,
    )

def test_static_joint_reference_settles_to_holding_without_position_velocity_split():
    value,fake=arm(max_acceleration=1.0,max_velocity=1.0)
    target=fake.last_q.copy(); target[0]+=.05
    value.joint_position(target,dt=.05)

    previous_position=np.asarray(fake.calls[-1]['action'][:7])
    previous_velocity=np.asarray(fake.calls[-1]['joint_velocity_reference'])
    for _ in range(200):
        fake.last_q=previous_position.copy()
        fake.last_dq=previous_velocity.copy()
        fake.last_state=SimpleNamespace(dq_d=previous_velocity.copy())
        value.joint_position(target,dt=.05)
        call=fake.calls[-1]
        position=np.asarray(call['action'][:7])
        velocity=np.asarray(call['joint_velocity_reference'])
        np.testing.assert_allclose(
            position-previous_position,
            .5*(previous_velocity+velocity)*.05,
            atol=1e-12,
        )
        previous_position=position
        previous_velocity=velocity
        if value.motion_state is ArmMotionState.HOLDING:
            break

    assert value.motion_state is ArmMotionState.HOLDING
    np.testing.assert_allclose(previous_position,target,atol=1e-6)
    np.testing.assert_allclose(previous_velocity,0.0,atol=1e-6)

def test_streaming_joint_reference_integrates_one_coupled_position_velocity_state():
    value,fake=arm(max_acceleration=1.0,max_velocity=1.0)
    start=fake.last_q.copy()
    value.joint_position(start,dt=.05)

    previous_position=np.asarray(fake.calls[-1]['action'][:7])
    previous_velocity=np.asarray(fake.calls[-1]['joint_velocity_reference'])
    transmitted_velocities=[]
    for sample in range(1,21):
        fake.last_q=previous_position.copy()
        fake.last_dq=previous_velocity.copy()
        fake.last_state=SimpleNamespace(dq_d=previous_velocity.copy())
        target=start.copy(); target[0]+=.2*sample*.05
        feedforward=np.zeros(7); feedforward[0]=.2
        state=value.get_state()
        value._joint_position_reference(
            target,
            feedforward,
            step=.05,
            state=state,
            streaming=True,
        )
        call=fake.calls[-1]
        position=np.asarray(call['action'][:7])
        velocity=np.asarray(call['joint_velocity_reference'])
        np.testing.assert_allclose(
            position-previous_position,
            .5*(previous_velocity+velocity)*.05,
            atol=1e-12,
        )
        transmitted_velocities.append(velocity[0])
        previous_position=position
        previous_velocity=velocity

    assert value.motion_state is ArmMotionState.STREAMING
    assert min(transmitted_velocities) > 0.0
    assert transmitted_velocities[-1] == pytest.approx(.2,abs=.01)

def test_pinocchio_ik_stays_near_seed_for_small_base_frame_target():
    solver=_PinocchioKinematics(); seed=np.array([.09,-.2,-.02,-2.47,-.01,2.3,.85])
    measured=solver.forward_pose(seed)
    target=Pose(measured.position+np.array([.005,-.003,.004]),measured.quaternion_xyzw)
    solution=solver.solve(seed,measured,target)
    assert solution.position_residual_m < .002
    assert np.max(np.abs(solution.joint_positions-seed)) < .2

def test_pinocchio_ik_tracks_small_rotation_without_branch_jump():
    solver=_PinocchioKinematics(); seed=np.array([.09,-.2,-.02,-2.47,-.01,2.3,.85])
    measured=solver.forward_pose(seed); current=quaternion_xyzw_to_matrix(measured.quaternion_xyzw)
    target=Pose(measured.position,rotation_matrix_to_quaternion_xyzw(rotation_vector_to_matrix([.01,-.015,.02])@current))
    solution=solver.solve(seed,measured,target)
    assert solution.rotation_residual_rad < .01
    assert np.max(np.abs(solution.joint_positions-seed)) < .2

def test_absolute_ik_advances_through_bounded_cartesian_subtarget():
    class Solver:
        def solve(self,seed,measured,target):
            self.measured=measured; self.target=target
            return _IKSolution(np.asarray(seed).copy(),0.0,0.0)
    solver=Solver(); value,fake=arm(ik_solver=solver,max_ik_translation_step=.01)
    value.ik_joint_position(Pose([.05,0,0],[0,0,0,1]),dt=.05)
    assert np.linalg.norm(solver.target.position-solver.measured.position)==pytest.approx(.01)
