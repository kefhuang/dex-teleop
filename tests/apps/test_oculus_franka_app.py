import json

import numpy as np

import dex_teleop.apps.oculus_franka as app_module
from dex_teleop.apps.oculus_franka import _make_robots, main, parse_args, run
from dex_teleop.core.config import load_config
from dex_teleop.core.types import ControllerAction, ControllerState, DeviceState, Pose, Twist
from dex_teleop.diagnostics import ControlTracer
from dex_teleop.robots import MockArm, MockGripper
from dex_teleop.robots.arm import ArmMotionState
from dex_teleop.robots.robot import Robot

def test_dry_run_factory_builds_robot_composition():
    robots=_make_robots(load_config('configs/oculus_franka_right.yml'))
    assert isinstance(robots['right'].arm,MockArm)
    assert isinstance(robots['right'].end_effector,MockGripper)

def test_dry_run_control_loop_smoke():
    cfg=load_config('configs/oculus_franka_right.yml')
    assert run(cfg,duration=.01)==0

def test_visualizer_receives_device_samples_and_is_closed():
    class FakeVisualizer:
        def __init__(self):
            self.states=[]
            self.closed=False

        def update(self,state):
            self.states.append(state)

        def close(self):
            self.closed=True

    cfg=load_config('configs/oculus_franka_right.yml')
    cfg['app']['rate_hz']=1000
    visualizer=FakeVisualizer()

    assert run(cfg,duration=.005,visualizer=visualizer)==0
    assert visualizer.states
    assert set(visualizer.states[0].controllers)=={'left','right'}
    assert visualizer.closed

def test_parse_args_accepts_controller_visualization_flag():
    args=parse_args([
        '--config','configs/oculus_franka_right.yml',
        '--visualize-controllers',
    ])

    assert args.visualize_controllers is True

def test_main_reports_missing_config():
    assert main(['--config','does-not-exist.yml'])==2

def test_control_trace_captures_complete_mock_control_path(tmp_path):
    cfg=load_config('configs/oculus_franka_right.yml')
    cfg['app']['rate_hz']=1000
    path=tmp_path/'control-trace.jsonl'
    tracer=ControlTracer(path)
    try:
        assert run(cfg,duration=.02,tracer=tracer)==0
    finally:
        tracer.close()

    records=[json.loads(line) for line in path.read_text().splitlines()]
    cycles=[record for record in records if record['record_type']=='control_cycle']
    events=[event for cycle in cycles for event in cycle['events']]
    names={event['name'] for event in events}

    assert len(cycles)>=3
    assert 'device.read.result' in names
    assert 'mock.arm.state.result' in names
    assert 'teleop.clutch.target_built' in names
    assert 'mock.arm.ik_joint_position.solved' in names
    assert 'mock.arm.command.executed' in names
    assert 'mock.gripper.position.executed' in names
    command=next(event for event in events if event['name']=='mock.arm.command.executed')
    assert command['scope']['arm_id']=='right'
    assert len(command['data']['joint_positions'])==7

def test_main_record_trace_option_writes_session_lifecycle(tmp_path):
    path=tmp_path/'cli-trace.jsonl'
    result=main([
        '--config','configs/oculus_franka_right.yml',
        '--rate-hz','1000',
        '--duration','.005',
        '--record-trace',str(path),
    ])

    assert result==0
    records=[json.loads(line) for line in path.read_text().splitlines()]
    event_names=[record.get('name') for record in records if record['record_type']=='event']
    assert 'session.start' in event_names
    assert 'session.end' in event_names
    assert any(record['record_type']=='control_cycle' for record in records)

def test_real_oculus_filter_cutoff_stays_below_robot_loop_nyquist(monkeypatch):
    captured={}
    class FakeOculus:
        def __init__(self,**kwargs): captured.update(kwargs)
    monkeypatch.setattr(app_module,'OculusDevice',FakeOculus)
    cfg=load_config('configs/oculus_franka_right.yml')
    cfg['input']['mock_controller']=False
    cfg['app']['rate_hz']=8.0

    app_module._make_device(cfg)

    assert captured['velocity_cutoff_hz']==3.6

def test_control_loop_passes_zero_then_mapped_controller_twist(monkeypatch):
    raw_twist=Twist([1,2,3],[.1,.2,.3])
    class FakeDevice:
        def __init__(self): self.step=0
        def start(self): pass
        def read(self):
            self.step+=1
            calibrating=self.step==1
            controller=ControllerState(
                'right',Pose.identity(),buttons={'grip':not calibrating},
                twist=raw_twist,timestamp=float(self.step),
                actions=(
                    {ControllerAction.CALIBRATE} if calibrating else set()
                ),
                pointer_direction=[0,0,1],
            )
            return DeviceState(float(self.step),{'right':controller})
        def stop(self): pass
    class SpyArm(MockArm):
        def __init__(self):
            super().__init__(); self.twist_references=[]
        def ik_joint_position(self,pose,*,twist_ref=None,delta=False,dt):
            self.twist_references.append(twist_ref)
            return super().ik_joint_position(
                pose,twist_ref=twist_ref,delta=delta,dt=dt
            )
    arm=SpyArm(); robot=Robot(arm,None)
    monkeypatch.setattr(app_module,'_make_device',lambda _cfg:FakeDevice())
    monkeypatch.setattr(app_module,'_make_robots',lambda _cfg:{'right':robot})
    cfg=load_config('configs/oculus_franka_right.yml')
    cfg['app']['rate_hz']=1000.0
    cfg['scales']['rotation']=.3

    assert app_module.run(cfg,duration=.006)==0

    assert len(arm.twist_references)>=2
    np.testing.assert_allclose(arm.twist_references[0].linear,0.0)
    np.testing.assert_allclose(arm.twist_references[0].angular,0.0)
    np.testing.assert_allclose(arm.twist_references[1].linear,[2.4,.8,1.6])
    np.testing.assert_allclose(arm.twist_references[1].angular,[.09,.03,.06])

def test_left_y_and_right_b_calibrate_only_their_bound_arms(monkeypatch):
    class FakeDevice:
        def __init__(self): self.step=0
        def start(self): pass
        def read(self):
            self.step+=1
            left_calibrating=self.step==1
            right_calibrating=self.step==3
            left=ControllerState(
                'left',Pose.identity(),
                buttons={'grip':self.step in {2} or self.step>=4},
                actions=(
                    {ControllerAction.CALIBRATE}
                    if left_calibrating else set()
                ),
                pointer_direction=[0,0,1],
            )
            right=ControllerState(
                'right',Pose.identity(),
                buttons={'grip':self.step==2 or self.step>=4},
                actions=(
                    {ControllerAction.CALIBRATE}
                    if right_calibrating else set()
                ),
                pointer_direction=[0,0,-1],
            )
            return DeviceState(
                float(self.step),{'left':left,'right':right}
            )
        def stop(self): pass
    class SpyArm(MockArm):
        def __init__(self,device):
            super().__init__(); self.device=device; self.ik_steps=[]
        def ik_joint_position(self,pose,*,twist_ref=None,delta=False,dt):
            self.ik_steps.append(self.device.step)
            return super().ik_joint_position(
                pose,twist_ref=twist_ref,delta=delta,dt=dt
            )

    device=FakeDevice()
    left_arm=SpyArm(device); right_arm=SpyArm(device)
    robots={
        'left':Robot(left_arm,None),
        'right':Robot(right_arm,None),
    }
    monkeypatch.setattr(app_module,'_make_device',lambda _cfg:device)
    monkeypatch.setattr(app_module,'_make_robots',lambda _cfg:robots)
    cfg=load_config('configs/oculus_franka_bimanual.yml')
    cfg['app']['rate_hz']=1000.0

    assert app_module.run(cfg,duration=.01)==0

    assert 2 in left_arm.ik_steps
    assert 2 not in right_arm.ik_steps
    assert min(right_arm.ik_steps)>=4

def test_home_action_latches_after_button_release_and_preempts_clutch(monkeypatch):
    class FakeDevice:
        def __init__(self): self.step=0
        def start(self): pass
        def read(self):
            self.step+=1
            actions=(
                {ControllerAction.HOME}
                if self.step==1
                else set()
            )
            controller=ControllerState(
                'right',Pose.identity(),buttons={'grip':True},actions=actions,
                timestamp=float(self.step),
            )
            return DeviceState(float(self.step),{'right':controller})
        def stop(self): pass
    class SpyArm(MockArm):
        def __init__(self):
            super().__init__(); self.home_calls=0; self.ik_calls=0
        def home(self,*,dt):
            self.home_calls+=1
            self._motion_state=ArmMotionState.STREAMING
            self._is_settled=False
        def ik_joint_position(self,pose,*,twist_ref=None,delta=False,dt):
            self.ik_calls+=1
    arm=SpyArm(); robot=Robot(arm,None)
    monkeypatch.setattr(app_module,'_make_device',lambda _cfg:FakeDevice())
    monkeypatch.setattr(app_module,'_make_robots',lambda _cfg:{'right':robot})
    cfg=load_config('configs/oculus_franka_right.yml')
    cfg['app']['rate_hz']=1000.0

    assert app_module.run(cfg,duration=.006)==0

    assert arm.home_calls>=2
    assert arm.ik_calls==0

def test_held_home_action_triggers_only_once_after_completion(monkeypatch):
    class FakeDevice:
        def start(self): pass
        def read(self):
            controller=ControllerState(
                'right',Pose.identity(),actions={ControllerAction.HOME}
            )
            return DeviceState(1.0,{'right':controller})
        def stop(self): pass
    class SpyArm(MockArm):
        def __init__(self): super().__init__(); self.home_calls=0
        def home(self,*,dt):
            self.home_calls+=1
            self._motion_state=ArmMotionState.HOLDING
            self._is_settled=True
    arm=SpyArm(); robot=Robot(arm,None)
    monkeypatch.setattr(app_module,'_make_device',lambda _cfg:FakeDevice())
    monkeypatch.setattr(app_module,'_make_robots',lambda _cfg:{'right':robot})
    cfg=load_config('configs/oculus_franka_right.yml')
    cfg['app']['rate_hz']=1000.0

    assert app_module.run(cfg,duration=.006)==0

    assert arm.home_calls==1
