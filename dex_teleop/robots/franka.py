from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pinocchio as pin

from dex_teleop.core.types import ArmState, EndEffectorState, Pose, Twist
from dex_teleop.diagnostics import trace_event
from dex_teleop.robots.arm import ArmMotionState
from dex_teleop.robots.base import (IKError, IKUnreachableError, RobotCommandError,
    RobotConnectionError, RobotStateStale, RobotStateUnavailable)
from dex_teleop.robots.franka_constants import FRANKA_HOME_JOINTS
from dex_teleop.robots.robot import Robot
from dex_teleop.utils.transforms import quaternion_xyzw_to_matrix, rotation_matrix_to_quaternion_xyzw

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
PANDA_LOWER = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
PANDA_UPPER = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

@dataclass(frozen=True)
class _IKSolution:
    joint_positions: np.ndarray
    position_residual_m: float
    rotation_residual_rad: float

@dataclass(frozen=True)
class _JointReferenceState:
    """One internally consistent position/velocity sample for streaming."""
    positions: np.ndarray
    velocities: np.ndarray

class _PinocchioKinematics:
    """Seed-relative Franka FK, Jacobian, and continuous pose IK."""
    def __init__(self, *, max_iterations=120, damping=.02, max_iteration_step=.02,
                 max_seed_delta=.2, posture_gain=.05):
        urdf=Path(__file__).resolve().parents[2]/"dependency"/"deoxys_control"/"deoxys"/"deoxys"/"franka_interface"/"robot_models"/"panda"/"panda.urdf"
        self.model=pin.buildModelFromUrdf(str(urdf)); self.data=self.model.createData()
        self.frame_id=self.model.getFrameId("panda_grasptarget")
        if self.frame_id >= len(self.model.frames): raise RuntimeError("Panda end-effector frame is missing from URDF")
        self.max_iterations=int(max_iterations); self.damping=float(damping)
        self.max_iteration_step=float(max_iteration_step); self.max_seed_delta=float(max_seed_delta); self.posture_gain=float(posture_gain)
    def _joints(self,value):
        value=np.asarray(value,dtype=float)
        if value.shape!=(7,) or not np.all(np.isfinite(value)): raise ValueError("joint_positions must be a finite shape-(7,) vector")
        return value
    def forward_pose(self,joints):
        q=np.r_[self._joints(joints),.04,.04]
        pin.forwardKinematics(self.model,self.data,q); pin.updateFramePlacements(self.model,self.data)
        placement=self.data.oMf[self.frame_id]
        return Pose(np.asarray(placement.translation).copy(),rotation_matrix_to_quaternion_xyzw(np.asarray(placement.rotation)))
    def jacobian(self,joints):
        q=np.r_[self._joints(joints),.04,.04]
        pin.forwardKinematics(self.model,self.data,q); pin.updateFramePlacements(self.model,self.data)
        return np.asarray(pin.computeFrameJacobian(self.model,self.data,q,self.frame_id,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:,:7].copy()
    def solve(self,seed,measured,target):
        seed=self._joints(seed); model_pose=self.forward_pose(seed)
        model_rotation=quaternion_xyzw_to_matrix(model_pose.quaternion_xyzw)
        measured_rotation=quaternion_xyzw_to_matrix(measured.quaternion_xyzw)
        target_rotation=quaternion_xyzw_to_matrix(target.quaternion_xyzw)
        target_position=model_pose.position+target.position-measured.position
        target_model_rotation=target_rotation@measured_rotation.T@model_rotation
        solved=seed.copy()
        for _ in range(self.max_iterations):
            actual=self.forward_pose(solved)
            position_error=target_position-actual.position
            current_rotation=quaternion_xyzw_to_matrix(actual.quaternion_xyzw)
            rotation_error=np.asarray(pin.log3(target_model_rotation@current_rotation.T))
            if np.linalg.norm(position_error)<=1e-4 and np.linalg.norm(rotation_error)<=1e-3: break
            jacobian=self.jacobian(solved)
            error=np.r_[position_error,rotation_error]
            inverse=jacobian.T@np.linalg.solve(jacobian@jacobian.T+(self.damping**2)*np.eye(6),np.eye(6))
            step=inverse@error
            nullspace=np.eye(7)-inverse@jacobian
            step += nullspace@(-self.posture_gain*(solved-seed))
            largest=float(np.max(np.abs(step)))
            if largest>self.max_iteration_step: step*=self.max_iteration_step/largest
            solved=np.clip(solved+step,np.maximum(PANDA_LOWER,seed-self.max_seed_delta),np.minimum(PANDA_UPPER,seed+self.max_seed_delta))
        actual=self.forward_pose(solved)
        position_error=float(np.linalg.norm(actual.position-target_position))
        relative=target_model_rotation@quaternion_xyzw_to_matrix(actual.quaternion_xyzw).T
        rotation_error=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
        return _IKSolution(solved,position_error,rotation_error)

def _ensure_deoxys_on_path() -> None:
    try: __import__("deoxys"); return
    except ModuleNotFoundError: pass
    root=Path(__file__).resolve().parents[2]/"dependency"/"deoxys_control"/"deoxys"
    if str(root) not in sys.path: sys.path.insert(0,str(root))

def _make_interface(path: str, *, has_gripper: bool) -> Any:
    _ensure_deoxys_on_path(); from deoxys.franka_interface import FrankaInterface
    return FrankaInterface(
        path,
        has_gripper=has_gripper,
        use_visualizer=False,
        automatic_gripper_reset=False,
    )

def _controller_config(kind: str) -> Any:
    _ensure_deoxys_on_path(); from deoxys.utils.config_utils import get_default_controller_config
    if kind == "JOINT_VELOCITY":
        config=get_default_controller_config("JOINT_POSITION")
        config.controller_type="JOINT_VELOCITY"
        config.is_delta=False
        return config
    return get_default_controller_config(kind)

def _trace_attribute(value: Any, name: str) -> Any:
    """Read optional backend diagnostics without changing command behavior."""
    try:
        return getattr(value, name, None)
    except Exception as exc:
        return exc

def _trace_last_gripper_state(interface: Any) -> Any:
    """Best-effort snapshot of Deoxys' most recent full gripper state."""
    try:
        states=getattr(interface,"_gripper_state_buffer",None)
        return None if not states else states[-1]
    except Exception as exc:
        return exc

def _send_no_control(interface: Any) -> None:
    trace_event("franka.backend.no_control.request")
    try:
        if hasattr(interface,"send_no_control"):
            interface.send_no_control()
        else:
            _ensure_deoxys_on_path(); from deoxys.proto.franka_interface import franka_controller_pb2 as pb
            message=pb.FrankaControlMessage(); message.controller_type=pb.FrankaControlMessage.ControllerType.NO_CONTROL
            message.control_msg.Pack(pb.FrankaDummyControllerMessage()); message.timeout=.2
            interface._publisher.send(message.SerializeToString())
    except Exception as exc:
        trace_event("franka.backend.no_control.failed", exception=exc)
        raise
    trace_event("franka.backend.no_control.sent")

class FrankaConnection:
    """Own a Deoxys interface shared by an arm and its gripper."""
    def __init__(self, interface_cfg: str, interface: Any | None = None, *, has_gripper=False):
        self.interface_cfg=interface_cfg; self.interface=interface; self.has_gripper=bool(has_gripper); self.closed=False
    def connect(self):
        trace_event("franka.connection.connect.request", interface_cfg=self.interface_cfg, has_gripper=self.has_gripper, interface_preconfigured=self.interface is not None)
        if self.closed: raise RobotConnectionError("connection is closed")
        if self.interface is None: self.interface=_make_interface(self.interface_cfg,has_gripper=self.has_gripper)
        trace_event("franka.connection.connect.completed", interface_type=type(self.interface).__name__)
    def close(self):
        if self.closed: return
        trace_event("franka.connection.close.request")
        self.closed=True
        if self.interface is not None:
            try: _send_no_control(self.interface)
            finally: self.interface.close()
        trace_event("franka.connection.close.completed")

class FrankaArm:
    _HOLD_POSITION_TOLERANCE_RAD = 1e-3
    _HOLD_VELOCITY_TOLERANCE_RAD_S = 1e-2
    _REFERENCE_POSITION_TOLERANCE_RAD = 1e-6
    _REFERENCE_VELOCITY_TOLERANCE_RAD_S = 1e-4
    _JOINT_REFERENCE_POSITION_GAIN_S = 4.0

    def __init__(self, connection: FrankaConnection, *, state_timeout=.2, connect_timeout=2.0, max_dt=.1,
                 max_velocity=1.0, max_acceleration=1.0, ik_position_tolerance=.01,
                 ik_rotation_tolerance=.1, max_ik_joint_delta=.25,
                 max_ik_translation_step=.01, max_ik_rotation_step=.05, damping=.05,
                 config_loader: Callable[[str],Any]=_controller_config,
                 ik_solver: Any | None=None):
        self.connection=connection; self.state_timeout=float(state_timeout); self.connect_timeout=float(connect_timeout); self.max_dt=float(max_dt)
        self.max_velocity=np.broadcast_to(max_velocity,(7,)).astype(float).copy()
        self.max_acceleration=np.broadcast_to(max_acceleration,(7,)).astype(float).copy()
        self.lower_limits=PANDA_LOWER.copy(); self.upper_limits=PANDA_UPPER.copy()
        self.ik_position_tolerance=float(ik_position_tolerance); self.ik_rotation_tolerance=float(ik_rotation_tolerance)
        self.max_ik_joint_delta=float(max_ik_joint_delta)
        self.max_ik_translation_step=float(max_ik_translation_step); self.max_ik_rotation_step=float(max_ik_rotation_step)
        self.damping=float(damping); self._config_loader=config_loader; self._configs={}; self._ik=ik_solver
        self._state_object=None; self._state_time=None; self._last_safe_position=None; self._last_safe_velocity=np.zeros(7); self._active_mode=None; self._hold_position=None; self._connected=False
        self._last_commanded_joint_positions=None; self._last_commanded_joint_velocities=None; self._last_ik_joint_positions=None; self._last_ik_joint_velocities=None
        self._joint_reference=None; self._motion_target_position=None
        self._motion_state=ArmMotionState.HOLDING
        self._is_settled=True
        for name,value in (("state_timeout",self.state_timeout),("connect_timeout",self.connect_timeout),("max_dt",self.max_dt),("max_ik_joint_delta",self.max_ik_joint_delta),("max_ik_translation_step",self.max_ik_translation_step),("max_ik_rotation_step",self.max_ik_rotation_step),("damping",self.damping)):
            if not np.isfinite(value) or value <= 0: raise ValueError(f"{name} must be finite and positive")
        self._last_command_dt=min(.05,self.max_dt)
    @property
    def dof(self): return 7
    @property
    def interface(self): return self.connection.interface
    @property
    def motion_state(self): return self._motion_state
    @property
    def is_settled(self): return self._is_settled
    @property
    def last_commanded_joint_positions(self):
        return None if self._last_commanded_joint_positions is None else self._last_commanded_joint_positions.copy()
    @property
    def last_ik_joint_positions(self):
        return None if self._last_ik_joint_positions is None else self._last_ik_joint_positions.copy()
    @property
    def last_commanded_joint_velocities(self):
        return None if self._last_commanded_joint_velocities is None else self._last_commanded_joint_velocities.copy()
    @property
    def last_ik_joint_velocities(self):
        return None if self._last_ik_joint_velocities is None else self._last_ik_joint_velocities.copy()
    def connect(self):
        trace_event("franka.arm.connect.request")
        self.connection.connect(); self._connected=True
        deadline=time.monotonic()+self.connect_timeout
        while getattr(self.interface,"last_state",None) is None:
            if time.monotonic() >= deadline:
                self._connected=False
                raise RobotStateUnavailable("timed out waiting for initial Franka state")
            time.sleep(.01)
        prepare_controller=getattr(self.interface,"prepare_controller",None)
        if not callable(prepare_controller):
            self._connected=False
            raise RobotConnectionError(
                "Deoxys interface does not support controller preparation"
            )
        trace_event(
            "franka.arm.controller.prepare.request",
            controller_type="JOINT_POSITION",
        )
        try:
            prepare_controller("JOINT_POSITION")
        except Exception as exc:
            self._connected=False
            trace_event(
                "franka.arm.controller.prepare.failed",
                controller_type="JOINT_POSITION",
                exception=exc,
            )
            raise RobotConnectionError(
                "failed to prepare initial JOINT_POSITION controller"
            ) from exc
        trace_event(
            "franka.arm.controller.prepare.completed",
            controller_type="JOINT_POSITION",
        )
        trace_event("franka.arm.connect.completed")
    def _require(self):
        if not self._connected or self.interface is None: raise RobotConnectionError("Franka arm is not connected")
    def get_state(self):
        self._require(); source=getattr(self.interface,"last_state",None)
        if source is None:
            trace_event("franka.backend.state.unavailable")
            raise RobotStateUnavailable("Franka state is unavailable")
        now=time.monotonic()
        source_is_new=source is not self._state_object
        if source_is_new: self._state_object=source; self._state_time=now
        trace_event(
            "franka.backend.state.read",
            raw_state=source,
            source_is_new=source_is_new,
            state_received_monotonic_s=self._state_time,
            read_monotonic_s=now,
            state_age_s=None if self._state_time is None else now-self._state_time,
        )
        if self._state_time is None or now-self._state_time > self.state_timeout:
            trace_event("franka.backend.state.stale", state_received_monotonic_s=self._state_time, read_monotonic_s=now, state_timeout_s=self.state_timeout)
            raise RobotStateStale("Franka state is stale")
        q=np.asarray(getattr(self.interface,"last_q",None),dtype=float)
        dq=np.asarray(getattr(self.interface,"last_dq",np.zeros(7)),dtype=float)
        matrix=np.asarray(getattr(self.interface,"last_eef_pose",None),dtype=float)
        if q.shape!=(7,) or dq.shape!=(7,) or matrix.shape!=(4,4) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)) or not np.all(np.isfinite(matrix)):
            trace_event("franka.backend.state.invalid", joint_positions=q, joint_velocities=dq, end_effector_matrix=matrix)
            raise RobotStateUnavailable("Franka state contains invalid values")
        rot=matrix[:3,:3]
        if not np.allclose(rot.T@rot,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(rot),1,atol=1e-3):
            trace_event("franka.backend.state.invalid_rotation", rotation_matrix=rot)
            raise RobotStateUnavailable("Franka end-effector rotation is invalid")
        state=ArmState(q,dq,Pose(matrix[:3,3],rotation_matrix_to_quaternion_xyzw(rot)),self._state_time)
        if self._active_mode == "JOINT_POSITION" and self._joint_reference is not None and self._motion_target_position is not None:
            was_settled=self._is_settled
            self._is_settled=self._position_reference_is_settled(state)
            if self._is_settled and self._motion_state is ArmMotionState.STREAMING:
                self._motion_state=ArmMotionState.HOLDING
            if self._is_settled != was_settled:
                trace_event(
                    "franka.arm.settled.changed",
                    is_settled=self._is_settled,
                    motion_state=self._motion_state.value,
                )
        trace_event(
            "franka.arm.state.result",
            state=state,
            backend_desired_joint_positions=_trace_attribute(self.interface,"last_q_d"),
        )
        return state
    def _position_reference_is_settled(self,state,*,positions=None,velocities=None,target=None):
        positions=(self._joint_reference.positions if positions is None and self._joint_reference is not None else positions)
        velocities=(self._joint_reference.velocities if velocities is None and self._joint_reference is not None else velocities)
        target=self._motion_target_position if target is None else target
        if positions is None or velocities is None or target is None:
            return False
        backend_state=getattr(self.interface,"last_state",None)
        desired_velocity=np.asarray(getattr(backend_state,"dq_d",np.full(7,np.inf)),dtype=float)
        reference_settled=(np.max(np.abs(positions-target)) <= self._REFERENCE_POSITION_TOLERANCE_RAD and np.max(np.abs(velocities)) <= self._REFERENCE_VELOCITY_TOLERANCE_RAD_S)
        position_settled=np.max(np.abs(state.joint_positions-target)) <= self._HOLD_POSITION_TOLERANCE_RAD
        measured_settled=np.max(np.abs(state.joint_velocities)) <= self._HOLD_VELOCITY_TOLERANCE_RAD_S
        desired_settled=desired_velocity.shape==(7,) and np.all(np.isfinite(desired_velocity)) and np.max(np.abs(desired_velocity)) <= self._HOLD_VELOCITY_TOLERANCE_RAD_S
        return bool(reference_settled and position_settled and measured_settled and desired_settled)
    def _dt(self,dt):
        dt=float(dt)
        if not np.isfinite(dt) or dt<=0: raise ValueError("dt must be finite and positive")
        self._last_command_dt=min(dt,self.max_dt)
        return self._last_command_dt
    def _vector(self,value,name):
        value=np.asarray(value,dtype=float)
        if value.shape!=(7,) or not np.all(np.isfinite(value)): raise ValueError(f"{name} must be a finite shape-(7,) vector")
        return value
    def _shape_velocity(self,desired,dt):
        desired=np.clip(desired,-self.max_velocity,self.max_velocity)
        return self._last_safe_velocity+np.clip(desired-self._last_safe_velocity,-self.max_acceleration*dt,self.max_acceleration*dt)
    def _send(self,kind,values,*,velocity_reference=None):
        try:
            if kind not in self._configs:
                self._configs[kind]=self._config_loader(kind)
                interpolation=getattr(self._configs[kind],"traj_interpolator_cfg",None)
                if interpolation is None and isinstance(self._configs[kind],dict): interpolation=self._configs[kind].get("traj_interpolator_cfg")
                if kind == "JOINT_POSITION" and interpolation is not None:
                    interpolation["traj_interpolator_type"]="LINEAR_JOINT_POSITION"
                    interpolation["time_fraction"]=1.0
            cfg=self._configs[kind]
            action=np.asarray(values).tolist()+[0.0]
            backend_kwargs={}
            if kind == "JOINT_POSITION":
                if velocity_reference is None:
                    velocity_reference=np.zeros(7)
                velocity_reference=self._vector(velocity_reference,"velocity_reference")
                backend_kwargs["joint_velocity_reference"]=velocity_reference.tolist()
            trace_event(
                "franka.backend.command.request",
                controller_type=kind,
                joint_values=values,
                action=action,
                controller_config=cfg,
                termination=False,
                control_gripper=False,
                joint_velocity_reference=velocity_reference,
            )
            self.interface.control(
                controller_type=kind,
                action=action,
                controller_cfg=cfg,
                termination=False,
                control_gripper=False,
                **backend_kwargs,
            )
            if kind in ("JOINT_POSITION","JOINT_IMPEDANCE"):
                self._last_commanded_joint_positions=np.asarray(values,dtype=float).copy()
            if kind == "JOINT_POSITION":
                self._last_commanded_joint_velocities=np.asarray(velocity_reference,dtype=float).copy()
            trace_event(
                "franka.backend.command.sent",
                controller_type=kind,
                joint_values=values,
                action=action,
                joint_velocity_reference=velocity_reference,
            )
        except Exception as exc:
            trace_event("franka.backend.command.failed", controller_type=kind, joint_values=values, exception=exc)
            raise RobotCommandError(f"Deoxys {kind} command failed") from exc
    def _arm_mode(self,kind,state):
        """Start a backend mode with a command continuous with measured state."""
        if self._active_mode == kind:
            trace_event("franka.arm.mode.unchanged", mode=kind)
            return False
        previous_mode=self._active_mode
        if self._active_mode is not None:
            _send_no_control(self.interface)
        if kind == "JOINT_VELOCITY": initial=state.joint_velocities
        else: initial=state.joint_positions
        trace_event("franka.arm.mode.transition", previous_mode=previous_mode, requested_mode=kind, arming_values=initial)
        self._send(
            kind,
            initial,
            velocity_reference=np.zeros(7) if kind == "JOINT_POSITION" else None,
        )
        self._last_safe_position=None if kind == "JOINT_VELOCITY" else np.asarray(initial,dtype=float).copy()
        self._last_safe_velocity=(np.zeros(7) if kind == "JOINT_POSITION" else np.clip(state.joint_velocities,-self.max_velocity,self.max_velocity).copy())
        self._active_mode=kind
        if kind == "JOINT_POSITION":
            initial_reference=np.asarray(initial,dtype=float).copy()
            self._joint_reference=_JointReferenceState(initial_reference,np.zeros(7))
            self._motion_target_position=initial_reference.copy()
            self._motion_state=ArmMotionState.HOLDING
            self._is_settled=np.max(np.abs(state.joint_velocities)) <= self._HOLD_VELOCITY_TOLERANCE_RAD_S
        else:
            self._joint_reference=None; self._motion_target_position=None
            self._motion_state=ArmMotionState.STREAMING
            self._is_settled=False
        trace_event("franka.arm.mode.armed", mode=kind, safe_position=self._last_safe_position, safe_velocity=self._last_safe_velocity)
        return True
    def _joint_position_reference(self,target,velocity_reference,*,step,state=None,streaming=False,holding=False):
        target=np.clip(self._vector(target,"positions"),self.lower_limits,self.upper_limits)
        feedforward=np.clip(self._vector(velocity_reference,"velocity_reference"),-self.max_velocity,self.max_velocity)
        state=self.get_state() if state is None else state
        if self._arm_mode("JOINT_POSITION",state):
            self._motion_target_position=None if streaming else target.copy()
            self._motion_state=ArmMotionState.HOLDING if holding else ArmMotionState.STREAMING
            self._is_settled=False
            trace_event("franka.arm.joint_position.armed_only", clipped_target=target, velocity_reference=feedforward, holding=holding, effective_dt=step)
            return False
        if self._joint_reference is None:
            raise RobotCommandError("joint-position reference state is unavailable")
        previous_position=self._joint_reference.positions
        previous_velocity=self._joint_reference.velocities
        position_error=target-previous_position

        if streaming:
            position_feedback=self._JOINT_REFERENCE_POSITION_GAIN_S*position_error
            desired_velocity=feedforward+position_feedback
            self._motion_target_position=None
            self._motion_state=ArmMotionState.STREAMING
            self._is_settled=False
        else:
            position_feedback=np.sign(position_error)*np.minimum(
                self._JOINT_REFERENCE_POSITION_GAIN_S*np.abs(position_error),
                np.sqrt(2.0*self.max_acceleration*np.abs(position_error)),
            )
            desired_velocity=position_feedback
            self._motion_target_position=target.copy()
            self._motion_state=ArmMotionState.HOLDING if holding else ArmMotionState.STREAMING

        desired_velocity=np.clip(desired_velocity,-self.max_velocity,self.max_velocity)
        next_velocity=previous_velocity+np.clip(
            desired_velocity-previous_velocity,
            -self.max_acceleration*step,
            self.max_acceleration*step,
        )
        next_velocity=np.clip(next_velocity,-self.max_velocity,self.max_velocity)

        if not streaming:
            # If two samples can land exactly at (target, 0), start that
            # terminal sequence now.  This avoids the old final-position snap,
            # which made q_ref and dq_ref describe two different trajectories.
            terminal_velocity=position_error/step-0.5*previous_velocity
            terminal_feasible=(
                (np.abs(terminal_velocity-previous_velocity) <= self.max_acceleration*step)
                & (np.abs(terminal_velocity) <= self.max_acceleration*step)
                & (np.abs(terminal_velocity) <= self.max_velocity)
            )
            next_velocity=np.where(terminal_feasible,terminal_velocity,next_velocity)

        lower_velocity=2.0*(self.lower_limits-previous_position)/step-previous_velocity
        upper_velocity=2.0*(self.upper_limits-previous_position)/step-previous_velocity
        next_velocity=np.maximum(lower_velocity,np.minimum(upper_velocity,next_velocity))
        next_position=previous_position+0.5*(previous_velocity+next_velocity)*step
        next_position=np.clip(next_position,self.lower_limits,self.upper_limits)
        next_velocity=2.0*(next_position-previous_position)/step-previous_velocity

        if not streaming:
            self._is_settled=self._position_reference_is_settled(
                state,
                positions=next_position,
                velocities=next_velocity,
                target=target,
            )
            if self._is_settled and not holding:
                self._motion_state=ArmMotionState.HOLDING

        trace_event("franka.arm.joint_position.shaped", clipped_target=target, base_position=previous_position, safe_position=next_position, safe_velocity=next_velocity, feedforward_velocity_reference=feedforward, position_feedback_velocity=position_feedback, desired_velocity_reference=desired_velocity, safe_velocity_reference=next_velocity, motion_state=self._motion_state.value, is_settled=self._is_settled, holding=holding, effective_dt=step)
        self._send("JOINT_POSITION",next_position,velocity_reference=next_velocity)
        self._joint_reference=_JointReferenceState(next_position.copy(),next_velocity.copy())
        self._last_safe_position=next_position.copy(); self._last_safe_velocity=next_velocity.copy()
        return True
    def home(self,*,dt):
        trace_event("franka.arm.home.request", action="HOME", joint_positions=FRANKA_HOME_JOINTS, dt=dt)
        self._hold_position=None
        step=self._dt(dt)
        state=self.get_state()
        self._joint_position_reference(
            FRANKA_HOME_JOINTS,
            np.zeros(7),
            step=step,
            state=state,
            streaming=False,
        )
        measured_at_home=(
            np.max(np.abs(state.joint_positions-FRANKA_HOME_JOINTS))
            <= self._HOLD_POSITION_TOLERANCE_RAD
            and np.max(np.abs(state.joint_velocities))
            <= self._HOLD_VELOCITY_TOLERANCE_RAD_S
        )
        if not measured_at_home:
            self._motion_target_position=FRANKA_HOME_JOINTS.copy()
            self._motion_state=ArmMotionState.STREAMING
            self._is_settled=False
        elif self._joint_reference is not None:
            self._motion_state=ArmMotionState.HOLDING
            self._is_settled=True
        trace_event(
            "franka.arm.home.completed",
            action="HOME",
            target_joint_positions=FRANKA_HOME_JOINTS,
            measured_joint_positions=state.joint_positions,
            motion_state=self._motion_state.value,
        )
    def joint_position(self,positions,*,dt):
        trace_event("franka.arm.joint_position.request", positions=positions, dt=dt)
        self._hold_position=None
        step=self._dt(dt); target=self._vector(positions,"positions")
        self._joint_position_reference(target,np.zeros(7),step=step,streaming=False)
    def joint_velocity(self,velocities,*,dt):
        trace_event("franka.arm.joint_velocity.request", velocities=velocities, dt=dt)
        self._hold_position=None
        step=self._dt(dt); desired=self._vector(velocities,"velocities"); state=self.get_state(); velocity=self._shape_velocity(desired,step)
        if self._arm_mode("JOINT_VELOCITY",state):
            trace_event("franka.arm.joint_velocity.armed_only", requested_velocities=velocities, effective_dt=step)
            return
        predicted=state.joint_positions+velocity*step
        velocity=np.where((predicted<self.lower_limits)|(predicted>self.upper_limits),0.0,velocity)
        trace_event("franka.arm.joint_velocity.shaped", requested_velocities=velocities, desired_velocity=desired, safe_velocity=velocity, predicted_joint_positions=predicted, effective_dt=step)
        self._send("JOINT_VELOCITY",velocity); self._last_safe_velocity=velocity
        self._is_settled=(np.max(np.abs(velocity)) <= self._HOLD_VELOCITY_TOLERANCE_RAD_S and np.max(np.abs(state.joint_velocities)) <= self._HOLD_VELOCITY_TOLERANCE_RAD_S)
        self._motion_state=ArmMotionState.HOLDING if self._is_settled else ArmMotionState.STREAMING
    def joint_impedance(self,positions,*,dt):
        trace_event("franka.arm.joint_impedance.request", positions=positions, dt=dt)
        self._hold_position=None
        step=self._dt(dt); target=np.clip(self._vector(positions,"positions"),self.lower_limits,self.upper_limits); state=self.get_state()
        if self._arm_mode("JOINT_IMPEDANCE",state):
            trace_event("franka.arm.joint_impedance.armed_only", requested_positions=positions, clipped_target=target, effective_dt=step)
            return
        base=self._last_safe_position if self._last_safe_position is not None else state.joint_positions
        velocity=self._shape_velocity((target-base)/step,step); safe=base+velocity*step
        overshot=np.sign(target-base)!=np.sign(target-safe)
        safe[overshot]=target[overshot]; velocity[overshot]=0.0
        safe=np.clip(safe,self.lower_limits,self.upper_limits); velocity[(safe<=self.lower_limits)|(safe>=self.upper_limits)]=0.0
        trace_event("franka.arm.joint_impedance.shaped", requested_positions=positions, clipped_target=target, base_position=base, safe_position=safe, safe_velocity=velocity, effective_dt=step)
        self._send("JOINT_IMPEDANCE",safe); self._last_safe_position=safe.copy(); self._last_safe_velocity=velocity
        self._is_settled=(np.max(np.abs(safe-target)) <= self._HOLD_POSITION_TOLERANCE_RAD and np.max(np.abs(state.joint_velocities)) <= self._HOLD_VELOCITY_TOLERANCE_RAD_S)
        self._motion_state=ArmMotionState.HOLDING if self._is_settled else ArmMotionState.STREAMING
    def _solver(self):
        if self._ik is None: self._ik=_PinocchioKinematics()
        return self._ik
    def _solve_ik(self,target_pose,seed_joint_positions):
        solution=self._solver().solve(seed_joint_positions,self.get_state().end_effector_pose,target_pose)
        trace_event("franka.arm.pose_ik.result", target_pose=target_pose, seed_joint_positions=seed_joint_positions, joint_positions=solution.joint_positions, position_residual_m=solution.position_residual_m, rotation_residual_rad=solution.rotation_residual_rad)
        if solution.position_residual_m>self.ik_position_tolerance or solution.rotation_residual_rad>self.ik_rotation_tolerance:
            raise IKUnreachableError(f"IK residual too large: {solution.position_residual_m:.4f} m, {solution.rotation_residual_rad:.4f} rad")
        if np.max(np.abs(solution.joint_positions-seed_joint_positions)) > self.max_ik_joint_delta:
            raise IKUnreachableError("IK solution is discontinuous with the measured joint state")
        return solution.joint_positions.copy()
    def _solve_differential_ik(self,twist,joint_positions):
        jacobian=self._solver().jacobian(joint_positions); desired=np.r_[twist.linear,twist.angular]
        regularized=jacobian@jacobian.T+(self.damping**2)*np.eye(6)
        try:
            result=jacobian.T@np.linalg.solve(regularized,desired)
            trace_event("franka.arm.differential_ik.result", twist=twist, joint_positions=joint_positions, jacobian=jacobian, damping=self.damping, joint_velocities=result)
            return result
        except np.linalg.LinAlgError as exc: raise IKError("differential IK failed") from exc
    def ik_joint_position(self,pose,*,twist_ref=None,delta=False,dt):
        trace_event("franka.arm.ik_joint_position.request", pose=pose, twist_ref=twist_ref, delta=delta, dt=dt)
        if not isinstance(pose,Pose): raise TypeError("pose must be a Pose")
        if twist_ref is not None and not isinstance(twist_ref,Twist): raise TypeError("twist_ref must be a Twist or None")
        step=self._dt(dt); state=self.get_state(); target=pose
        if delta:
            current=state.end_effector_pose
            target=Pose(current.position+pose.position,rotation_matrix_to_quaternion_xyzw(quaternion_xyzw_to_matrix(pose.quaternion_xyzw)@quaternion_xyzw_to_matrix(current.quaternion_xyzw)))
        absolute_target=target
        current=state.end_effector_pose
        translation=target.position-current.position
        distance=float(np.linalg.norm(translation))
        if distance>self.max_ik_translation_step: translation*=self.max_ik_translation_step/distance
        current_rotation=quaternion_xyzw_to_matrix(current.quaternion_xyzw)
        target_rotation=quaternion_xyzw_to_matrix(target.quaternion_xyzw)
        rotation_step=np.asarray(pin.log3(target_rotation@current_rotation.T))
        angle=float(np.linalg.norm(rotation_step))
        if angle>self.max_ik_rotation_step: rotation_step*=self.max_ik_rotation_step/angle
        target=Pose(current.position+translation,rotation_matrix_to_quaternion_xyzw(pin.exp3(rotation_step)@current_rotation))
        trace_event("franka.arm.ik_joint_position.subtarget", requested_pose=pose, delta=delta, absolute_target_pose=absolute_target, bounded_target_pose=target, translation_step=translation, rotation_step=rotation_step, measured_state=state)
        solution=self._solve_ik(target,state.joint_positions)
        velocity_reference=(np.zeros(7) if twist_ref is None else self._solve_differential_ik(twist_ref,solution))
        self._last_ik_joint_positions=solution.copy()
        self._last_ik_joint_velocities=velocity_reference.copy()
        streaming=twist_ref is not None and (np.linalg.norm(twist_ref.linear)>1e-9 or np.linalg.norm(twist_ref.angular)>1e-9)
        trace_event("franka.arm.ik_joint_position.solved", joint_positions=solution, joint_velocities=velocity_reference, streaming=streaming)
        self._hold_position=None
        self._joint_position_reference(solution,velocity_reference,step=step,streaming=streaming)
    def ik_joint_velocity(self,twist,*,dt):
        trace_event("franka.arm.ik_joint_velocity.request", twist=twist, dt=dt)
        if not isinstance(twist,Twist): raise TypeError("twist must be a Twist")
        self._dt(dt); state=self.get_state(); self.joint_velocity(self._solve_differential_ik(twist,state.joint_positions),dt=dt)
    def stop(self):
        trace_event("franka.arm.stop.request", active_mode=self._active_mode, hold_position=self._hold_position)
        self._require()
        try:
            state=self.get_state()
            if self._active_mode not in (None,"JOINT_POSITION"):
                _send_no_control(self.interface)
                self._active_mode=None
            if self._hold_position is None:
                desired=getattr(self.interface,"last_q_d",None)
                desired=np.asarray(desired,dtype=float) if desired is not None else state.joint_positions
                self._hold_position=(desired if desired.shape==(7,) and np.all(np.isfinite(desired)) else state.joint_positions).copy()
            trace_event("franka.arm.stop.hold", measured_state=state, hold_position=self._hold_position)
            self._joint_position_reference(self._hold_position,np.zeros(7),step=self._last_command_dt,state=state,streaming=False,holding=True)
            self._motion_state=ArmMotionState.HOLDING
            trace_event("franka.arm.stop.completed", active_mode=self._active_mode, hold_position=self._hold_position, motion_state=self._motion_state.value, is_settled=self._is_settled)
        except Exception as exc: raise RobotCommandError(f"failed to stop Franka arm: {exc}") from exc

class FrankaGripper:
    def __init__(self,connection: FrankaConnection): self.connection=connection; self._connected=False; self._position=1.0; self._command_state=None
    def connect(self):
        trace_event("franka.gripper.connect.request")
        self.connection.connect(); self._connected=True
        trace_event("franka.gripper.connect.completed")
    def position(self,normalized):
        trace_event("franka.gripper.position.request", normalized_position=normalized, previous_command_state=self._command_state, backend_state=_trace_last_gripper_state(self.connection.interface))
        value=float(normalized)
        if not np.isfinite(value) or not 0<=value<=1: raise ValueError("normalized must be in [0, 1]")
        if not self._connected or self.connection.interface is None: raise RobotConnectionError("Franka gripper is not connected")
        state=self._command_state
        if value <= .4: state="closed"
        elif value >= .6: state="open"
        if state == self._command_state:
            self._position=value
            trace_event("franka.gripper.position.noop", reason="discrete_state_unchanged", command_state=state, normalized_position=value)
            return
        if state is None:
            self._position=value
            trace_event("franka.gripper.position.noop", reason="inside_hysteresis_band", command_state=self._command_state, normalized_position=value)
            return
        backend_action=1.0 if state=="closed" else -1.0
        trace_event("franka.backend.gripper_command.request", command_state=state, normalized_position=value, action=backend_action)
        try: self.connection.interface.gripper_control(backend_action)
        except Exception as exc:
            trace_event("franka.backend.gripper_command.failed", command_state=state, action=backend_action, exception=exc)
            raise RobotCommandError("Deoxys gripper command failed") from exc
        self._command_state=state; self._position=value
        trace_event("franka.backend.gripper_command.sent", command_state=state, normalized_position=value, action=backend_action, backend_width=_trace_attribute(self.connection.interface,"last_gripper_q"), backend_state=_trace_last_gripper_state(self.connection.interface))
    def get_state(self):
        if not self._connected: raise RobotConnectionError("Franka gripper is not connected")
        state=EndEffectorState(time.monotonic(),position=self._position)
        trace_event("franka.gripper.state.result", state=state, backend_width=_trace_attribute(self.connection.interface,"last_gripper_q"), backend_state=_trace_last_gripper_state(self.connection.interface))
        return state
    def stop(self):
        trace_event("franka.gripper.stop")

def make_franka_robot(interface_cfg: str, *, with_gripper=True, interface=None, **arm_options) -> Robot:
    connection=FrankaConnection(interface_cfg,interface,has_gripper=with_gripper)
    return Robot(FrankaArm(connection,**arm_options),FrankaGripper(connection) if with_gripper else None,connection)
