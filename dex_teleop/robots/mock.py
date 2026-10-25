from __future__ import annotations
import time
import numpy as np
from dex_teleop.core.types import ArmState, EndEffectorState, Pose, Twist
from dex_teleop.diagnostics import trace_event
from dex_teleop.robots.arm import ArmMotionState
from dex_teleop.robots.base import IKUnreachableError, RobotConnectionError
from dex_teleop.utils.transforms import quaternion_xyzw_to_matrix, rotation_matrix_to_quaternion_xyzw, rotation_vector_to_matrix


def _dt(value: float, maximum: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0: raise ValueError("dt must be finite and positive")
    return min(value, maximum)


class MockArm:
    def __init__(self, dof: int = 7, *, max_velocity=2.0, max_acceleration=10.0, max_dt=0.1, home_positions=None):
        self._dof = int(dof); self.max_dt = float(max_dt)
        self.max_velocity = np.broadcast_to(max_velocity, (self._dof,)).astype(float).copy()
        self.max_acceleration = np.broadcast_to(max_acceleration, (self._dof,)).astype(float).copy()
        if home_positions is None:
            home_positions = np.zeros(self._dof)
        self._home_positions = self._vector(home_positions, "home_positions").copy()
        self.connected = False; self.stopped = False
        self._q = np.zeros(self._dof); self._dq = np.zeros(self._dof)
        self._motion_state = ArmMotionState.HOLDING
        self._is_settled = True
        self.commands: list[tuple[str, np.ndarray]] = []

    @property
    def dof(self): return self._dof
    @property
    def motion_state(self): return self._motion_state
    @property
    def is_settled(self): return self._is_settled
    def connect(self):
        self.connected = True; self.stopped = False
        trace_event("mock.arm.connected", dof=self.dof)
    def _require(self):
        if not self.connected: raise RobotConnectionError("arm is not connected")
    def get_state(self):
        self._require()
        p = np.zeros(3); p[:min(3,self.dof)] = self._q[:min(3,self.dof)]
        state=ArmState(self._q.copy(), self._dq.copy(), Pose(p, [0,0,0,1]), time.monotonic())
        trace_event("mock.arm.state.result", state=state)
        return state
    def _vector(self, value, name):
        a=np.asarray(value,dtype=float)
        if a.shape!=(self.dof,) or not np.all(np.isfinite(a)): raise ValueError(f"{name} must be a finite shape-({self.dof},) vector")
        return a
    def _limit_velocity(self, desired, dt):
        desired=np.clip(desired,-self.max_velocity,self.max_velocity)
        return self._dq + np.clip(desired-self._dq,-self.max_acceleration*dt,self.max_acceleration*dt)
    def home(self, *, dt):
        trace_event("mock.arm.home.request", action="HOME", joint_positions=self._home_positions, dt=dt)
        self.joint_position(self._home_positions,dt=dt)
        self.commands[-1] = ("HOME",self.commands[-1][1])
        trace_event("mock.arm.home.completed", action="HOME", joint_positions=self._q, joint_velocities=self._dq, motion_state=self._motion_state.value)
    def joint_position(self, positions, *, dt):
        trace_event("mock.arm.joint_position.request", positions=positions, dt=dt)
        self._require(); step=_dt(dt,self.max_dt); target=self._vector(positions,"positions")
        velocity=self._limit_velocity((target-self._q)/step,step); self._q += velocity*step; self._dq=velocity
        self.commands.append(("joint_position",self._q.copy()))
        self._is_settled=np.max(np.abs(target-self._q))<=1e-9 and np.max(np.abs(self._dq))<=1e-9
        self._motion_state=ArmMotionState.HOLDING if self._is_settled else ArmMotionState.STREAMING
        trace_event("mock.arm.command.executed", command="joint_position", requested=target, effective_dt=step, joint_positions=self._q, joint_velocities=self._dq)
    def joint_velocity(self, velocities, *, dt):
        trace_event("mock.arm.joint_velocity.request", velocities=velocities, dt=dt)
        self._require(); step=_dt(dt,self.max_dt); self._dq=self._limit_velocity(self._vector(velocities,"velocities"),step); self._q += self._dq*step
        self.commands.append(("joint_velocity",self._dq.copy()))
        self._is_settled=np.max(np.abs(self._dq))<=1e-9
        self._motion_state=ArmMotionState.HOLDING if self._is_settled else ArmMotionState.STREAMING
        trace_event("mock.arm.command.executed", command="joint_velocity", requested=velocities, effective_dt=step, joint_positions=self._q, joint_velocities=self._dq)
    def joint_impedance(self, positions, *, dt):
        trace_event("mock.arm.joint_impedance.request", positions=positions, dt=dt)
        self._require(); step=_dt(dt,self.max_dt); target=self._vector(positions,"positions")
        velocity=self._limit_velocity((target-self._q)/step,step); self._q += velocity*step; self._dq=velocity
        self.commands.append(("joint_impedance",self._q.copy()))
        self._is_settled=np.max(np.abs(target-self._q))<=1e-9 and np.max(np.abs(self._dq))<=1e-9
        self._motion_state=ArmMotionState.HOLDING if self._is_settled else ArmMotionState.STREAMING
        trace_event("mock.arm.command.executed", command="joint_impedance", requested=target, effective_dt=step, joint_positions=self._q, joint_velocities=self._dq)
    def _solve_ik(self, target_pose, seed_joint_positions):
        if self.dof < 3: raise IKUnreachableError("mock arm needs at least three joints for pose IK")
        q=np.asarray(seed_joint_positions).copy(); q[:3]=target_pose.position; return q
    def _solve_differential_ik(self, twist, joint_positions):
        result=np.zeros(self.dof); result[:min(3,self.dof)]=twist.linear[:min(3,self.dof)]; return result
    def ik_joint_position(self, pose, *, twist_ref=None, delta=False, dt):
        trace_event("mock.arm.ik_joint_position.request", pose=pose, twist_ref=twist_ref, delta=delta, dt=dt)
        if not isinstance(pose,Pose): raise TypeError("pose must be a Pose")
        if twist_ref is not None and not isinstance(twist_ref,Twist): raise TypeError("twist_ref must be a Twist or None")
        target=pose
        if delta:
            current=self.get_state().end_effector_pose
            rot=rotation_vector_to_matrix(np.zeros(3)) @ quaternion_xyzw_to_matrix(current.quaternion_xyzw)
            target=Pose(current.position+pose.position, rotation_matrix_to_quaternion_xyzw(quaternion_xyzw_to_matrix(pose.quaternion_xyzw) @ rot))
        solution=self._solve_ik(target,self._q)
        trace_event("mock.arm.ik_joint_position.solved", target_pose=target, joint_positions=solution)
        self.joint_position(solution,dt=dt)
        if twist_ref is not None and (np.linalg.norm(twist_ref.linear)>1e-9 or np.linalg.norm(twist_ref.angular)>1e-9):
            self._motion_state=ArmMotionState.STREAMING
            self._is_settled=False
    def ik_joint_velocity(self, twist, *, dt):
        trace_event("mock.arm.ik_joint_velocity.request", twist=twist, dt=dt)
        if not isinstance(twist,Twist): raise TypeError("twist must be a Twist")
        solution=self._solve_differential_ik(twist,self._q)
        trace_event("mock.arm.ik_joint_velocity.solved", joint_velocities=solution)
        self.joint_velocity(solution,dt=dt)
    def stop(self):
        self._dq[:]=0; self.stopped=True
        self._motion_state=ArmMotionState.HOLDING
        self._is_settled=True
        trace_event("mock.arm.stopped", joint_positions=self._q, joint_velocities=self._dq)


class MockEndEffector:
    def __init__(self): self.connected=False; self.stopped=False; self.normalized=1.0
    def connect(self):
        self.connected=True
        trace_event("mock.end_effector.connected")
    def get_state(self):
        state=EndEffectorState(time.monotonic(), position=self.normalized)
        trace_event("mock.end_effector.state.result", state=state)
        return state
    def stop(self):
        self.stopped=True
        trace_event("mock.end_effector.stopped", normalized_position=self.normalized)

class MockGripper(MockEndEffector):
    def position(self, normalized):
        trace_event("mock.gripper.position.request", normalized_position=normalized)
        value=float(normalized)
        if not np.isfinite(value) or not 0 <= value <= 1: raise ValueError("normalized must be in [0, 1]")
        self.normalized=value
        trace_event("mock.gripper.position.executed", normalized_position=self.normalized)
