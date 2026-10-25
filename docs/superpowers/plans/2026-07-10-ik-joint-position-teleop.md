# IK Joint-Position Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-selectable, dual-arm-capable 6DoF Oculus teleoperation strategy that solves latest-target MuJoCo IK asynchronously and sends safely shaped Deoxys `JOINT_POSITION` commands, with the first hardware configuration limited to the right arm.

**Architecture:** The 20 Hz application loop constructs clutch-relative target poses and remains the sole robot-command owner. One latest-target worker per configured arm solves relative-pose IK off the main loop; the policy validates fresh solutions, applies per-joint velocity and acceleration shaping, and relies on Deoxys for 500 Hz interpolation. Per-arm epochs, request IDs, and an explicit hold/fault state machine prevent stale or discontinuous results from executing.

**Tech Stack:** Python 3.10, NumPy, MuJoCo, Deoxys, PyYAML, pytest, JSONL telemetry.

## Global Constraints

- Use the existing environment exactly: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dex-teleop && export PYTHONNOUSERSITE=1`.
- Preserve all pre-existing working-tree changes. Before every commit, inspect `git diff --cached`; stage only feature hunks and new files. If a feature hunk cannot be separated from an existing user hunk, leave that file unstaged and report the deferred commit instead of bundling unrelated work.
- Baseline verification on 2026-07-10 is `179 passed in 1.02s` from `pytest -q`.
- Do not modify the Deoxys C++ backend or the existing Cartesian-velocity policy behavior.
- Missing `app.strategy` must continue to select `cartesian_velocity`.
- The application policy loop remains 20 Hz; Deoxys `JOINT_POSITION` interpolation remains 500 Hz.
- Initial IK safety values are exactly: 0.10 m anchor translation, 0.50 rad anchor rotation, 0.15 s result age, 0.005 m position residual, 0.05 rad rotation residual, 0.20 rad target jump, 0.50 rad/s joint velocity, 1.00 rad/s2 joint acceleration, 0.10 s shaping `dt` cap, and 0.05 rad joint-limit margin.
- Grip release must publish a measured joint-position hold on the next cycle. Fault recovery requires release and a new press.
- The implementation supports every arm in `cfg["arms"]`; the first real-robot config contains and connects only `right` / `franka-three`.
- Every cycle must retain current JSONL fields and add complete IK, state-machine, and joint-command diagnostics.
- Use test-driven development: observe each new test fail for the intended missing behavior before adding production code.

## File Responsibility Map

- `dex_teleop/utils/transforms.py`: quaternion, rotation-vector, rotation-matrix, and rotation-error math shared by target construction and IK.
- `dex_teleop/policies/ik_target.py`: immutable clutch anchor snapshots and pure 6DoF target construction/clamping.
- `dex_teleop/policies/joint_setpoint.py`: per-joint velocity/acceleration-limited command shaping.
- `dex_teleop/core/types.py`: `ArmState` joint data, `JointPositionCommand`, and command-batch grouping.
- `dex_teleop/robots/arm.py`: abstract joint-position send interface.
- `dex_teleop/robots/franka.py`: Deoxys state extraction and `JOINT_POSITION` action translation.
- `dex_teleop/robots/mock.py`: deterministic joint state and command recording for dry runs/tests.
- `dex_teleop/retargeting/mujoco_pose_ik.py`: relative-pose MuJoCo solve and FK residual reporting.
- `dex_teleop/retargeting/latest_ik_worker.py`: latest-only request mailbox and asynchronous solver lifecycle.
- `dex_teleop/policies/bimanual_ik_joint_position.py`: per-arm clutch/state-machine/result-validation policy.
- `dex_teleop/policies/factory.py`: strategy selection without application-level type branching.
- `dex_teleop/core/config.py`: conditional strategy schema and numeric validation.
- `dex_teleop/apps/oculus_franka.py`: lifecycle, command dispatch, printing, and JSONL integration.
- `configs/oculus_franka_right_ik_joint_position.yml`: conservative right-only first-run config.
- Matching files under `tests/`: isolated unit/integration coverage for every responsibility above.

---

### Task 1: 6DoF Transform Math and Clutch Target Construction

**Files:**
- Modify: `dex_teleop/utils/transforms.py:79-132`
- Create: `dex_teleop/policies/ik_target.py`
- Modify: `tests/utils/test_transforms.py:8-142`
- Create: `tests/policies/test_ik_target.py`

**Interfaces:**
- Consumes: existing `Pose`, `axis_mapping`, and `clamp_vector_norm`.
- Produces: `quaternion_xyzw_to_matrix(q) -> np.ndarray`, `rotation_matrix_to_quaternion_xyzw(rotation) -> np.ndarray`, `quaternion_delta_rotation_vector(anchor_xyzw, current_xyzw) -> np.ndarray`, `rotation_vector_to_matrix(vector) -> np.ndarray`, `rotation_error_angle(target, actual) -> float`, `ClutchAnchor.snapshot(controller_pose, robot_pose)`, and `build_clutch_target(...) -> TargetPoseResult`.

- [ ] **Step 1: Add failing transform tests**

Add imports and tests that cover shortest-branch quaternion deltas, Rodrigues conversion, matrix/quaternion round trips, and rotation error:

```python
from dex_teleop.utils.transforms import (
    quaternion_delta_rotation_vector,
    quaternion_xyzw_to_matrix,
    rotation_error_angle,
    rotation_matrix_to_quaternion_xyzw,
    rotation_vector_to_matrix,
)

def test_quaternion_delta_rotation_vector_uses_shortest_yaw_branch():
    angle = -math.pi / 2.0
    current = np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])
    result = quaternion_delta_rotation_vector(np.array([0.0, 0.0, 0.0, 1.0]), current)
    np.testing.assert_allclose(result, [0.0, 0.0, angle], atol=1e-8)

def test_rotation_vector_matrix_quaternion_round_trip():
    vector = np.array([0.2, -0.3, 0.4])
    rotation = rotation_vector_to_matrix(vector)
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    np.testing.assert_allclose(quaternion_xyzw_to_matrix(quaternion), rotation, atol=1e-8)
    assert rotation_error_angle(rotation, rotation) == pytest.approx(0.0, abs=1e-8)
```

- [ ] **Step 2: Run transform tests and confirm the missing imports fail**

Run:

```bash
pytest tests/utils/test_transforms.py -q
```

Expected: collection fails because the five new transform functions are not defined.

- [ ] **Step 3: Implement the transform API**

Add normalized xyzw conversions, shortest-branch quaternion log, Rodrigues exponential, and trace-based rotation error. Use these exact formulas:

```python
def quaternion_delta_rotation_vector(anchor_xyzw: np.ndarray, current_xyzw: np.ndarray) -> np.ndarray:
    anchor = _normalize_quat(anchor_xyzw)
    current = _normalize_quat(current_xyzw)
    delta = _normalize_quat(_quat_multiply(current, _quat_conjugate(anchor)))
    if delta[3] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(delta[3]))
    return delta[:3] * (angle / vector_norm)

def rotation_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("rotation vector must be a finite shape-(3,) vector")
    angle = float(np.linalg.norm(values))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = values / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)

def rotation_error_angle(target: np.ndarray, actual: np.ndarray) -> float:
    relative = np.asarray(target, dtype=float) @ np.asarray(actual, dtype=float).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))
```

Implement `quaternion_xyzw_to_matrix` with the normalized standard xyzw formula. Move the existing trace-branch matrix-to-quaternion implementation from `dex_teleop/robots/franka.py` into public `rotation_matrix_to_quaternion_xyzw`, retaining normalization and shape/finiteness checks.

- [ ] **Step 4: Add failing pure target-construction tests**

Create `tests/policies/test_ik_target.py` with identity mapping, axis remapping, base-frame rotation composition, snapshot immutability, and direction-preserving clamps:

```python
import math
import numpy as np
from dex_teleop.core.types import Pose
from dex_teleop.policies.ik_target import ClutchAnchor, build_clutch_target

IDENTITY_POSITION = {"x": "+x", "y": "+y", "z": "+z"}
IDENTITY_ROTATION = {"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"}

def test_build_clutch_target_maps_full_pose_and_clamps_anchor_norms():
    anchor = ClutchAnchor.snapshot(Pose.identity(), Pose.identity())
    yaw = 1.0
    controller = Pose([0.3, 0.4, 0.0], [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    result = build_clutch_target(
        controller, anchor,
        position_axes=IDENTITY_POSITION,
        rotation_axes=IDENTITY_ROTATION,
        position_scale=1.0,
        rotation_scale=1.0,
        max_translation_from_anchor_m=0.1,
        max_rotation_from_anchor_rad=0.5,
    )
    np.testing.assert_allclose(result.pose.position, [0.06, 0.08, 0.0], atol=1e-8)
    assert result.translation_clamped is True
    assert result.rotation_clamped is True
```

- [ ] **Step 5: Run the target test and confirm the module is missing**

Run: `pytest tests/policies/test_ik_target.py -q`

Expected: FAIL with `ModuleNotFoundError: dex_teleop.policies.ik_target`.

- [ ] **Step 6: Implement immutable anchors and target construction**

Create the module with these dataclasses and function signature:

```python
from dataclasses import dataclass
import numpy as np
from dex_teleop.core.types import Pose
from dex_teleop.utils.transforms import (
    axis_mapping, clamp_vector_norm, quaternion_delta_rotation_vector,
    quaternion_xyzw_to_matrix, rotation_matrix_to_quaternion_xyzw,
    rotation_vector_to_matrix,
)

@dataclass(frozen=True)
class ClutchAnchor:
    controller_pose: Pose
    robot_pose: Pose

    @classmethod
    def snapshot(cls, controller_pose: Pose, robot_pose: Pose) -> "ClutchAnchor":
        return cls(
            Pose(controller_pose.position.copy(), controller_pose.orientation.copy()),
            Pose(robot_pose.position.copy(), robot_pose.orientation.copy()),
        )

@dataclass(frozen=True)
class TargetPoseResult:
    pose: Pose
    translation_clamped: bool
    rotation_clamped: bool
    mapped_translation: np.ndarray
    mapped_rotation_vector: np.ndarray

def build_clutch_target(
    controller_pose: Pose,
    anchor: ClutchAnchor,
    *,
    position_axes: dict[str, str],
    rotation_axes: dict[str, str],
    position_scale: float,
    rotation_scale: float,
    max_translation_from_anchor_m: float,
    max_rotation_from_anchor_rad: float,
) -> TargetPoseResult:
    raw_translation = controller_pose.position - anchor.controller_pose.position
    mapped_translation = axis_mapping(raw_translation, position_axes, ("x", "y", "z"), ("x", "y", "z")) * position_scale
    limited_translation = clamp_vector_norm(mapped_translation, max_translation_from_anchor_m)
    raw_rotation = quaternion_delta_rotation_vector(anchor.controller_pose.orientation, controller_pose.orientation)
    mapped_rotation = axis_mapping(raw_rotation, rotation_axes, ("roll", "pitch", "yaw"), ("roll", "pitch", "yaw")) * rotation_scale
    limited_rotation = clamp_vector_norm(mapped_rotation, max_rotation_from_anchor_rad)
    anchor_rotation = quaternion_xyzw_to_matrix(anchor.robot_pose.orientation)
    target_rotation = rotation_vector_to_matrix(limited_rotation) @ anchor_rotation
    return TargetPoseResult(
        pose=Pose(anchor.robot_pose.position + limited_translation, rotation_matrix_to_quaternion_xyzw(target_rotation)),
        translation_clamped=not np.allclose(mapped_translation, limited_translation),
        rotation_clamped=not np.allclose(mapped_rotation, limited_rotation),
        mapped_translation=limited_translation,
        mapped_rotation_vector=limited_rotation,
    )
```

- [ ] **Step 7: Run focused tests, then commit only these hunks/files**

Run: `pytest tests/utils/test_transforms.py tests/policies/test_ik_target.py -q`

Expected: PASS.

Commit after cached-diff inspection:

```bash
git add dex_teleop/utils/transforms.py dex_teleop/policies/ik_target.py tests/utils/test_transforms.py tests/policies/test_ik_target.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: add 6DoF IK target construction"
```

### Task 2: Joint Setpoint Velocity and Acceleration Shaping

**Files:**
- Create: `dex_teleop/policies/joint_setpoint.py`
- Create: `tests/policies/test_joint_setpoint.py`

**Interfaces:**
- Consumes: finite seven-joint targets and lower/upper joint bounds.
- Produces: `JointSetpointLimiter(max_velocity, max_acceleration, max_dt_s)`, `reset(joint_positions)`, and `step(target, dt, lower_limits, upper_limits) -> tuple[np.ndarray, np.ndarray]` returning `(q_cmd, dq_cmd)`.

- [ ] **Step 1: Write limiter failure tests**

Create tests for acceleration ramp, velocity cap, no overshoot, joint clipping, scalar/list configuration, and delayed-cycle `dt` cap:

```python
import numpy as np
from dex_teleop.policies.joint_setpoint import JointSetpointLimiter

def test_limiter_ramps_velocity_without_overshooting():
    limiter = JointSetpointLimiter(0.5, 1.0, 0.1)
    limiter.reset(np.zeros(7))
    q_cmd, dq_cmd = limiter.step(np.ones(7), dt=0.05, lower_limits=-np.ones(7) * 2, upper_limits=np.ones(7) * 2)
    np.testing.assert_allclose(dq_cmd, np.ones(7) * 0.05)
    np.testing.assert_allclose(q_cmd, np.ones(7) * 0.0025)

def test_limiter_caps_delayed_dt_and_stops_exactly_at_target():
    limiter = JointSetpointLimiter([0.5] * 7, [100.0] * 7, 0.1)
    limiter.reset(np.zeros(7))
    q_cmd, dq_cmd = limiter.step(np.ones(7) * 0.01, dt=1.0, lower_limits=-np.ones(7), upper_limits=np.ones(7))
    np.testing.assert_allclose(q_cmd, np.ones(7) * 0.01)
    np.testing.assert_allclose(dq_cmd, np.zeros(7))
```

- [ ] **Step 2: Run the limiter tests and confirm the module is missing**

Run: `pytest tests/policies/test_joint_setpoint.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the limiter**

Normalize scalar-or-seven-element limits once, retain private `_q_cmd` and `_dq_cmd`, and implement this update order:

```python
effective_dt = min(float(dt), self.max_dt_s)
error = target - self._q_cmd
desired_dq = np.clip(error / effective_dt, -self.max_velocity, self.max_velocity)
max_delta_dq = self.max_acceleration * effective_dt
next_dq = self._dq_cmd + np.clip(desired_dq - self._dq_cmd, -max_delta_dq, max_delta_dq)
next_q = self._q_cmd + next_dq * effective_dt
overshot = np.sign(target - self._q_cmd) != np.sign(target - next_q)
next_q[overshot] = target[overshot]
next_dq[overshot] = 0.0
next_q = np.clip(next_q, lower_limits, upper_limits)
next_dq[(next_q <= lower_limits) | (next_q >= upper_limits)] = 0.0
```

Reject non-finite data, nonpositive `dt`, nonpositive limits, inverted joint bounds, or calls before `reset`. Return copies so callers cannot mutate internal state.

- [ ] **Step 4: Verify and commit the isolated limiter**

Run: `pytest tests/policies/test_joint_setpoint.py -q`

Expected: PASS.

```bash
git add dex_teleop/policies/joint_setpoint.py tests/policies/test_joint_setpoint.py
git diff --cached --check
git commit -m "feat: add joint setpoint shaping"
```

### Task 3: Joint Command Types and Robot Adapters

**Files:**
- Modify: `dex_teleop/core/types.py:59-134`
- Modify: `dex_teleop/robots/arm.py:5-24`
- Modify: `dex_teleop/robots/franka.py:11-293`
- Modify: `dex_teleop/robots/mock.py:3-37`
- Modify: `tests/core/test_types.py:6-149`
- Modify: `tests/robots/test_franka_adapter.py:9-370`
- Modify: `tests/robots/test_mock_robot.py:3-36`

**Interfaces:**
- Consumes: Task 1 `rotation_matrix_to_quaternion_xyzw`.
- Produces: `ArmState.joint_positions`, `JointPositionCommand`, `TeleopCommandBatch.joint_positions`, `ArmController.send_joint_position`, and concrete Franka/mock implementations.

- [ ] **Step 1: Add failing type and adapter tests**

Add tests proving optional joint state compatibility, finite shape validation, disabled construction, state extraction from `interface.last_q`, exact eight-element Deoxys action, controller mismatch rejection, and mock recording:

```python
def test_joint_position_command_validates_and_disables():
    command = JointPositionCommand("right", np.arange(7) * 0.1)
    disabled = JointPositionCommand.disabled("left")
    assert command.joint_positions.shape == (7,)
    assert disabled.enabled is False
    np.testing.assert_allclose(disabled.joint_positions, np.zeros(7))

def test_franka_arm_sends_absolute_joint_position_action():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("right", "unused.yml", interface=fake, controller_cfg={}, controller_type="JOINT_POSITION")
    arm.connect()
    arm.send_joint_position(JointPositionCommand("right", [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]))
    assert fake.control_calls[-1] == (
        "JOINT_POSITION", [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.0], {}, False
    )
```

Set `FakeFrankaInterface.last_q` to a finite seven-vector and assert it appears
in `get_state()`. When `last_q` is missing, wrong-shaped, or non-finite, assert
the adapter still returns the valid Cartesian pose with
`joint_positions is None`; the IK policy, rather than the generic adapter,
rejects that incomplete state. This preserves existing Cartesian policy
behavior.

- [ ] **Step 2: Run focused tests and confirm missing symbols/behavior**

Run:

```bash
pytest tests/core/test_types.py tests/robots/test_franka_adapter.py tests/robots/test_mock_robot.py -q
```

Expected: FAIL because `JointPositionCommand`, joint state extraction, and adapter methods do not exist.

- [ ] **Step 3: Add the command/state types**

Extend `ArmState` compatibly and add the command before `GripperCommand`:

```python
@dataclass
class ArmState:
    arm_id: str
    pose: Pose
    joint_positions: np.ndarray | list[float] | tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.joint_positions is not None:
            self.joint_positions = _array(self.joint_positions, (7,), "joint_positions")
            if not np.all(np.isfinite(self.joint_positions)):
                raise ValueError("joint_positions must contain only finite values")

@dataclass
class JointPositionCommand:
    arm_id: str
    joint_positions: np.ndarray | list[float] | tuple[float, ...]
    enabled: bool = True

    def __post_init__(self) -> None:
        self.joint_positions = _array(self.joint_positions, (7,), "joint_positions")
        if not np.all(np.isfinite(self.joint_positions)):
            raise ValueError("joint_positions must contain only finite values")

    @classmethod
    def disabled(cls, arm_id: str) -> "JointPositionCommand":
        return cls(arm_id, np.zeros(7), enabled=False)
```

Add `joint_positions: dict[str, JointPositionCommand] = field(default_factory=dict)` to `TeleopCommandBatch`.

- [ ] **Step 4: Implement protocol and adapters**

Import the new type in arm, Franka, and mock modules. Add:

```python
def send_joint_position(self, command: JointPositionCommand) -> None:
    if not command.enabled:
        self.stop_motion()
        return
    if self.controller_type != "JOINT_POSITION":
        raise RuntimeError(
            "send_joint_position requires controller_type='JOINT_POSITION', "
            f"got {self.controller_type!r}"
        )
    action = command.joint_positions.tolist() + [0.0]
    self._control(action, termination=False)
```

In `FrankaArmController.get_state`, read `last_q` when available, convert a
finite shape-(7,) value to a copy, and otherwise set `joint_positions=None`
without discarding an otherwise valid pose. Replace the private matrix
conversion with Task 1's public helper. In `MockArmController`, initialize
`self.joint_positions = np.zeros(7)`, return it from `get_state`, record
`joint_position_commands`, and update the mock joint state when an enabled
joint command is sent.

- [ ] **Step 5: Run focused and compatibility tests**

Run:

```bash
pytest tests/core/test_types.py tests/robots/test_franka_adapter.py tests/robots/test_mock_robot.py tests/policies/test_bimanual_clutch.py tests/policies/test_bimanual_cartesian_velocity.py -q
```

Expected: PASS, including existing callers that construct `ArmState` without joints.

- [ ] **Step 6: Commit only separable feature hunks**

```bash
git add -p dex_teleop/core/types.py dex_teleop/robots/arm.py dex_teleop/robots/franka.py dex_teleop/robots/mock.py tests/core/test_types.py tests/robots/test_franka_adapter.py tests/robots/test_mock_robot.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: add Franka joint position commands"
```

If interactive staging shows a hunk containing pre-existing Cartesian-velocity work, do not stage that hunk; keep the tested working-tree change and report that its commit is deferred.

### Task 4: Relative-Pose MuJoCo IK Solver

**Files:**
- Create: `dex_teleop/retargeting/mujoco_pose_ik.py`
- Create: `tests/retargeting/test_mujoco_pose_ik.py`

**Interfaces:**
- Consumes: `Pose` and Task 1 transform functions; existing `deoxys.utils.ik_utils.IKWrapper`.
- Produces: immutable `PoseIKSolution` and `MujocoPoseIKSolver.solve(seed_joint_positions, measured_robot_pose, target_robot_pose) -> PoseIKSolution`.

- [ ] **Step 1: Write reachable full-pose and residual tests**

Use the real model, home joints from the existing benchmark, and a small combined translation/rotation target:

```python
import numpy as np
from dex_teleop.core.types import Pose
from dex_teleop.retargeting.mujoco_pose_ik import MujocoPoseIKSolver
from dex_teleop.utils.transforms import rotation_matrix_to_quaternion_xyzw, rotation_vector_to_matrix

HOME = np.array([0.09162, -0.19826, -0.01990, -2.47323, -0.01307, 2.30397, 0.84809])

def test_solver_reaches_relative_translation_and_rotation():
    solver = MujocoPoseIKSolver()
    measured = solver.forward_pose(HOME)
    target_rotation = rotation_vector_to_matrix(np.array([0.02, -0.01, 0.03])) @ solver.forward_rotation(HOME)
    target = Pose(measured.position + np.array([0.005, -0.004, 0.006]), rotation_matrix_to_quaternion_xyzw(target_rotation))
    solution = solver.solve(HOME, measured, target)
    assert solution.position_residual_m <= 0.005
    assert solution.rotation_residual_rad <= 0.05
    assert solution.joint_positions.shape == (7,)
    assert np.all(solution.joint_positions >= solution.lower_limits)
    assert np.all(solution.joint_positions <= solution.upper_limits)
```

Also test a nonidentity measured pose offset to prove the solver applies robot
pose error to model FK rather than treating the real pose as a model-absolute
target.

- [ ] **Step 2: Run the integration test and confirm the module is missing**

Run: `pytest tests/retargeting/test_mujoco_pose_ik.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the solver and residual result**

Create:

```python
@dataclass(frozen=True)
class PoseIKSolution:
    joint_positions: np.ndarray
    position_residual_m: float
    rotation_residual_rad: float
    lower_limits: np.ndarray
    upper_limits: np.ndarray

class MujocoPoseIKSolver:
    def __init__(self) -> None:
        _ensure_deoxys_on_path()
        from deoxys.utils.ik_utils import IKWrapper
        self.wrapper = IKWrapper()
        self.model = self.wrapper.model
        self.data = self.wrapper.data
        self.site_id = self.model.site("grip_site").id
        self.lower_limits = np.asarray(self.model.jnt_range[:7, 0], dtype=float)
        self.upper_limits = np.asarray(self.model.jnt_range[:7, 1], dtype=float)

    def forward_pose(self, joint_positions: np.ndarray) -> Pose:
        rotation = self.forward_rotation(joint_positions)
        position = np.copy(self.data.site(self.site_id).xpos)
        return Pose(position, rotation_matrix_to_quaternion_xyzw(rotation))

    def forward_rotation(self, joint_positions: np.ndarray) -> np.ndarray:
        q = _finite_joints(joint_positions)
        self.data.qpos[:] = q.tolist() + [0.04, 0.04]
        mujoco.mj_fwdPosition(self.model, self.data)
        return np.copy(self.data.site(self.site_id).xmat).reshape(3, 3)

    def solve(self, seed_joint_positions: np.ndarray, measured_robot_pose: Pose, target_robot_pose: Pose) -> PoseIKSolution:
        seed = _finite_joints(seed_joint_positions)
        model_pose = self.forward_pose(seed)
        model_rotation = quaternion_xyzw_to_matrix(model_pose.orientation)
        measured_rotation = quaternion_xyzw_to_matrix(measured_robot_pose.orientation)
        target_rotation = quaternion_xyzw_to_matrix(target_robot_pose.orientation)
        model_target_position = model_pose.position + target_robot_pose.position - measured_robot_pose.position
        model_target_rotation = target_rotation @ measured_rotation.T @ model_rotation
        solved = np.asarray(self.wrapper.inverse_kinematics(
            self.model, self.data, model_target_rotation, model_target_position, seed.tolist()
        ), dtype=float)
        solved_pose = self.forward_pose(solved)
        return PoseIKSolution(
            solved.copy(),
            float(np.linalg.norm(solved_pose.position - model_target_position)),
            rotation_error_angle(model_target_rotation, quaternion_xyzw_to_matrix(solved_pose.orientation)),
            self.lower_limits.copy(),
            self.upper_limits.copy(),
        )
```

Define local `_ensure_deoxys_on_path` using the repository dependency path and
set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, matching the Franka
adapter. `_finite_joints` enforces shape `(7,)` and finite values.

- [ ] **Step 4: Verify deterministic model behavior and commit**

Run:

```bash
pytest tests/retargeting/test_mujoco_pose_ik.py -q
pytest tests/scripts/test_benchmark_deoxys_controllers_script.py -q
```

Expected: PASS.

```bash
git add dex_teleop/retargeting/mujoco_pose_ik.py tests/retargeting/test_mujoco_pose_ik.py
git diff --cached --check
git commit -m "feat: add relative pose MuJoCo IK solver"
```

### Task 5: Latest-Only Asynchronous IK Worker

**Files:**
- Create: `dex_teleop/retargeting/latest_ik_worker.py`
- Create: `tests/retargeting/test_latest_ik_worker.py`

**Interfaces:**
- Consumes: any solver implementing the Task 4 `solve` signature.
- Produces: `IKRequest.snapshot(...)`, `IKWorkerResult`, `LatestTargetIKWorker.submit`, `poll`, `overwrite_count`, and `close(timeout_s=1.0)`.

- [ ] **Step 1: Write deterministic concurrency tests**

Use events instead of sleeps so tests prove replacement and nonblocking poll:

```python
import threading
import numpy as np
from dex_teleop.core.types import Pose
from dex_teleop.retargeting.latest_ik_worker import IKRequest, LatestTargetIKWorker
from dex_teleop.retargeting.mujoco_pose_ik import PoseIKSolution

class BlockingSolver:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def solve(self, seed_joint_positions, measured_robot_pose, target_robot_pose):
        self.calls.append(target_robot_pose.position.copy())
        self.started.set()
        self.release.wait(timeout=1.0)
        return PoseIKSolution(
            joint_positions=np.zeros(7),
            position_residual_m=0.0,
            rotation_residual_rad=0.0,
            lower_limits=-np.ones(7),
            upper_limits=np.ones(7),
        )

def request(request_id: int, x: float) -> IKRequest:
    return IKRequest.snapshot(1, request_id, float(request_id), np.zeros(7), Pose.identity(), Pose([x, 0, 0], [0, 0, 0, 1]))

def test_pending_slot_keeps_only_latest_request():
    solver = BlockingSolver()
    worker = LatestTargetIKWorker(solver)
    worker.submit(request(1, 0.1))
    assert solver.started.wait(timeout=1.0)
    worker.submit(request(2, 0.2))
    worker.submit(request(3, 0.3))
    assert worker.overwrite_count == 1
    solver.release.set()
    worker.close(timeout_s=1.0)
    assert any(np.allclose(call, [0.3, 0.0, 0.0]) for call in solver.calls)
```

Add tests that `poll()` returns immediately, errors become result data instead
of killing the thread silently, snapshots cannot be mutated through source
arrays, and `close` raises if a worker cannot join within its bound.

- [ ] **Step 2: Run and confirm the missing worker module**

Run: `pytest tests/retargeting/test_latest_ik_worker.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement request/result snapshots and worker lifecycle**

Use these public records:

```python
@dataclass(frozen=True)
class IKRequest:
    epoch: int
    request_id: int
    submitted_at: float
    seed_joint_positions: np.ndarray
    measured_robot_pose: Pose
    target_robot_pose: Pose

    @classmethod
    def snapshot(cls, epoch, request_id, submitted_at, seed, measured, target):
        return cls(
            int(epoch), int(request_id), float(submitted_at), np.asarray(seed, dtype=float).copy(),
            Pose(measured.position.copy(), measured.orientation.copy()),
            Pose(target.position.copy(), target.orientation.copy()),
        )

@dataclass(frozen=True)
class IKWorkerResult:
    request: IKRequest
    completed_at: float
    solution: PoseIKSolution | None
    error_type: str | None = None
    error_message: str | None = None
```

Implement a `Condition`, one pending slot, one result slot, and a daemon-free
thread. `submit` increments `overwrite_count` only when it replaces an
existing pending request. `_run` waits for work, clears the pending slot before
solving, stores either a solution or exception metadata, and exits only after
close is requested and no pending work remains. `poll` atomically returns and
clears the result. `close` signals, joins, and raises `RuntimeError` if the
thread remains alive after `timeout_s`.

- [ ] **Step 4: Verify race-free behavior repeatedly and commit**

Run:

```bash
pytest tests/retargeting/test_latest_ik_worker.py -q
for i in {1..20}; do pytest tests/retargeting/test_latest_ik_worker.py -q || exit 1; done
```

Expected: all runs PASS.

```bash
git add dex_teleop/retargeting/latest_ik_worker.py tests/retargeting/test_latest_ik_worker.py
git diff --cached --check
git commit -m "feat: add latest target IK worker"
```

### Task 6: Bimanual IK Joint-Position Policy and Safety State Machine

**Files:**
- Create: `dex_teleop/policies/bimanual_ik_joint_position.py`
- Create: `tests/policies/test_bimanual_ik_joint_position.py`
- Modify: `dex_teleop/policies/__init__.py:1-6`

**Interfaces:**
- Consumes: Tasks 1-5 target, limiter, command, worker, and solution interfaces.
- Produces: `ArmSafetyState`, `BimanualIKJointPositionPolicy.from_config(cfg, worker_factory, clock)`, `update(state, arm_states)`, `close()`, and `last_diagnostics`.

- [ ] **Step 1: Build a fake-worker policy fixture and failing state tests**

The fake worker records requests and lets tests inject exact results without
threads:

```python
class FakeWorker:
    def __init__(self, solver=None):
        self.requests = []
        self.results = []
        self.overwrite_count = 0
        self.closed = False

    def submit(self, request):
        self.requests.append(request)

    def poll(self):
        return self.results.pop(0) if self.results else None

    def close(self, timeout_s=1.0):
        self.closed = True
```

Add focused tests with one or two arms proving:

```python
def test_released_grip_holds_measured_joint_position():
    policy, workers = make_policy()
    measured_q = np.arange(7) * 0.1
    batch = policy.update(device_state(grip=False), arm_states={"right": arm_state(measured_q)})
    np.testing.assert_allclose(batch.joint_positions["right"].joint_positions, measured_q)
    assert policy.last_diagnostics["right"]["safety_state"] == "HOLDING"

def test_first_press_anchors_and_next_motion_submits_full_pose_request():
    policy, workers = make_policy()
    q = np.zeros(7)
    policy.update(device_state(grip=True), arm_states={"right": arm_state(q)})
    hold = policy.update(device_state(grip=True, position=(0.01, 0.0, 0.0), yaw=0.05), arm_states={"right": arm_state(q)})
    assert hold.joint_positions["right"].enabled is True
    assert workers["right"].requests[-1].target_robot_pose.position[0] == pytest.approx(0.01)
```

Also cover missing state (`WAITING_FOR_STATE`), old epoch result discard,
over-age result discard, residual/joint-limit/jump fault, no-fresh-result
timeout, release/re-press recovery, independent left/right epochs, trigger
gripper output, and worker closure.

- [ ] **Step 2: Run and confirm the policy module is missing**

Run: `pytest tests/policies/test_bimanual_ik_joint_position.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement per-arm runtime state and construction**

Define:

```python
class ArmSafetyState(str, Enum):
    WAITING_FOR_STATE = "WAITING_FOR_STATE"
    HOLDING = "HOLDING"
    TRACKING = "TRACKING"
    FAULT_HOLD = "FAULT_HOLD"

@dataclass
class _ArmRuntime:
    safety_state: ArmSafetyState = ArmSafetyState.WAITING_FOR_STATE
    epoch: int = 0
    next_request_id: int = 1
    last_accepted_request_id: int = 0
    anchor: ClutchAnchor | None = None
    hold_q: np.ndarray | None = None
    accepted_target_q: np.ndarray | None = None
    last_accepted_at: float | None = None
    last_target: Pose | None = None
    fault_reason: str | None = None
```

`from_config` builds one runtime, limiter, and worker per configured arm. Pass
`worker_factory` and `clock` as injectable keyword dependencies. Copy all
numeric limits from `ik_joint_position`, normalize scalar/list joint limits,
and retain existing arm/controller/gripper bindings.

- [ ] **Step 4: Implement explicit transition helpers**

Use small helpers rather than embedding resets throughout `update`:

```python
def _enter_hold(self, arm_id: str, measured_q: np.ndarray, reason: str) -> None:
    runtime = self._runtime[arm_id]
    runtime.epoch += 1
    runtime.safety_state = ArmSafetyState.HOLDING
    runtime.anchor = None
    runtime.hold_q = measured_q.copy()
    runtime.accepted_target_q = measured_q.copy()
    runtime.last_accepted_at = self._clock()
    runtime.fault_reason = reason
    self._limiters[arm_id].reset(measured_q)

def _enter_fault(self, arm_id: str, measured_q: np.ndarray, reason: str) -> None:
    self._enter_hold(arm_id, measured_q, reason)
    self._runtime[arm_id].safety_state = ArmSafetyState.FAULT_HOLD
```

Releasing grip from any state calls `_enter_hold`. A press from `HOLDING`
increments the epoch again, snapshots anchors, resets request ordering, enters
`TRACKING`, and initially emits the held joints. `FAULT_HOLD` with grip still
pressed never submits work; release moves to `HOLDING`, and the next press
creates a new epoch.

- [ ] **Step 5: Implement request submission, result validation, shaping, and diagnostics**

For each arm during `TRACKING`:

1. Build the Task 1 target.
2. Snapshot an `IKRequest` with current measured q/pose, current epoch, next
   request ID, and monotonic submission time; submit without waiting.
3. Poll at most one result.
4. Discard old-epoch, old-request-ID, or over-age results and record the reason.
5. Enter fault for worker error, non-finite values, residual overflow, joint
   limit margin violation, or per-component jump over 0.20 rad.
6. On acceptance, store target q, request ID, and acceptance time.
7. Run `JointSetpointLimiter.step` toward the accepted target; until the first
   acceptance, emit the anchor hold q.
8. If the target differs from the accepted target beyond existing position or
   rotation deadzones and no fresh acceptance arrives within 0.15 s, enter
   fault hold.

Populate `last_diagnostics[arm_id]` with exact keys:

```python
{
    "safety_state": runtime.safety_state.value,
    "epoch": runtime.epoch,
    "latest_submitted_request_id": runtime.next_request_id - 1,
    "last_accepted_request_id": runtime.last_accepted_request_id,
    "hold_reason": runtime.fault_reason,
    "target_pose": target.pose if target is not None else None,
    "controller_anchor_pose": runtime.anchor.controller_pose if runtime.anchor is not None else None,
    "robot_anchor_pose": runtime.anchor.robot_pose if runtime.anchor is not None else None,
    "translation_clamped": target.translation_clamped if target is not None else False,
    "rotation_clamped": target.rotation_clamped if target is not None else False,
    "ik": ik_diagnostics,
    "q_ik_target": runtime.accepted_target_q,
    "q_cmd": q_cmd,
    "dq_cmd": dq_cmd,
    "worker_overwrite_count": self._workers[arm_id].overwrite_count,
}
```

`ik_diagnostics` always contains the same keys, using `None` when no result is
available: `request_id`, `seed_joint_positions`, `candidate_joint_positions`,
`solve_duration_s`, `result_age_s`, `position_residual_m`,
`rotation_residual_rad`, `rejection_reason`, `worker_error_type`, and
`worker_error_message`. This exact mapping is what the app serializes into
JSONL.

`close()` calls every worker's bounded close and continues closing remaining
workers if one raises, then re-raises the first failure.

- [ ] **Step 6: Run policy, transform, limiter, and worker tests together**

Run:

```bash
pytest tests/policies/test_bimanual_ik_joint_position.py tests/policies/test_ik_target.py tests/policies/test_joint_setpoint.py tests/retargeting/test_latest_ik_worker.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the policy unit**

```bash
git add dex_teleop/policies/bimanual_ik_joint_position.py dex_teleop/policies/__init__.py tests/policies/test_bimanual_ik_joint_position.py
git diff --cached --check
git commit -m "feat: add IK joint position teleop policy"
```

### Task 7: Conditional Config, Policy Factory, App Dispatch, and JSONL Telemetry

**Files:**
- Create: `dex_teleop/policies/factory.py`
- Modify: `dex_teleop/core/config.py:14-157`
- Modify: `dex_teleop/apps/oculus_franka.py:17-496`
- Create: `configs/oculus_franka_right_ik_joint_position.yml`
- Modify: `tests/core/test_config.py:10-300`
- Modify: `tests/apps/test_oculus_franka_app.py`

**Interfaces:**
- Consumes: Task 6 policy and Task 3 command/adapter methods.
- Produces: `make_policy(cfg)`, valid right-only YAML, application joint command dispatch, lifecycle cleanup, printed joint commands, and expanded JSONL.

- [ ] **Step 1: Add failing conditional-config tests**

Add tests asserting legacy default behavior, exact IK keys, scalar/list numeric
validation, strategy/controller compatibility, and right-only config content:

```python
def test_legacy_config_defaults_to_cartesian_velocity_strategy():
    cfg = load_config("configs/oculus_franka_bimanual.yml")
    assert cfg["app"].get("strategy", "cartesian_velocity") == "cartesian_velocity"

def test_right_ik_config_is_right_only_joint_position():
    cfg = load_config("configs/oculus_franka_right_ik_joint_position.yml")
    assert cfg["app"]["strategy"] == "ik_joint_position"
    assert set(cfg["arms"]) == {"right"}
    assert cfg["arms"]["right"]["controller_type"] == "JOINT_POSITION"
    assert cfg["ik_joint_position"]["max_dt_s"] == 0.10
```

Create a temporary IK config with one required field removed and assert a
path-specific `ConfigError`. Test invalid strategy and a seven-element limit
list of the wrong length.

- [ ] **Step 2: Run config tests and confirm the new config/schema fails**

Run: `pytest tests/core/test_config.py -q`

Expected: FAIL because the right IK config and conditional schema do not exist.

- [ ] **Step 3: Implement conditional validation and the right-only config**

Remove `cartesian_velocity` from unconditional `REQUIRED_TOP_LEVEL`. Determine:

```python
strategy = str(app.get("strategy", "cartesian_velocity"))
if strategy == "cartesian_velocity":
    cartesian_velocity = _require_mapping(_require(cfg, "cartesian_velocity"), "cartesian_velocity")
    _validate_required_keys(cartesian_velocity, REQUIRED_CARTESIAN_VELOCITY, "cartesian_velocity")
elif strategy == "ik_joint_position":
    ik = _require_mapping(_require(cfg, "ik_joint_position"), "ik_joint_position")
    _validate_required_keys(ik, REQUIRED_IK_JOINT_POSITION, "ik_joint_position")
    for arm_id, arm_cfg in arms.items():
        if arm_cfg.get("controller_type") != "JOINT_POSITION":
            raise ConfigError(f"arms.{arm_id}.controller_type must be 'JOINT_POSITION' for ik_joint_position")
else:
    raise ConfigError(f"Unsupported app.strategy: {strategy!r}")
```

Define `REQUIRED_IK_JOINT_POSITION` with the ten exact fields from Global
Constraints, including `max_dt_s`. Add helpers that accept a finite positive
scalar or a finite positive seven-list for velocity/acceleration and require a
finite positive scalar for all other IK values.

Build `configs/oculus_franka_right_ik_joint_position.yml` from the current
right-arm mapping/input/gripper configuration, remove `cartesian_velocity`,
set `strategy` and `controller_type`, and copy the ten exact initial values.

- [ ] **Step 4: Add failing factory and app integration tests**

Test factory selection with monkeypatched constructors, then run a right-only
dry-run with a fake IK policy that emits one `JointPositionCommand`. Assert:

```python
assert cycle["arm_states"]["right"]["joint_positions"] == [0.0] * 7
joint = cycle["commands"]["joint_positions"]["right"]
assert joint["joint_positions"] == [0.0] * 7
assert joint["sent"] is True
assert cycle["policy"]["right"]["safety_state"] == "HOLDING"
assert records[0]["strategy"] == "ik_joint_position"
assert records[-1]["event"] == "shutdown"
```

Add a policy double with `close()` and assert it closes before arm adapters.
Retain the existing velocity JSONL test unchanged to prove coexistence.

- [ ] **Step 5: Run integration tests and confirm factory/dispatch gaps**

Run:

```bash
pytest tests/core/test_config.py tests/apps/test_oculus_franka_app.py -q
```

Expected: FAIL at missing `make_policy`, joint dispatch, state joints, or new JSONL fields.

- [ ] **Step 6: Implement policy factory and app integration**

Create:

```python
def make_policy(cfg: dict[str, Any]):
    strategy = str(cfg["app"].get("strategy", "cartesian_velocity"))
    if strategy == "cartesian_velocity":
        return BimanualCartesianVelocityPolicy.from_config(cfg)
    if strategy == "ik_joint_position":
        return BimanualIKJointPositionPolicy.from_config(cfg)
    raise ValueError(f"Unsupported app.strategy: {strategy!r}")
```

In the app, replace the hardcoded policy constructor, add
`_joint_position_command_record`, include joint state in `_arm_states_record`,
print joint commands, and dispatch `batch.joint_positions` through
`arm.send_joint_position`. Add `joint_positions` beside existing motion groups
in every cycle JSONL record. Write `strategy` in start/shutdown records.

Add a startup-failure test whose IK worker constructor raises; assert no arm
receives a command, the JSONL error/shutdown records are flushed, and every
already-created resource is closed.

Initialize `policy = None`; in `finally`, call `policy.close` when present
before stopping/closing arms. Preserve best-effort cleanup and ensure logger
closure even if policy or worker initialization raises.

- [ ] **Step 7: Verify both strategies with dry-run smoke commands**

Run:

```bash
python -m dex_teleop.apps.oculus_franka --config configs/oculus_franka_bimanual.yml --mock-controller --dry-run --duration 0.15 --teleop-log-file /tmp/cartesian_velocity_smoke.jsonl
python -m dex_teleop.apps.oculus_franka --config configs/oculus_franka_right_ik_joint_position.yml --mock-controller --dry-run --duration 0.15 --teleop-log-file /tmp/ik_joint_position_smoke.jsonl
```

Expected: both exit 0, print `Teleop shutdown complete`, and end their JSONL
files with a shutdown record. The IK file contains only a right-arm joint
command group; the velocity file retains its existing velocity group.

- [ ] **Step 8: Run focused tests and commit separable integration hunks**

Run:

```bash
pytest tests/core/test_config.py tests/apps/test_oculus_franka_app.py tests/robots/test_franka_adapter.py -q
```

Expected: PASS.

```bash
git add dex_teleop/policies/factory.py configs/oculus_franka_right_ik_joint_position.yml
git add -p dex_teleop/core/config.py dex_teleop/apps/oculus_franka.py tests/core/test_config.py tests/apps/test_oculus_franka_app.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: integrate IK joint position teleop"
```

Do not stage pre-existing logging, stale-state, or Cartesian-velocity hunks
that are already present in the dirty workspace unless the feature hunk cannot
function without them; report any inseparable baseline dependency explicitly.

### Task 8: Full Verification and Right-Arm Hardware Acceptance

**Files:**
- Modify only if verification exposes a feature defect: files from Tasks 1-7 and their matching tests.
- Runtime artifacts: `logs/oculus_franka_<timestamp>.jsonl` and a concise acceptance note in the final handoff; do not commit generated logs.

**Interfaces:**
- Consumes: the complete feature.
- Produces: test evidence, dry-run evidence, timing/continuity metrics, and a staged hardware-validation sequence that never connects the left arm.

- [ ] **Step 1: Run formatting-independent static checks and the full suite**

Run:

```bash
python -m py_compile \
  dex_teleop/policies/ik_target.py \
  dex_teleop/policies/joint_setpoint.py \
  dex_teleop/retargeting/mujoco_pose_ik.py \
  dex_teleop/retargeting/latest_ik_worker.py \
  dex_teleop/policies/bimanual_ik_joint_position.py \
  dex_teleop/policies/factory.py \
  dex_teleop/apps/oculus_franka.py
pytest -q
```

Expected: compilation succeeds and at least the 179-test baseline plus all new tests pass.

- [ ] **Step 2: Verify asynchronous loop timing in a deterministic test**

Add or run an app test whose fake solver blocks for 80 ms while the app runs
for at least 0.5 seconds. Parse cycle timestamps and assert at least 9 cycles
were logged and no adjacent application-cycle interval exceeds 0.075 seconds.

Run: `pytest tests/apps/test_oculus_franka_app.py -q -k slow_ik`

Expected: PASS, demonstrating that IK never blocks the 20 Hz main loop.

- [ ] **Step 3: Inspect dry-run telemetry completeness**

Run:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_right_ik_joint_position.yml \
  --mock-controller --dry-run --duration 1.0 \
  --teleop-log-file logs/ik_joint_position_dry_run.jsonl
```

Check with:

```bash
python - <<'PY'
import json
from pathlib import Path
records = [json.loads(line) for line in Path("logs/ik_joint_position_dry_run.jsonl").read_text().splitlines()]
cycles = [record for record in records if record["event"] == "cycle"]
assert cycles
assert all(set(cycle["arm_states"]) == {"right"} for cycle in cycles)
assert all("joint_positions" in cycle["commands"] for cycle in cycles)
assert all("right" in cycle["policy"] for cycle in cycles)
assert records[-1]["event"] == "shutdown"
print(f"cycles={len(cycles)} strategy={records[0]['strategy']}")
PY
```

Expected: assertions pass; strategy is `ik_joint_position` and no left key appears.

- [ ] **Step 4: Gate real-hardware work on backend and home-state checks**

Use the existing reset command before each directional trial:

```bash
python scripts/reset_franka_joints.py \
  --config dependency/deoxys_control/deoxys/config/franka-three.yml
```

Expected: `franka-three` reaches the existing home joints. Confirm only the
franka-three Deoxys backend is used for this application config. Do not launch
or connect `franka-two`.

- [ ] **Step 5: Run hold and small-axis hardware trials**

Launch:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_right_ik_joint_position.yml \
  --real-robot --real-controller --print-commands \
  --teleop-log-file logs/ik_joint_position_right_acceptance.jsonl
```

With an operator ready at the stop control, perform in order from the same
home pose: two-second grip-released hold; +/-1 cm x/y/z; +/-5 degree
roll/pitch/yaw; then a short continuous 6DoF motion within 0.10 m and 0.50 rad.
Release grip between trials so each trial gets a new epoch and anchor.

Expected: no left-arm connection; hold drift <=0.01 rad/joint; translation
error <=0.005 m; rotation error <=3 degrees; no emitted command violates
configured joint velocity or acceleration limits.

- [ ] **Step 6: Verify fault hold and log metrics**

Using test doubles first, then only safe operator-controlled real conditions,
verify stale controller state, a 200 ms solver delay that ages beyond 0.15 s,
and an unreachable target. Each must enter `FAULT_HOLD`, keep sending the
latest measured hold q, ignore late old-epoch results, and resume only after
release plus a new press.

Parse the acceptance JSONL and report: cycle-rate median/p95, IK solve-time
median/p95, result-age median/p95, overwrite count, accepted/rejected result
counts, maximum commanded joint velocity/acceleration, hold drift, and each
translation/rotation endpoint error.

- [ ] **Step 7: Commit only verification fixes, then perform the completion review**

If verification required code changes, rerun the focused failing test and the
full suite, then stage only those fix hunks:

```bash
git add -p dex_teleop tests configs
git diff --cached --check
git diff --cached --name-status
git commit -m "fix: harden IK joint position teleop"
```

If no fix was required, create no empty commit. Confirm generated `logs/*.jsonl`
remain untracked/ignored. Before claiming completion, invoke
`superpowers:verification-before-completion`, inspect `git status --short`,
and report any pre-existing or deferred unstaged changes separately from the
feature verification result.

---

## Execution Notes

- Tasks 1, 2, 4, and 5 create isolated modules and are straightforward review gates.
- Task 3 and Task 7 overlap files that already have uncommitted user changes; cached-diff inspection is mandatory and takes precedence over the task commit examples.
- Do not run the hardware steps without a reachable manually started `franka-three` backend and an operator able to stop the robot.
- Hardware acceptance is evidence collection, not permission to change safety limits automatically. Any tuning beyond the exact initial values requires a separate reviewed change.
