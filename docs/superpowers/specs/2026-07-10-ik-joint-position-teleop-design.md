# IK Joint-Position Teleoperation Design

## Problem

The current Oculus-to-Franka application uses a Cartesian-velocity policy.
That policy must remain available, but the repository also needs a complete
6DoF teleoperation strategy that converts clutch-relative Cartesian targets to
joint targets with inverse kinematics and executes them through Deoxys
`JOINT_POSITION` control.

The strategy must be responsive even when an IK solve takes longer than one
20 Hz policy period. It must not send discontinuous IK solutions directly to
the robot, and releasing the grip must hold the arm at its current joint
position. The implementation must support both arms, while the first hardware
configuration and validation run use only the right arm (`franka-three`).

## Goals

- Track clutch-relative translation and rotation with full 6DoF targets.
- Keep `cartesian_velocity` and `ik_joint_position` selectable by config.
- Keep the 20 Hz device/control loop independent of IK solve latency.
- Produce continuous, bounded joint-position commands.
- Hold the measured joint position while the clutch is released.
- Isolate state, IK work, failures, and recovery per arm.
- Persist enough JSONL telemetry to diagnose input, IK, shaping, and backend
  behavior after a run.

## Non-Goals

- Changing the Deoxys C++ joint-position controller or its 500 Hz trajectory
  interpolation.
- Replacing the existing Cartesian-velocity policy.
- Sending commands to the left arm during the first hardware validation.
- Adding collision checking against arbitrary scene geometry. Initial safety
  uses anchor bounds, model joint limits, residual checks, and conservative
  command-rate limits.
- Automatically switching IK branches to optimize a secondary objective.

## Selected Approach

Use one latest-target asynchronous IK worker per configured arm. The 20 Hz
main loop remains the only owner of robot command transmission. Each worker
loads its own MuJoCo Panda model, consumes the newest pending target, and
publishes a candidate joint target plus diagnostics. The main loop validates
the result, shapes it with joint velocity and acceleration limits, and sends a
`JointPositionCommand` every cycle.

Long precomputed trajectories are not used because continuously changing
teleoperation targets would make queued trajectories stale. Inline IK in the
main loop is not used because one slow solve would delay controller reads,
clutch handling, logging, and command publication.

## Architecture

The data path is:

```text
Oculus state + Franka pose/q
    -> clutch-relative 6DoF target
    -> per-arm latest-target IK worker
    -> IK/FK/joint validation
    -> joint setpoint velocity/acceleration shaping
    -> 20 Hz JointPositionCommand
    -> Deoxys 500 Hz JOINT_POSITION interpolation
    -> Franka
```

The worker never holds a robot adapter and cannot send a robot command. The
main loop does not block waiting for an IK result. If no new valid result is
available, it continues publishing the current shaped target or enters a hold
state according to the safety state machine.

### Command and state types

Add `JointPositionCommand` with:

- `arm_id`;
- seven finite absolute joint positions;
- `enabled`.

Add a `joint_positions` mapping to `TeleopCommandBatch`. Exactly one motion
family may be populated for an arm in a batch.

Extend `ArmState` with optional seven-element `joint_positions`. This remains
optional for compatibility with existing tests and non-joint consumers, but
the IK policy treats a missing value as unavailable state. `FrankaArmController`
populates it from the current Deoxys joint state. Mocks expose a configurable
joint state.

Extend `ArmController` and its Franka/mock implementations with
`send_joint_position`. The Franka adapter requires
`controller_type == "JOINT_POSITION"` and sends seven absolute joint values
plus the existing gripper sentinel expected by Deoxys.

### Policy selection

The application constructs a policy through a small factory:

- missing `app.strategy` preserves the current `cartesian_velocity` default;
- `app.strategy: cartesian_velocity` selects the current policy;
- `app.strategy: ik_joint_position` selects the new policy and conditionally
  validates the `ik_joint_position` config block.

Existing configs continue to run without modification. Add a new right-only
config whose only arm is `right`, whose strategy is `ik_joint_position`, and
whose Franka controller type is `JOINT_POSITION`. The policy itself iterates
over every arm in `cfg["arms"]`, so a later two-arm config needs no code change.

## 6DoF Target Construction

When the configured clutch button transitions from released to pressed, save
per-arm snapshots of:

- controller anchor position and orientation;
- measured robot end-effector anchor position and orientation;
- measured seven-joint state.

For controller anchor position `p_C0`, current controller position `p_C`, and
robot anchor position `p_R0`, construct:

```text
p_target = p_R0 + map_position(p_C - p_C0) * position_scale
```

For rotation, compute the controller quaternion relative to its anchor and
convert it to a rotation vector. Apply the configured roll/pitch/yaw axis
permutation and signs to the rotation-vector components, then apply
`rotation_scale`. If `r_mapped` is that scaled vector and `R_R0` is the robot
anchor orientation, construct the base-frame relative target:

```text
R_target = Exp(r_mapped) * R_R0
```

This avoids Euler-angle accumulation and singularities. Left multiplication
matches the base-frame relative-rotation convention used by the current
Deoxys OSC delta-pose implementation.

Clamp the translation vector from the anchor by norm and clamp the rotation
vector from the anchor by angle. Clamping preserves direction and records a
diagnostic; it does not enter a fault state.

## Relative-Pose IK

The existing benchmark validated delta-position IK from a real Franka joint
state. The teleoperation solver generalizes that pattern to a relative 6DoF
target so it does not assume that the MuJoCo model and real robot have
identical absolute end-effector frames.

For the measured robot pose `(p_R, R_R)`, requested robot pose
`(p_target, R_target)`, and MuJoCo forward kinematics at the measured seed
joint state `(p_M, R_M)`, calculate:

```text
delta_p = p_target - p_R
delta_R = R_target * transpose(R_R)
p_M_target = p_M + delta_p
R_M_target = delta_R * R_M
```

Solve the MuJoCo model from the measured seven-joint seed toward
`(p_M_target, R_M_target)`. Re-run model forward kinematics on the result and
return position and rotation residuals with the candidate joint target. The
solver object has no policy, device, or robot-adapter dependencies and can be
tested independently.

## Asynchronous Worker Semantics

Each arm owns a worker with:

- a condition-protected pending-request slot;
- a nonblocking latest-result slot;
- a dedicated solver instance and thread;
- an explicit close signal and bounded thread join.

Submitting a request replaces any pending request that has not started. A
solve already in progress is allowed to finish because the MuJoCo solve is not
interruptible, but its result still passes freshness and epoch validation.

Use two identifiers with different meanings:

- `epoch` changes on clutch release, re-anchoring, entry into fault hold, or
  arm-state invalidation. Results from another epoch are always discarded.
- `request_id` increases for each target submission. A result may be accepted
  even if a newer request is pending, provided its epoch matches, its age is
  within the configured limit, and it is newer than the last accepted result.

The separate identifiers prevent starvation when a valid solve takes slightly
longer than the 50 ms policy period while still guaranteeing that results from
an old clutch session can never execute. The worker records how many pending
requests were overwritten.

Initialize all selected-strategy workers before the application starts its
command loop. A worker/solver initialization failure aborts startup before any
joint-position command is sent. Shutdown signals and joins workers before
closing robot adapters.

## Result Validation and Joint Setpoint Shaping

The main loop accepts an IK result only if all of the following are true:

- its epoch equals the arm's current epoch;
- its request age does not exceed `max_result_age_s`;
- its request ID is newer than the last accepted result;
- all seven joint values and residuals are finite;
- FK position and rotation residuals are within configured bounds;
- every joint remains inside the MuJoCo limit with the configured margin;
- the candidate differs from the previous accepted target by no more than
  `max_target_joint_jump_rad` per component. The first candidate is compared
  with its measured seed.

An accepted candidate becomes `q_ik_target`; it is not sent directly. A
per-arm `JointSetpointLimiter` owns `q_cmd` and `dq_cmd`. Each 20 Hz update
moves them toward `q_ik_target` while enforcing per-component joint velocity
and acceleration limits using the measured loop `dt`. Invalid or excessively
large `dt` is capped consistently with the existing loop safety pattern.

The resulting finite, in-limit `q_cmd` is sent every policy cycle. Deoxys then
uses its configured `SMOOTH_JOINT_POSITION` interpolator at 500 Hz. The Python
shaper prevents target discontinuities; the Deoxys interpolator provides the
higher-rate motion samples.

## Safety State Machine

Each arm has one of four states:

### `WAITING_FOR_STATE`

No valid measured pose and seven-joint state has been received. Do not submit
IK requests or send a joint-position command.

### `HOLDING`

The clutch is released. Invalidate the previous epoch, cancel its logical
results, snapshot the latest measured joints as `q_hold`, reset the setpoint
limiter velocity to zero, and publish `q_hold` every cycle. If the state feed
becomes stale after a valid hold was established, continue publishing the
last known hold target rather than creating a new target from stale data.

### `TRACKING`

The clutch has been pressed with a valid pose and joint state, anchors have
been established, and requests may be submitted. Until the first valid IK
result returns, continue holding the anchor joint position. Accept results and
shape commands according to the rules above.

### `FAULT_HOLD`

Freeze at the most recent measured joint state, invalidate the epoch, and
reset command velocity to zero. Enter this state when:

- device or robot state becomes stale while tracking;
- IK returns a non-finite, out-of-limit, discontinuous, or high-residual
  result;
- no acceptable result is available for longer than `max_result_age_s` while
  the latest target differs from the current accepted target by more than the
  existing `safety.position_deadzone` or `safety.rotation_deadzone`. Before
  the first accepted result, compare the latest target with the clutch anchor;
- the worker reports an unexpected runtime exception;
- command shaping produces an invalid value.

Recovery requires the user to release the clutch and press it again. Holding
the clutch does not automatically resume after a fault. A result from an old
epoch, older than the last accepted request ID, or over the age limit is
discarded without creating a new fault by itself; the no-fresh-result timeout
above decides whether tracking subsequently enters `FAULT_HOLD`.

## Initial Configuration

The right-arm hardware config uses these conservative initial values:

```yaml
app:
  rate_hz: 20
  strategy: ik_joint_position

arms:
  right:
    controller_type: JOINT_POSITION

ik_joint_position:
  max_translation_from_anchor_m: 0.10
  max_rotation_from_anchor_rad: 0.50
  max_result_age_s: 0.15
  max_position_residual_m: 0.005
  max_rotation_residual_rad: 0.05
  max_target_joint_jump_rad: 0.20
  max_joint_velocity_rad_s: 0.50
  max_joint_acceleration_rad_s2: 1.00
  joint_limit_margin_rad: 0.05
```

Scalar joint velocity and acceleration limits apply to all seven joints in
the initial configuration. The parser may also accept a seven-element list so
future tuning does not require a schema change. Values must be finite and
strictly positive, except that no zero-valued motion limit is needed for this
strategy.

The config reuses the existing `mapping`, `scales`, `controls`, input, arm,
gripper, and safety sections. The new strategy continues to emit gripper
commands from the configured controller axis.

## Telemetry

Extend each JSONL cycle record without removing existing fields. For each IK
joint-position arm, record:

- safety state, epoch, latest submitted/accepted request IDs, and hold reason;
- measured pose and joints;
- controller and robot anchor poses;
- desired robot end-effector pose and whether translation/rotation was
  clamped;
- IK seed, candidate target, solve duration, result age, FK position residual,
  FK rotation residual, and rejection reason;
- current accepted `q_ik_target`, emitted `q_cmd`, and `dq_cmd`;
- worker pending-request overwrite count and worker error, if any;
- whether the joint-position command was sent.

Start and shutdown records identify the selected strategy. An error or safety
transition is also present in the following cycle record so the complete
context remains on one timeline with Oculus and robot state.

## Testing

### Unit tests

1. Validate `JointPositionCommand` shape, finiteness, disabled construction,
   and batch placement.
2. Validate quaternion delta, rotation-vector mapping, base-frame composition,
   anchor snapshotting, and direction-preserving target clamps.
3. Validate `JointSetpointLimiter` velocity, acceleration, joint-limit, zero
   velocity reset, and irregular-`dt` behavior.
4. Use a controllable fake solver to validate worker latest-only replacement,
   request ID ordering, epoch invalidation, old-result rejection, exception
   reporting, and clean shutdown.
5. Validate every safety-state transition, the release/re-press recovery gate,
   per-arm isolation, and gripper output.
6. Validate Franka/mock adapter action construction and the required
   `JOINT_POSITION` controller type.
7. Validate strategy selection, conditional config validation, legacy config
   compatibility, right-only dry-run behavior, JSONL diagnostics, and worker
   cleanup on normal and exceptional exit.

### MuJoCo integration tests

Use the real Panda model in the existing `dex-teleop` conda environment to
verify:

- FK round trips at representative joint states;
- reachable translation-plus-rotation targets converge within configured
  residuals;
- nearby sequential targets stay on a continuous joint branch;
- targets outside limits or reachability are rejected;
- translation and rotation residual calculations match independently
  computed values.

### Hardware validation

Use only `franka-three` and the right controller/arm config:

1. Hold for two seconds with the clutch released.
2. Command separate `+/-x`, `+/-y`, and `+/-z` 1 cm motions from the same home
   state.
3. Command separate positive and negative roll, pitch, and yaw rotations of
   five degrees from the same home state.
4. Run a short continuous 6DoF teleoperation session within the configured
   10 cm and 0.5 rad anchor bounds.
5. Inject stale state, slow IK, and unreachable-target conditions and verify
   fault hold plus release/re-press recovery.

The left arm is not connected or commanded in this validation.

## Acceptance Criteria

- The main application loop sustains at least 18 Hz and never waits for IK.
- Every emitted joint command is finite, within joint margins, and satisfies
  configured velocity and acceleration limits.
- Accepted IK results have model FK residuals at or below 5 mm and 0.05 rad.
- A result older than 0.15 seconds or from another epoch is never executed.
- Clutch release begins joint-position holding on the next application cycle.
- Fault recovery requires a release and a new press.
- Hardware hold drift over two seconds is no more than 0.01 rad per joint.
- Each 1 cm hardware translation finishes within 5 mm of its target.
- Each five-degree hardware rotation finishes within three degrees of its
  target.
- No left-arm connection or command occurs during the initial hardware run.

## Implementation Boundaries

Keep the policy, IK solver/worker, and joint setpoint limiter in separate
modules with injected dependencies for tests. The application owns policy and
worker lifecycle; the Franka adapter owns only Deoxys command translation.
Avoid unrelated changes to the Cartesian-velocity policy or the Deoxys C++
backend while implementing this strategy.
