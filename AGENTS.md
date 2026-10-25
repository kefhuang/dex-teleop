# AGENTS.md

## Development environment

Use the existing conda environment for this repository:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
```

Environment details:

- Name: `dex-teleop`
- Python: `3.10`
- Created with: `conda create -n dex-teleop python=3.10 pip`
- Location: `/home/kefhuang/Apps/miniconda3/envs/dex-teleop`
- Test runner: `pytest 9.0.2` installed inside this environment
- Java: OpenJDK `11.0.27` installed inside this environment for the Oculus APK
  Gradle 6.1.1 build

`PYTHONNOUSERSITE=1` is intentional: this machine has user-site Python packages that can otherwise shadow packages inside the conda environment, including `pip`.

Do not recreate the environment unless explicitly requested. Install any additional project dependencies into `dex-teleop` and document important changes here.


## Target architecture

The approved architecture is documented in
`docs/superpowers/specs/2026-07-10-robot-arm-end-effector-api-design.md`.
Preserve these boundaries when changing the project:

- Model a robot as `Robot = Arm + EndEffector`; access control through
  `robot.arm` and `robot.end_effector` rather than proxy methods on `Robot`.
- `Arm` exposes a robot-specific fixed `home` action plus direct
  `joint_position`, `joint_velocity`, `joint_impedance`, `ik_joint_position`,
  and `ik_joint_velocity` methods. Public methods own input validation,
  robot-specific targets/IK, joint limits, and command shaping.
- `ik_joint_position` accepts an absolute or delta pose in the robot base
  frame. `ik_joint_velocity` accepts a base-frame end-effector twist.
- Keep robot-specific FK, Jacobian, IK, limits, and backend translation with
  the corresponding robot implementation. Private helpers such as
  `_solve_ik` may be tested directly but are not part of the public protocol.
- Keep clutch anchors, controller-to-base coordinate mapping, and teleoperation
  target generation outside robot implementations in a lightweight `teleop`
  layer. These are task semantics, not robot kinematics.
- Do not reintroduce `MotionCommand`, `CartesianVelocityCommand`,
  `TeleopCommandBatch`, or a public standalone `JointSetpointLimiter` unless a
  new design explicitly justifies them. Applications should invoke robot and
  end-effector APIs directly.
- Deoxys is the Franka execution backend for joint position, joint velocity,
  and joint impedance, including real-time watchdog and final safety handling;
  IK remains in the robot implementation in `dex-teleop`.
- Express position in metres, rotation in radians, linear velocity in metres
  per second, angular and joint velocity in radians per second, and
  quaternions in XYZW order.
- Require explicit positive `dt` for rate-shaped arm commands. Hardware tests
  must be marked `hardware` and excluded from the default pytest run.


## Implementation and maintenance guide

### Current package boundaries

- `dex_teleop/core/types.py` contains value/state types only: `Pose`, `Twist`,
  `ArmState`, `EndEffectorState`, and controller/device states. Do not put
  executable robot commands or backend payloads in `core`.
- `dex_teleop/robots/arm.py` and `gripper.py` define the public protocols.
  Keep these protocols small and independent of Deoxys, Oculus, and task
  semantics.
- `dex_teleop/robots/robot.py` owns aggregate lifecycle. `Robot.connect()`,
  `stop()`, and `close()` must attempt safe component cleanup; shared resources
  are closed exactly once.
- `dex_teleop/robots/franka.py` owns all Franka-specific behavior, including
  the Pinocchio Panda model, FK, Jacobian, seed-relative pose and differential IK, Panda limits,
  command shaping, state freshness checks, Deoxys conversion, and the shared
  arm/gripper connection.
- `dex_teleop/robots/mock.py` implements the same direct public API. Extend it
  when adding application behavior so tests never need real hardware.
- `dex_teleop/teleop/calibration.py` owns each arm's independent tracking-frame
  rotation calibration. `teleop/clutch.py` owns clutch anchors and produces
  base-frame target poses and twists from that rotation. Neither layer may
  solve IK, enforce robot limits, or send backend commands.
- `dex_teleop/apps/oculus_franka.py` owns device/robot construction, controller
  side bindings, the control loop, direct API calls, and shutdown coordination.
  Avoid rebuilding a policy or command-batch abstraction in the app.

Historical design and plan documents may describe the removed controller,
policy, command-wrapper, or standalone limiter architecture. They are retained
as project history, not as implementation guidance. The 2026-07-10
Robot/Arm/EndEffector design named above and this file take precedence.

### Public API invariants

- `Pose.position` is a finite shape-`(3,)` vector in metres.
  `Pose.quaternion_xyzw` is a finite, normalized shape-`(4,)` quaternion.
  Do not add ambiguous names such as `orientation` or accept WXYZ ordering.
- `Twist.linear` and `Twist.angular` are base-frame shape-`(3,)` vectors in
  metres/second and radians/second.
- `ArmState.joint_positions` and `joint_velocities` must have exactly `dof`
  elements. Its end-effector pose is in the robot base frame and its timestamp
  represents when the backend state was received, not when it was read again.
- Successful control methods return `None`. Invalid caller input raises
  `ValueError`/`TypeError`; connection, state, IK, and backend failures use the
  explicit exceptions in `dex_teleop/robots/base.py`.
- Validate command values and `dt` before reading hardware state. Clamp the
  effective `dt` to the robot-configured maximum for rate calculations, but
  reject non-finite or non-positive values.
- Every public arm path follows the same safety order: validate input, obtain a
  finite fresh state, solve IK if needed, reject bad/discontinuous solutions,
  apply joint-position limits, apply velocity/acceleration shaping, predict
  limit crossings, send to the backend, and update the last safe command.
- `home(dt=...)` advances toward a fixed joint target owned by the concrete arm
  implementation through the same validated, rate-shaped joint-position path.
  Controller adapters may emit the device-independent `ControllerAction.HOME`
  intent; the Oculus binding is left X and right A.
- Controller adapters may emit `ControllerAction.CALIBRATE`; Oculus left Y
  calibrates only the left controller/robot binding and right B calibrates only
  the right binding. Each arm keeps a separate VR-to-robot rotation.
- `ik_joint_position(delta=True)` composes a base-frame translation and
  rotation increment with the current measured end-effector pose. Local/tool
  frame deltas are unsupported unless a future API names that frame explicitly.
- Absolute and delta pose requests are final targets, not a requirement to
  solve the entire Cartesian error in one cycle. Franka advances through a
  bounded base-frame Cartesian subtarget (currently at most 1 cm and 0.05 rad
  per call), then applies IK and joint-rate shaping. Preserve this layered
  behavior so a controller can request a 5 cm move while the arm safely takes
  five or more local steps toward it.
- `ik_joint_velocity` uses a damped robot-specific Jacobian and accepts only a
  base-frame twist. It ultimately executes through joint velocity control.
- Franka pose IK starts from every cycle's measured joints, applies damped
  Jacobian updates, regularizes toward that seed, bounds every iteration and
  total seed displacement, and rejects residual or branch discontinuities.
  Do not replace it with an unconstrained one-shot IK call.
- Joint impedance gains belong in robot/backend configuration; do not add
  stiffness or damping parameters to each public command call.
- Gripper normalized position is strictly `0.0 = closed`, `1.0 = open`.
  Reject values outside this range rather than silently changing intent.
- Oculus trigger is operator intent in the opposite direction (`0 = released`,
  `1 = pressed`), so the app converts it to `normalized = 1 - trigger`.
  Franka Hand commands are hysteretic (`<=0.4` closes, `>=0.6` opens) and are
  sent only when the discrete state changes; do not stream repeated grasp calls
  every control cycle.

### Franka and Deoxys rules

- `FrankaConnection` is the sole owner of a Deoxys interface shared by
  `FrankaArm` and `FrankaGripper`. Factories should return a composed `Robot`
  through `make_franka_robot()` rather than exposing independently owned
  adapters.
- When a configured robot has a gripper, start both `auto_arm.sh` and
  `auto_gripper.sh` (normally via `auto_robot.sh`) and construct
  `FrankaInterface` with gripper support. Starting only the arm backend cannot
  execute trigger/gripper commands.
- The gripper backend marks a command executable only after successfully
  parsing a new ZMQ message and clears that flag after one execution. An empty
  nonblocking receive must never replay the previous grasp command.
- Load and send the matching Deoxys controller for each direct method:
  `JOINT_POSITION`, `JOINT_VELOCITY`, or `JOINT_IMPEDANCE`. Do not translate IK
  requests to OSC commands; IK must produce a safe joint command in this repo.
- On idle, failure, and shutdown, use the backend's safe no-control/watchdog
  behavior. Do not send a zero OSC pose as a stop command because OSC nullspace
  behavior can still move the arm.
- Treat transitions between `NO_CONTROL` and a joint controller as explicit
  state-machine edges. Arm a position/impedance controller first with the
  current measured joint position (or a velocity controller with measured
  velocity), then begin rate-shaped motion on later cycles.
- For bimanual startup, prepare each interface's initial `JOINT_POSITION`
  controller during connection, before either arm is armed. Deoxys preparation
  sends blocking `NO_CONTROL` traffic for about one second; arming one arm
  before preparing the other starves the first arm's 0.2-second watchdog.
- Franka idle uses a repeated, immutable measured joint-position hold so the
  Deoxys watchdog stays fed and state publication does not stop between clutch
  sessions. `Robot.close()` still transitions the shared connection to
  `NO_CONTROL`. Never use a zero OSC command as an idle hold.
- Streaming `JOINT_POSITION` targets use the backend's stateful C2
  desired-state tracker at 1 kHz. `LINEAR_JOINT_POSITION` remains the legacy
  wire/config name for this streaming mode; it no longer means that the
  callback emits independent position-only linear segments. Every callback
  continues from libfranka's latest `q_d`, `dq_d`, and `ddq_d`; a new 20 Hz
  target updates the position reference without resetting desired velocity or
  acceleration and without forcing the target to be reached within one policy
  period. Dynamically infeasible updates must produce bounded tracking lag,
  not an increasingly steep catch-up segment. Do not use the smooth
  point-to-point interpolator for a target stream because resetting it every
  cycle repeatedly returns it toward zero velocity. The final 1 kHz callback
  applies libfranka position, velocity, acceleration, and jerk limiting.
  Apply this exactly once through
  `robot.control(..., limit_rate=true, cutoff_frequency=kMaxCutoffFrequency)`.
  Do not manually call `franka::limitRate` in the callback while leaving
  libfranka's default 100 Hz low-pass enabled: the hidden filtered output and
  the manual limiter then use inconsistent previous states and settle into an
  approximately 0.003 rad/s limit cycle. Application-side rate state must be
  initialized from measured `q`/`dq`, reset after stop, and never assume the
  backend interpolator has the same internal velocity as an earlier session.
  While a mode remains active, advance position shaping from the previous safe
  commanded position—not from measured `q` on every cycle. Re-basing every
  command on lagging measurements collapses a streaming target into a small
  moving offset and can make the arm appear stationary.
- Treat missing, malformed, non-finite, or stale backend state as an error; do
  not fabricate a state or continue commanding from cached data.
- Software limits complement rather than replace Deoxys hard limits,
  continuity checks, communication timeout, watchdog, and final shutdown.

### Configuration and extension rules

- Example Oculus configs use `safety.max_translation_from_anchor_m` and
  `safety.max_rotation_from_anchor_rad`; these limit teleoperation targets from
  the clutch anchor. Robot joint/rate limits remain robot configuration, not
  teleoperation mapping configuration. Do not require manual per-axis mapping
  fields; operators calibrate each controller/robot binding independently.
- To add a robot, implement the `Arm` protocol with its own DoF, kinematics,
  limits, shaping, state checks, and backend translation. Do not inherit Franka
  kinematics or introduce a generic public IK/limiter object.
- To add an end effector, implement `EndEffector` plus the smallest specialized
  capability protocol (`Gripper`, `DexterousHand`, etc.) and compose it through
  `Robot`.
- To add a controller, convert its data to `DeviceState`/`ControllerState`, then
  implement controller-to-base mapping in `teleop`; robot implementations must
  not import device packages.

### Tests and required checks

Run checks inside the repository environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1

python -m compileall -q dex_teleop scripts
pytest -q
git diff --check
```

Keep tests organized by boundary: geometry/state in `tests/core`, clutch and
mapping in `tests/teleop`, direct APIs/lifecycle/backend translation/IK/safety
in `tests/robots`, and binding/control-loop/failure shutdown in `tests/apps`.
Default tests must use injected fake backends or mock robots. Any test that can
move hardware or requires a live backend must be marked `hardware` and excluded
from the default pytest selection.

For live teleoperation diagnostics, preserve the app's `--print-commands`
option. It must show clutch state, controller position, measured end-effector
position, final mapped target, measured joints, IK joints, actually transmitted
joint-position command, raw trigger, and normalized gripper intent so
input/mapping/execution failures can be distinguished.
