# Robot, Arm, and End-Effector API Design

## Objective

Simplify `dex-teleop` around the physical composition `Robot = Arm + EndEffector`. Remove command-batch and Cartesian command abstractions that are no longer needed. Expose direct joint-control and IK-control methods on `Arm`, while keeping teleoperation intent such as clutching outside the robot implementation.

## Architecture

The primary aggregate is:

```python
@dataclass
class Robot:
    arm: Arm
    end_effector: EndEffector | None = None
```

Responsibilities are divided as follows:

- `apps`: create devices and robots, run the control loop, bind controller sides to robots, and coordinate shutdown.
- `teleop`: convert controller motion into robot-base-frame target poses or twists; maintain clutch anchors and coordinate mappings.
- `robots`: own robot-specific kinematics, IK, safety shaping, backend translation, state access, and hardware execution.
- `devices`: convert hardware-specific input into common controller state types.
- `core`: provide common state and geometry types, configuration, and timing utilities.
- Deoxys: execute Franka joint-position, joint-velocity, and joint-impedance commands, including final real-time safety, watchdog, and exception handling.

The intended dependency flow is:

```text
OculusDevice
  -> DeviceState
  -> app + fixed-action binding or clutch mapping
  -> HOME intent, target Pose, or Twist
  -> Robot.arm
  -> robot-specific IK and safety shaping
  -> Deoxys or another robot backend
```

## Public API

### Robot

```python
class Robot:
    arm: Arm
    end_effector: EndEffector | None

    def connect(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
```

`Robot` does not proxy arm commands. `robot.arm` and `robot.end_effector` are the only control entry points.

When the arm and end effector share a backend connection, a robot-specific internal connection object owns that resource. `Robot.close()` closes it exactly once.

### Arm

```python
class Arm(Protocol):
    @property
    def dof(self) -> int: ...
    @property
    def motion_state(self) -> ArmMotionState: ...
    @property
    def is_settled(self) -> bool: ...

    def connect(self) -> None: ...
    def get_state(self) -> ArmState: ...

    def home(self, *, dt: float) -> None: ...
    def joint_position(self, positions: np.ndarray, *, dt: float) -> None: ...
    def joint_velocity(self, velocities: np.ndarray, *, dt: float) -> None: ...
    def joint_impedance(self, positions: np.ndarray, *, dt: float) -> None: ...

    def ik_joint_position(
        self,
        pose: Pose,
        *,
        delta: bool = False,
        dt: float,
    ) -> None: ...

    def ik_joint_velocity(self, twist: Twist, *, dt: float) -> None: ...

    def stop(self) -> None: ...
```

All pose and twist values are expressed in the robot base frame:

- `ik_joint_position(..., delta=False)` accepts an absolute base-frame end-effector pose.
- `ik_joint_position(..., delta=True)` accepts a base-frame pose increment relative to the current measured end-effector pose.
- `ik_joint_velocity()` accepts a base-frame end-effector twist in metres per second and radians per second.
- End-effector-local deltas are intentionally unsupported.

`home()` is a fixed action whose joint target belongs to each concrete robot
implementation. It follows the same state validation, limit enforcement, and
rate shaping as `joint_position()`; applications do not carry robot joint
constants.

`motion_state` describes top-level intent and has only `STREAMING` and
`HOLDING`. Stopping control immediately enters `HOLDING` with one immutable
joint-position target; backend velocity, acceleration, and jerk limits still
make the physical transition continuous. `is_settled` separately reports
whether the reference, backend desired velocity, and measured arm have
converged to that target.

All public methods validate and safely shape their commands. No public raw-send bypass is provided.

### Internal Arm methods

Concrete arm implementations may use private methods such as:

```python
_solve_ik(target_pose, seed_joint_positions)
_solve_differential_ik(twist, joint_positions)
_limit_joint_position(target_positions, dt)
_limit_joint_velocity(target_velocities, dt)
_send_joint_position(positions)
_send_joint_velocity(velocities)
_send_joint_impedance(positions)
```

These methods are tested directly where useful but are not part of the stable `Arm` protocol.

### End effector

The base protocol is deliberately small:

```python
class EndEffector(Protocol):
    def connect(self) -> None: ...
    def get_state(self) -> EndEffectorState: ...
    def stop(self) -> None: ...
```

Specialized protocols add capabilities:

```python
class Gripper(EndEffector, Protocol):
    def position(self, normalized: float) -> None: ...

class DexterousHand(EndEffector, Protocol):
    def joint_position(self, positions: np.ndarray) -> None: ...
```

For grippers, normalized position uses `0.0` for closed and `1.0` for open.

Joint-impedance stiffness and damping come from robot configuration. They are not supplied on every command.

## Common data types

`core/types.py` retains state and geometry values rather than command wrappers:

```python
@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray

@dataclass(frozen=True)
class Twist:
    linear: np.ndarray
    angular: np.ndarray

@dataclass(frozen=True)
class ArmState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pose: Pose
    timestamp: float

class ControllerAction(str, Enum):
    HOME = "HOME"
```

`ControllerAction` is input intent carried in `ControllerState`, not an
executable robot command wrapper. Device adapters translate hardware bindings
into this fixed vocabulary. The Oculus adapter maps left X and right A to
`HOME`; the application latches a rising edge and calls the corresponding
`robot.arm.home()` until `robot.arm.is_settled` is true.

Units are metres, seconds, radians, metres per second, and radians per second. Quaternions use XYZW order.

The following command abstractions are removed:

- `MotionCommand`
- `CartesianVelocityCommand`
- `JointPositionCommand`
- `GripperCommand`
- `TeleopCommandBatch`

Callers invoke arm and end-effector methods directly.

## Teleoperation mapping

Clutching and controller-to-robot coordinate conversion are task intent, not robot kinematics. They live in a lightweight `dex_teleop/teleop` package rather than `policies` or a robot implementation.

```text
dex_teleop/teleop/
  clutch.py
  mapping.py
```

A clutch mapper records controller and measured robot anchors, then produces a target pose in the robot base frame. It does not solve IK, enforce robot limits, send commands, or control an end effector.

The current `policies/ik_target.py` functionality moves into this teleoperation layer because it generates a clutch-relative target; it is not an IK solver.

## Robot-specific implementation

Initial organization remains compact:

```text
dex_teleop/robots/
  base.py
  robot.py
  franka.py
  dobot.py
  mock.py
```

`franka.py` contains or internally composes:

- a shared Deoxys connection owner;
- `FrankaArm`;
- `FrankaGripper`;
- Franka FK, Jacobian, pose IK, and differential IK;
- Franka limits and command shaping;
- Deoxys action translation;
- a `make_franka_robot()` factory.

Robot-specific code may later become a package when file size justifies it. It is not split prematurely.

Franka and Dobot share the public `Arm` contract but do not share robot models or IK solutions. Each implementation owns its correct DoF, kinematics, limits, and backend behavior.

`MockArm` and `MockEndEffector` implement the same public API so application tests do not require hardware.

## Safety and command processing

Every public arm command follows this sequence:

```text
validate input and dt
  -> read and validate robot state
  -> solve IK when applicable
  -> validate solution and continuity
  -> enforce software joint limits
  -> enforce velocity and acceleration limits
  -> send to backend
  -> record the last safe command
```

`dt` is explicit and must be finite and positive. An implementation also caps it with a configured maximum before rate calculations.

The old `JointSetpointLimiter` is no longer a public standalone concept. Equivalent stateful limiting remains inside each `Arm`, where robot-specific limits and execution semantics are available.

Safety remains layered:

- `dex-teleop` validates task-level commands, IK continuity, software joint limits, rate limits, stale state, differential-IK damping, and predicted limit crossings.
- The backend provides final hard limits, command continuity, watchdog behavior, communication timeout handling, and safe shutdown.

## Errors

Successful control methods return `None`. Failures raise explicit exceptions:

```python
class RobotError(RuntimeError): ...
class RobotConnectionError(RobotError): ...
class RobotStateUnavailable(RobotError): ...
class RobotStateStale(RobotError): ...
class RobotCommandError(RobotError): ...
class IKError(RobotError): ...
class IKUnreachableError(IKError): ...
```

The application stops the affected arm or robot after stale state, unreachable IK, or backend command failures. Shutdown attempts all components even when one component raises an error.

## Testing

Tests cover:

- geometry and state validation in `tests/core`;
- Oculus conversion in `tests/devices`;
- clutch anchors, base-frame deltas, and coordinate mapping in `tests/teleop`;
- robot lifecycle, mock APIs, Franka action conversion, IK, differential IK, rate limiting, stale state, and safe stopping in `tests/robots`;
- left/right controller binding, clutch release, end-effector control, and failure shutdown in `tests/apps`.

IK private methods are directly tested for finite output, correct dimensions, limit compliance, FK target accuracy, continuity, seed behavior, and unreachable-target errors.

Verification proceeds in levels:

1. pure unit tests;
2. MuJoCo/model FK, IK, and Jacobian tests;
3. mocked Deoxys interface tests;
4. explicitly marked low-speed hardware diagnostics;
5. single-arm and then bimanual Oculus teleoperation.

Hardware tests are excluded from the default pytest suite and marked `hardware`.

## Resulting application shape

The control loop is intentionally direct:

```python
device_state = device.read()

for side, robot in robots.items():
    controller = device_state.controllers[side]

    if controller.button("grip"):
        target_pose = clutch_mappers[side].update(
            controller_pose=controller.pose,
            arm_state=robot.arm.get_state(),
        )
        robot.arm.ik_joint_position(target_pose, dt=dt)

    if robot.end_effector is not None:
        robot.end_effector.position(controller.axis("trigger"))
```

There is no command batch and no Cartesian motion command abstraction. The app expresses teleoperation intent, while each arm owns robot-specific motion realization.
