# Oculus Franka Bimanual Teleoperation Design

Date: 2026-07-05

## Goal

Build the first runnable teleoperation path for this repository:

- Use Oculus/Quest left and right controllers to control two Franka arms.
- Use each controller trigger as continuous control for the corresponding Franka gripper.
- Preserve clear extension points for future robots, grippers, dexterous hands, and input devices.

The first version targets dual Franka arms with Franka grippers. It does not include SenseGlove, Inspire Hand, Dobot, or third-party grippers yet.

## Design approach

Use an interface-first modular framework rather than a one-off script or a plugin-heavy architecture.

The framework defines stable boundaries between:

- input devices,
- teleoperation policy,
- robot arm controllers,
- gripper controllers,
- application wiring and control loop.

This keeps the first implementation small while making future replacements possible. For example, adding Dobot should require a new `ArmController`; adding another gripper should require a new `GripperController`; adding SenseGlove/Inspire Hand should add hand-specific interfaces and policies without changing the Oculus-to-Franka arm policy.

## Module layout

```text
dex_teleop/
  apps/
    oculus_franka.py          # First runnable app entry point

  core/
    types.py                  # Pose, ControllerState, MotionCommand, GripperCommand, etc.
    config.py                 # YAML config loading and validation
    loop.py                   # Rate control and graceful shutdown helpers

  devices/
    base.py                   # InputDevice protocol/interface
    oculus.py                 # OculusReader-backed input implementation
    mock.py                   # Mock controller input for no-hardware tests

  robots/
    arm.py                    # ArmController protocol/interface
    gripper.py                # GripperController protocol/interface
    franka.py                 # Franka arm and gripper adapters
    mock.py                   # Mock robot/gripper output for dry-run

  policies/
    bimanual_clutch.py        # Oculus controller state -> arm/gripper commands

  utils/
    transforms.py             # Axis mapping, pose delta, deadzone, scaling, limits

configs/
  oculus_franka_bimanual.yml  # First example config
```

## Runtime data flow

```text
OculusDevice or MockControllerInput
        ↓
BimanualClutchPolicy
        ↓
left MotionCommand + left GripperCommand
right MotionCommand + right GripperCommand
        ↓
FrankaArmController / FrankaGripperController
or MockArmController / MockGripperController
```

The app layer wires these pieces together from YAML and CLI options.

## Core data structures

The first version should define a small set of typed data structures.

```python
@dataclass
class Pose:
    position: np.ndarray      # shape=(3,)
    orientation: np.ndarray   # shape=(4,), with convention documented in code

@dataclass
class ControllerState:
    name: str                 # "left" or "right"
    pose: Pose
    buttons: dict[str, bool]
    axes: dict[str, float]    # e.g. trigger

@dataclass
class DeviceState:
    timestamp: float
    controllers: dict[str, ControllerState]

@dataclass
class MotionCommand:
    arm_id: str
    delta_position: np.ndarray
    delta_rotation: np.ndarray
    enabled: bool

@dataclass
class GripperCommand:
    gripper_id: str
    normalized: float         # 0.0=open, 1.0=close
```

`MotionCommand` is intentionally expressed as relative motion for the first version. This matches the clutch-based interaction and avoids requiring a full workspace calibration workflow.

## Interfaces

```python
class InputDevice(Protocol):
    def start(self) -> None: ...
    def read(self) -> DeviceState: ...
    def stop(self) -> None: ...

class ArmController(Protocol):
    def connect(self) -> None: ...
    def get_state(self) -> ArmState: ...
    def send_motion(self, command: MotionCommand) -> None: ...
    def stop_motion(self) -> None: ...
    def close(self) -> None: ...

class GripperController(Protocol):
    def connect(self) -> None: ...
    def send_gripper(self, command: GripperCommand) -> None: ...
    def close(self) -> None: ...
```

The public interfaces keep arm and gripper control separate. The Franka adapter may share one underlying Deoxys `FrankaInterface` internally if that is the cleanest way to interact with Deoxys.

## Franka adapter behavior

`FrankaArmController` maps `MotionCommand` to Deoxys `OSC_POSE` action. If the shared Deoxys interface expects a gripper dimension in the same action vector, the arm adapter fills that slot with a neutral value while gripper intent remains represented by `GripperCommand`.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper_slot]
```

`FrankaGripperController` maps `GripperCommand.normalized` to Franka gripper behavior:

- `0.0` means open.
- `1.0` means close/grasp.
- intermediate values represent continuous trigger intent and may be converted to width or the closest Deoxys-supported command.

The adapter should hide Deoxys-specific details from the policy and app layers.

## Bimanual clutch policy

Each arm has an independent controller binding and clutch state.

Controller semantics:

- `grip` button: clutch/deadman control for arm motion.
- `trigger` axis: continuous gripper control.

State machine for each arm:

```text
IDLE
  grip pressed →
    record controller pose as controller_anchor
    record robot state/pose if needed
    enter ACTIVE

ACTIVE
  grip held →
    controller_delta = current_controller_pose - controller_anchor
    mapped_delta = configured axis mapping
    scaled_delta = configured scale
    limited_delta = deadzone and per-cycle limits
    emit MotionCommand(enabled=True, delta=limited_delta)

  grip released →
    emit disabled/stop motion command
    clear anchors
    enter IDLE
```

The first version should not implement absolute workspace mapping. It uses local relative movement under clutch control.

## Controller-to-arm binding

Bindings are explicit in config, not hardcoded.

```yaml
arms:
  left:
    type: franka
    interface_cfg: ~/Robots/franka/deoxys_control/deoxys/config/franka-two.yml
    controller: left
    gripper: left_gripper
  right:
    type: franka
    interface_cfg: ~/Robots/franka/deoxys_control/deoxys/config/franka-three.yml
    controller: right
    gripper: right_gripper
```

This supports unusual lab setups where the physical robot named `left` may need a different controller binding.

## Axis mapping and scaling

Oculus controller coordinates are mapped through YAML, so field calibration can be done by editing config rather than code.

```yaml
mapping:
  position_axes:
    x: "+z"
    y: "-x"
    z: "+y"
  rotation_axes:
    roll: "+roll"
    pitch: "+pitch"
    yaw: "+yaw"

scales:
  position: 0.5
  rotation: 0.3
```

The implementation should support signed axis names such as `+x`, `-x`, `+z`, and `-z` for position, and signed roll/pitch/yaw mappings for rotation.

## Configuration and CLI

Main command:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml
```

Useful CLI options:

```bash
--dry-run                 # Use mock robot/gripper outputs; do not command hardware
--mock-controller         # Use simulated Oculus/controller input
--duration 10             # Optional finite run for smoke tests
--log-level INFO
```

CLI options override YAML where applicable.

Example config:

```yaml
app:
  rate_hz: 30
  dry_run: true

input:
  type: oculus
  mock_controller: true
  ip: null
  port: 5555

controls:
  clutch_button: grip
  gripper_axis: trigger

arms:
  left:
    type: franka
    interface_cfg: ~/Robots/franka/deoxys_control/deoxys/config/franka-two.yml
    controller: left
    gripper: left_gripper
  right:
    type: franka
    interface_cfg: ~/Robots/franka/deoxys_control/deoxys/config/franka-three.yml
    controller: right
    gripper: right_gripper

grippers:
  left_gripper:
    type: franka
    arm: left
    open_width: 0.08
    close_width: 0.0
  right_gripper:
    type: franka
    arm: right
    open_width: 0.08
    close_width: 0.0

mapping:
  position_axes:
    x: "+z"
    y: "-x"
    z: "+y"
  rotation_axes:
    roll: "+roll"
    pitch: "+pitch"
    yaw: "+yaw"

scales:
  position: 0.5
  rotation: 0.3

safety:
  max_translation_per_cycle: 0.01
  max_rotation_per_cycle: 0.05
  position_deadzone: 0.002
  rotation_deadzone: 0.01
  send_zero_when_idle: true
```

## Mock and dry-run modes

Two separate concepts are required:

- `mock_controller`: do not connect to the real Oculus reader; generate deterministic left/right controller states.
- `dry_run`: do not command real robots or grippers; use mock robot and gripper controllers.

This distinction keeps input testing and robot-output safety independent.

Dry-run behavior:

- no import or initialization of Deoxys `FrankaInterface`,
- motion and gripper commands are logged or stored,
- `--duration` can run a finite smoke test without hardware.

Mock controller behavior:

- no ADB/Oculus connection,
- deterministic left/right poses, grip states, and trigger values,
- useful for policy and loop tests.

## Deoxys backend boundary

The first version does not start or stop Deoxys backend processes. The user must start the backend manually before real robot operation.

Startup should check config file paths and fail with clear messages when real hardware mode cannot initialize. If `FrankaInterface` connection fails, the error should suggest confirming that the Deoxys backend has already been started.

## Safety behavior

The first version implements basic safety controls:

- Ctrl-C triggers graceful shutdown.
- On shutdown, both arms receive stop/zero/termination behavior through their controllers.
- If grip is not held, the corresponding arm does not move.
- When grip is first pressed, the policy establishes an anchor and avoids an initial jump.
- Every motion command applies deadzone, scale, and per-cycle translation/rotation limits.
- If controller data for an active arm disappears, that arm exits active mode and sends stop/disabled motion.

The first version does not include a full safety supervisor. The config should leave room for later additions such as:

- workspace bounds,
- Oculus emergency-stop button,
- robot state watchdog,
- maximum consecutive dropped input frames.

## Error handling

Startup errors:

- Missing config file or required field: fail with field path.
- Missing Franka interface config: fail before robot initialization.
- Real Oculus import/connect failure: suggest ADB, Quest app, and `--mock-controller`.
- Real Franka initialization failure: suggest checking manually started Deoxys backend.

Runtime errors:

- Controller temporarily missing: stop the bound arm and keep the app running if possible.
- Robot command failure: stop the affected arm and, for the first version, exit the app after graceful shutdown.
- KeyboardInterrupt: stop both arms, close robots/grippers, stop input device.

## Testing strategy

The first version should prioritize tests that do not require hardware.

1. Config tests
   - load the bimanual YAML,
   - report missing fields clearly,
   - verify CLI overrides.

2. Mapping tests
   - signed axis mapping such as `+z` and `-x`,
   - scale, deadzone, and limits.

3. Policy tests
   - grip not pressed emits disabled motion,
   - first grip press establishes an anchor,
   - held grip with movement emits bounded motion,
   - grip release emits disabled/stop motion,
   - trigger maps continuously to gripper normalized value.

4. Dry-run smoke test

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --mock-controller \
  --dry-run \
  --duration 2
```

5. Real hardware smoke path
   - run `scripts/test_oculus_reader.py`,
   - run `scripts/test_deoxys_control.py`,
   - then run the teleop app with real controller and real robots.

## Explicit non-goals for first version

- Do not automatically start or manage the Deoxys backend.
- Do not build a workspace calibration UI.
- Do not build a plugin registry.
- Do not support SenseGlove or Inspire Hand yet.
- Do not support Dobot yet.
- Do not implement a full emergency-stop or workspace safety supervisor.
- Do not implement teleoperation recording/replay.

## First-version deliverables

- Core data types and config loading.
- Oculus input adapter and mock controller input.
- Franka arm/gripper adapters and mock robot/gripper outputs.
- Bimanual clutch policy.
- `oculus_franka` app entry point.
- Example bimanual config.
- Unit tests for config, mapping, policy, and dry-run smoke behavior.
