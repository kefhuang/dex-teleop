# dex-teleop

Dexterous teleoperation framework for VR/controller input, glove-based hand control, robot hand retargeting, and Franka control.

## Initial dependencies

- `dependency/deoxys_control`: Franka control via Deoxys
- `dependency/oculus_reader`: Oculus/Quest controller pose and button reader

## Architecture

```text
dex_teleop/
  devices/      # Oculus, SenseGlove, and other input devices
  teleop/       # Per-arm frame calibration, clutch anchors, and target mapping
  robots/       # Robot = Arm + EndEffector; kinematics and backends
  visualization/ # Optional live views of controller/application state
dependency/     # Git submodule dependencies
```

Applications command robots through `robot.arm` and `robot.end_effector`.
Arm commands are direct fixed-home, joint, or IK calls; there are no command
batches or Cartesian-command wrapper objects.

## Oculus + dual Franka teleop

Safe no-hardware smoke test:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1

python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --mock-controller \
  --dry-run \
  --duration 2
```

Real hardware flow:

1. Start the Deoxys backend manually for both Franka arms.
2. Verify Oculus input:

   ```bash
   python scripts/test_oculus_reader.py
   ```

3. Verify Franka state connectivity:

   ```bash
   python scripts/test_deoxys_control.py
   ```

4. Run teleop with real controller and robots:

   ```bash
   python -m dex_teleop.apps.oculus_franka \
     --config configs/oculus_franka_bimanual.yml \
     --real-controller \
     --real-robot
   ```

Controls:

- left/right grip: clutch for corresponding arm motion,
- left X/right A: trigger the corresponding arm's latched `HOME` action,
- left Y: calibrate only the left controller to the left robot base,
- right B: calibrate only the right controller to the right robot base,
- left/right trigger: continuous corresponding gripper command,
- Ctrl-C: graceful shutdown.

Before engaging either clutch, hold that controller so the visible green ray
points horizontally along the corresponding robot's +X (forward) direction,
release the grip, and press its calibration button once. The app uses gravity
for vertical alignment and the green ray only for horizontal heading, so roll
about the ray does not affect calibration. Each arm stores an independent
VR-to-robot rotation; calibrating one side does not change or enable the other.
Calibration is intentionally quick and in-memory, so repeat it after restarting
the app, headset tracking, or either robot setup. An uncalibrated arm remains in
hold even if its grip is pressed.

`HOME` takes precedence over clutch motion and continues after the button is
released until the robot reports that its robot-specific home joints have been
reached. The action uses the same state validation, joint limits, and
velocity/acceleration shaping as direct arm commands.

To show the raw left/right controller positions and recent trajectories in the
Oculus tracking coordinate system, add `--visualize-controllers`. The view uses
metres, can be rotated by dragging, and zoomed with the mouse wheel. For a safe
controller-only check without commanding hardware:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --real-controller \
  --dry-run \
  --visualize-controllers
```

Closing the visualization window leaves the teleoperation loop running; use
Ctrl-C in the terminal to stop the application.

To run only the Oculus input and visualization, without loading teleoperation
configuration or constructing any robot, use:

```bash
python scripts/visualize_oculus_controllers.py
```

Use `--ip <quest-ip>` for network ADB mode, or verify the visualization without
a headset using `--mock-controller`.

The example config defaults to `dry_run: true` and `mock_controller: true` for safety.

If an arm has no gripper attached, set that arm's gripper binding to `null`
and omit the matching entry from `grippers`:

```yaml
arms:
  right:
    type: franka
    interface_cfg: dependency/deoxys_control/deoxys/config/franka-three.yml
    controller: right
    gripper: null

grippers: {}
```

With this setup, Oculus grip still controls the arm clutch; trigger commands are
ignored for that arm.

## Diagnostic control traces

Use `--record-trace` while debugging to append a structured JSONL trace without
streaming large arrays through Python's text logger:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --real-controller \
  --real-robot \
  --record-trace logs/teleop-debug.jsonl \
  --print-commands
```

Each `control_cycle` line contains ordered, monotonic-timestamped events from
that app iteration. The trace includes the raw Oculus transforms/buttons and
converted controller state; clutch anchors and mapped targets; every arm state
read; pose or differential IK results; rate- and limit-shaped commands; exact
Deoxys controller actions; gripper hysteresis decisions and backend actions;
cycle timing; and exceptions with tracebacks. Startup, shutdown, configuration,
and the final dropped-record count are separate records in the same file.

Writing is handled by a background thread. If its bounded queue fills, control
continues, a warning is emitted, and the final `trace_summary` reports the
number of dropped records. Reusing a path appends a new session with a distinct
`session_id`; it does not overwrite an earlier trace.

For a quick overview of one trace:

```bash
python -c 'import json,sys; rs=[json.loads(x) for x in open(sys.argv[1])]; print([(r["record_type"], r.get("iteration"), len(r.get("events", []))) for r in rs])' logs/teleop-debug.jsonl
```
