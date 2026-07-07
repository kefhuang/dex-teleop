# Controller Axis Calibration Design

## Goal

Add one guided calibration script that identifies Oculus controller position and rotation axes for the teleop config mapping by using the robot itself as the visual reference.

## User flow

The script asks the user to choose one configured arm and one controller:

```bash
python scripts/calibrate_controller_axes.py \
  --config configs/oculus_franka_bimanual.yml \
  --arm left \
  --controller left \
  --real-robot \
  --real-controller
```

The robot moves first. The user then moves or rotates the controller to match the robot's just-observed direction. This avoids asking the user to infer robot axes from text alone.

## Position calibration

For each positive robot translation axis:

1. Prompt the user to press Enter.
2. Move the selected robot arm by `--position-step` along robot `+x`, `+y`, or `+z`.
3. Ask the user to mimic that visible robot motion with the selected controller.
4. Capture controller start and end poses and compute raw controller XYZ delta.
5. Return the robot by sending the opposite delta.

The script uses the three positive robot directions to infer `mapping.position_axes`. The negative robot movement is used to return to the starting pose and keep the calibration bounded.

## Rotation calibration

For each positive robot RPY rotation axis:

1. Prompt the user to press Enter.
2. Rotate the selected robot arm by `--rotation-step` around robot `+roll`, `+pitch`, or `+yaw`.
3. Ask the user to mimic that visible robot rotation with the controller.
4. Capture controller start and end orientations and compute raw controller RPY delta.
5. Return the robot by sending the opposite rotation delta.

The script uses the three positive robot rotations to infer `mapping.rotation_axes`.

## Safety defaults

The defaults are intentionally small:

- `--position-step 0.02` meters
- `--rotation-step 0.10` radians
- `--hold-time 0.8` seconds

The script sends one `MotionCommand` for the reference move, compensating for Deoxys `action_scale` so `--position-step` and `--rotation-step` are interpreted as robot-frame meters/radians. It then stops the robot, captures controller motion, and sends the opposite `MotionCommand` to return. If controller capture fails after a reference move, the return command still runs in a `finally` block.

## Scope

The script prints a suggested YAML snippet and does not overwrite config files. It can run with `--dry-run --mock-controller` to inspect the flow without hardware.

## Output

The output remains compatible with existing config format:

```yaml
mapping:
  position_axes:
    x: "+z"
    y: "-x"
    z: "+y"
  rotation_axes:
    roll: "+pitch"
    pitch: "-roll"
    yaw: "+yaw"
```
