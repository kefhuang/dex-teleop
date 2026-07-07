# Controller Axis Calibration Design

## Goal

Add one guided calibration script that helps identify the Oculus controller position and rotation axes used by the teleop config mapping.

## User flow

The script asks the user to calibrate the mapping from desired robot motion axes to raw controller axes:

1. Select a controller (`left` or `right`) from an existing teleop config.
2. For position, the script prompts the user to move the controller in the direction that should produce robot `+x`, `+y`, and `+z` motion.
3. For rotation, the script prompts the user to rotate the controller in the direction that should produce robot `+roll`, `+pitch`, and `+yaw` motion.
4. For every prompt, the script captures a before pose and an after pose, computes the raw controller delta, chooses the dominant raw axis and sign, and displays the result.
5. At the end, the script prints a YAML snippet containing both `mapping.position_axes` and `mapping.rotation_axes`.

## Scope

The first implementation only reads controller motion. It does not move the robot and does not overwrite config files. This keeps calibration safe and makes the result inspectable before use.

## Mapping rule

The existing config format maps each robot output axis to one signed controller input axis. The script therefore asks for desired robot axes and reports values such as:

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
```

## Components

- `scripts/calibrate_controller_axes.py`: CLI wizard and pure helper functions for dominant-axis detection.
- `tests/scripts/test_calibrate_controller_axes_script.py`: unit tests for axis inference and output formatting.

## Error handling

If the measured delta is too small, the script prints a clear error telling the user to repeat with a larger movement. If the selected controller is missing from the device state, the script raises a clear runtime error.
