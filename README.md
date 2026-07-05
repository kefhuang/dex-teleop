# dex-teleop

Dexterous teleoperation framework for VR/controller input, glove-based hand control, robot hand retargeting, and Franka control.

## Initial dependencies

- `dependency/deoxys_control`: Franka control via Deoxys
- `dependency/oculus_reader`: Oculus/Quest controller pose and button reader

## Planned structure

```text
dex_teleop/
  devices/      # Oculus, SenseGlove, and other input devices
  robots/       # Franka and other robot interfaces
  retargeting/  # Hand/control retargeting utilities
dependency/     # Git submodule dependencies
```
