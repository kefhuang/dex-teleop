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
- left/right trigger: continuous corresponding gripper command,
- Ctrl-C: graceful shutdown.

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
