# dex-teleop

Dexterous teleoperation framework for VR/controller input, glove-based hand control, robot hand retargeting, and Franka control.

## Shared hardware installation (HP)

The checkout is `~/Projects/dex-teleop/src`; outputs belong in
`~/Projects/dex-teleop/runs`. Drivers are not submodules. Read each device's
`~/Robots/<device>/README.md` before hardware use. Tested shared driver pins:

| Device | Checkout | Commit |
| --- | --- | --- |
| Franka | `~/Robots/franka/deoxys_control` | `630309a` |
| Inspire hand | `~/Robots/inspire-hand/inspirepy` | `a823afc` |
| Quest | `~/Robots/quest/oculus_reader` | `ab97790` |
| Dobot vendor SDK | `~/Robots/dobot/TCP-IP-Python-V3` | `7fb060c` |
| SenseGlove (native only, no executable integration) | `~/Robots/senseglove/SenseGlove-API` | `31900b3` |

Create a new env only when absent; do not recreate an existing one:

```bash
~/Apps/miniconda3/bin/conda create -y -n dex-teleop -c conda-forge --override-channels python=3.10 pip
source ~/Apps/miniconda3/etc/profile.d/conda.sh && conda activate dex-teleop
export PYTHONNOUSERSITE=1
cd ~/Projects/dex-teleop/src
python -m pip install -r requirements.txt
python -m pip install -e ~/Robots/franka/deoxys_control/deoxys
python -m pip install -e ~/Robots/inspire-hand/inspirepy
python -m pip install -e ~/Robots/quest/oculus_reader
# Dobot has no setup.py: create a .pth in this env, not a copied SDK.
python -c 'import site; from pathlib import Path; (Path(site.getsitepackages()[0])/"robots-dobot.pth").write_text(str(Path.home()/"Robots/dobot/TCP-IP-Python-V3")+"\n")'
# To test legacy-wip with this env, also install its recorded IK dependency:
python -m pip install pin==4.1.0
```

`protobuf<=3.20.3` is required for the committed protoc-3.13 Python modules.
Do not install modern grpcio-tools into this runtime: it requires a conflicting
protobuf release. APK rebuilding belongs in the shared Quest driver/toolchain,
not this project env. SenseGlove has no Python binding at the tested pin.
RealSense, when needed, uses a per-project pinned pyrealsense2 wheel; master
has no executable camera integration.

Legacy source work is preserved as the pure snapshot branch `legacy-wip`;
its driver pointer changes were intentionally omitted because those drivers'
unique contents are already preserved under `~/Robots`. Legacy logs are in
`../runs/legacy-logs`, dependency backend logs in `../runs/legacy-dependency-logs`.
Historical specs/plans retain old architecture, not current installation rules.
No live hardware behavior was verified by the migration.

## Planned structure

```text
dex_teleop/
  devices/      # Oculus, SenseGlove, and other input devices
  robots/       # Franka and other robot interfaces
  retargeting/  # Hand/control retargeting utilities
../runs/        # Project outputs (outside the checkout)
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
    interface_cfg: ~/Robots/franka/deoxys_control/deoxys/config/franka-three.yml
    controller: right
    gripper: null

grippers: {}
```

With this setup, Oculus grip still controls the arm clutch; trigger commands are
ignored for that arm.
