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
- Created with: `~/Apps/miniconda3/bin/conda create -y -n dex-teleop -c conda-forge --override-channels python=3.10 pip`
- Location: `/home/kefhuang/Apps/miniconda3/envs/dex-teleop`

`PYTHONNOUSERSITE=1` is intentional: this machine has user-site Python packages that can otherwise shadow packages inside the conda environment, including `pip`.

Do not recreate the environment unless explicitly requested. Install any additional project dependencies into `dex-teleop` and document important changes here.


## Environment changes

- 2026-07-06: Installed `pytest` into the existing `dex-teleop` conda environment for repository test execution.
- 2026-07-05: Installed `~/Robots/quest/oculus_reader/requirements.txt` into the existing `dex-teleop` conda environment for Oculus Reader diagnostics: `numpy`, `pure-python-adb`, and `pyyaml`.
- 2026-07-05: Installed deoxys Python diagnostic dependencies into the existing `dex-teleop` conda environment: `easydict`, `termcolor`, `simplejson`, `pyzmq`, `protobuf`, `grpcio-tools`, and `pybullet`.
- 2026-07-05: Generated deoxys Python protobuf modules from `~/Robots/franka/deoxys_control/deoxys/proto/franka-interface/*.proto` into `~/Robots/franka/deoxys_control/deoxys/deoxys/proto/franka_interface/` for `FrankaInterface` import/read diagnostics.

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
