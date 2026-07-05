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

`PYTHONNOUSERSITE=1` is intentional: this machine has user-site Python packages that can otherwise shadow packages inside the conda environment, including `pip`.

Do not recreate the environment unless explicitly requested. Install any additional project dependencies into `dex-teleop` and document important changes here.


## Environment changes

- 2026-07-05: Installed `dependency/oculus_reader/requirements.txt` into the existing `dex-teleop` conda environment for Oculus Reader diagnostics: `numpy`, `pure-python-adb`, and `pyyaml`.
- 2026-07-05: Installed deoxys Python diagnostic dependencies into the existing `dex-teleop` conda environment: `easydict`, `termcolor`, `simplejson`, `pyzmq`, `protobuf`, `grpcio-tools`, and `pybullet`.
- 2026-07-05: Generated deoxys Python protobuf modules from `dependency/deoxys_control/deoxys/proto/franka-interface/*.proto` into `dependency/deoxys_control/deoxys/deoxys/proto/franka_interface/` for `FrankaInterface` import/read diagnostics.
