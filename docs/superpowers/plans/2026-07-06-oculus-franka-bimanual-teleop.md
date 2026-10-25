# Oculus Franka Bimanual Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, testable Oculus-to-dual-Franka teleoperation path with clutch motion, continuous trigger gripper control, dry-run robot output, and mock controller input.

**Architecture:** Implement a small interface-first framework: core typed commands/config, input adapters, robot/gripper adapters, a bimanual clutch policy, and one app entry point. Keep real hardware adapters thin over vendored Oculus Reader and Deoxys while making all policy/config logic testable without hardware.

**Tech Stack:** Python 3.10, dataclasses, Protocol typing, NumPy, PyYAML, pytest, vendored `~/Robots/quest/oculus_reader`, vendored `~/Robots/franka/deoxys_control`.

---

## File structure and responsibilities

Create these files:

- `dex_teleop/core/__init__.py` — exports core data classes.
- `dex_teleop/core/types.py` — `Pose`, `ControllerState`, `DeviceState`, `MotionCommand`, `GripperCommand`, `ArmState`, `TeleopCommandBatch`.
- `dex_teleop/core/config.py` — YAML loading, required-field validation, CLI override helper, path resolution for Franka config files.
- `dex_teleop/core/loop.py` — fixed-rate sleeping and graceful shutdown helpers.
- `dex_teleop/devices/base.py` — `InputDevice` protocol.
- `dex_teleop/devices/mock.py` — deterministic `MockControllerInput`.
- `dex_teleop/devices/oculus.py` — `OculusDevice` adapter around vendored `OculusReader`.
- `dex_teleop/robots/arm.py` — `ArmController` protocol.
- `dex_teleop/robots/gripper.py` — `GripperController` protocol.
- `dex_teleop/robots/mock.py` — `MockArmController` and `MockGripperController` that record commands.
- `dex_teleop/robots/franka.py` — Deoxys-backed Franka arm/gripper adapter.
- `dex_teleop/policies/__init__.py` — exports policy.
- `dex_teleop/policies/bimanual_clutch.py` — clutch state machine and controller-to-arm binding.
- `dex_teleop/utils/__init__.py` — utility package marker.
- `dex_teleop/utils/transforms.py` — signed axis mapping, quaternion-to-RPY delta, deadzone, scale, clamp.
- `dex_teleop/apps/__init__.py` — app package marker.
- `dex_teleop/apps/oculus_franka.py` — command-line entry point and app wiring.
- `configs/oculus_franka_bimanual.yml` — example bimanual config, defaulting to safe mock/dry-run.
- `tests/core/test_config.py` — config loader tests.
- `tests/utils/test_transforms.py` — mapping and safety math tests.
- `tests/policies/test_bimanual_clutch.py` — clutch and trigger policy tests.
- `tests/apps/test_oculus_franka_app.py` — dry-run smoke tests.

Modify these files:

- `README.md` — add a short usage section for dry-run and hardware smoke path.
- `.gitignore` if needed — keep generated logs/caches out of commits; do not ignore source/config/test files.

Use the repository environment for every command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
```

---

### Task 1: Core data types and package structure

**Files:**
- Create: `dex_teleop/core/__init__.py`
- Create: `dex_teleop/core/types.py`
- Create: `dex_teleop/apps/__init__.py`
- Create: `dex_teleop/policies/__init__.py`
- Create: `dex_teleop/utils/__init__.py`
- Test: `tests/core/test_types.py`

- [ ] **Step 1: Write failing tests for core types**

Create `tests/core/test_types.py`:

```python
from __future__ import annotations

import numpy as np

from dex_teleop.core.types import (
    ArmState,
    ControllerState,
    DeviceState,
    GripperCommand,
    MotionCommand,
    Pose,
    TeleopCommandBatch,
)


def test_pose_arrays_are_float_arrays():
    pose = Pose(position=[1, 2, 3], orientation=[0, 0, 0, 1])

    assert pose.position.dtype == float
    assert pose.orientation.dtype == float
    assert pose.position.shape == (3,)
    assert pose.orientation.shape == (4,)


def test_motion_command_defaults_to_disabled_zero_delta():
    command = MotionCommand.disabled("left")

    assert command.arm_id == "left"
    assert command.enabled is False
    assert np.allclose(command.delta_position, np.zeros(3))
    assert np.allclose(command.delta_rotation, np.zeros(3))


def test_gripper_command_clamps_normalized_value():
    low = GripperCommand("left_gripper", -0.2)
    high = GripperCommand("right_gripper", 1.4)

    assert low.normalized == 0.0
    assert high.normalized == 1.0


def test_command_batch_groups_arm_and_gripper_commands():
    batch = TeleopCommandBatch(
        motions={"left": MotionCommand.disabled("left")},
        grippers={"left_gripper": GripperCommand("left_gripper", 0.5)},
    )

    assert set(batch.motions) == {"left"}
    assert set(batch.grippers) == {"left_gripper"}


def test_device_state_contains_controller_states():
    controller = ControllerState(
        name="left",
        pose=Pose.identity(),
        buttons={"grip": True},
        axes={"trigger": 0.25},
    )
    state = DeviceState(timestamp=123.0, controllers={"left": controller})

    assert state.controllers["left"].buttons["grip"] is True
    assert state.controllers["left"].axes["trigger"] == 0.25


def test_arm_state_can_hold_current_pose():
    state = ArmState(arm_id="left", pose=Pose.identity())

    assert state.arm_id == "left"
    assert np.allclose(state.pose.position, np.zeros(3))
```

- [ ] **Step 2: Run tests and verify they fail because modules are missing**

Run:

```bash
pytest tests/core/test_types.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dex_teleop.core'` or missing class imports.

- [ ] **Step 3: Implement core types**

Create `dex_teleop/core/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


@dataclass
class Pose:
    """Cartesian pose with quaternion orientation in xyzw convention."""

    position: np.ndarray | list[float] | tuple[float, float, float]
    orientation: np.ndarray | list[float] | tuple[float, float, float, float]

    def __post_init__(self) -> None:
        self.position = _array(self.position, (3,), "position")
        self.orientation = _array(self.orientation, (4,), "orientation")

    @classmethod
    def identity(cls) -> "Pose":
        return cls(position=np.zeros(3), orientation=np.array([0.0, 0.0, 0.0, 1.0]))


@dataclass
class ControllerState:
    name: str
    pose: Pose
    buttons: dict[str, bool] = field(default_factory=dict)
    axes: dict[str, float] = field(default_factory=dict)

    def button(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))

    def axis(self, name: str) -> float:
        return float(self.axes.get(name, 0.0))


@dataclass
class DeviceState:
    timestamp: float
    controllers: dict[str, ControllerState]


@dataclass
class ArmState:
    arm_id: str
    pose: Pose


@dataclass
class MotionCommand:
    arm_id: str
    delta_position: np.ndarray | list[float] | tuple[float, float, float]
    delta_rotation: np.ndarray | list[float] | tuple[float, float, float]
    enabled: bool

    def __post_init__(self) -> None:
        self.delta_position = _array(self.delta_position, (3,), "delta_position")
        self.delta_rotation = _array(self.delta_rotation, (3,), "delta_rotation")

    @classmethod
    def disabled(cls, arm_id: str) -> "MotionCommand":
        return cls(
            arm_id=arm_id,
            delta_position=np.zeros(3),
            delta_rotation=np.zeros(3),
            enabled=False,
        )


@dataclass
class GripperCommand:
    gripper_id: str
    normalized: float

    def __post_init__(self) -> None:
        self.normalized = _clamp01(float(self.normalized))


@dataclass
class TeleopCommandBatch:
    motions: dict[str, MotionCommand]
    grippers: dict[str, GripperCommand]
```

Create `dex_teleop/core/__init__.py`:

```python
from dex_teleop.core.types import (
    ArmState,
    ControllerState,
    DeviceState,
    GripperCommand,
    MotionCommand,
    Pose,
    TeleopCommandBatch,
)

__all__ = [
    "ArmState",
    "ControllerState",
    "DeviceState",
    "GripperCommand",
    "MotionCommand",
    "Pose",
    "TeleopCommandBatch",
]
```

Create package marker files:

```bash
printf '' > dex_teleop/apps/__init__.py
printf '' > dex_teleop/policies/__init__.py
printf '' > dex_teleop/utils/__init__.py
```

- [ ] **Step 4: Run type tests and verify they pass**

Run:

```bash
pytest tests/core/test_types.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit core types**

```bash
git add dex_teleop/core dex_teleop/apps/__init__.py dex_teleop/policies/__init__.py dex_teleop/utils/__init__.py tests/core/test_types.py
git commit -m "feat: add teleop core command types"
```

---

### Task 2: Transform utilities for axis mapping, deadzone, scaling, and limits

**Files:**
- Create: `dex_teleop/utils/transforms.py`
- Test: `tests/utils/test_transforms.py`

- [ ] **Step 1: Write failing transform tests**

Create `tests/utils/test_transforms.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from dex_teleop.utils.transforms import (
    apply_deadzone,
    axis_mapping,
    clamp_vector_norm,
    limit_components,
    quaternion_delta_rpy,
)


def test_axis_mapping_applies_signs_and_source_axes():
    result = axis_mapping(
        vector=np.array([1.0, 2.0, 3.0]),
        mapping={"x": "+z", "y": "-x", "z": "+y"},
        output_axes=("x", "y", "z"),
        input_axes=("x", "y", "z"),
    )

    assert np.allclose(result, np.array([3.0, -1.0, 2.0]))


def test_axis_mapping_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unknown source axis"):
        axis_mapping(
            vector=np.zeros(3),
            mapping={"x": "+bad", "y": "+x", "z": "+z"},
            output_axes=("x", "y", "z"),
            input_axes=("x", "y", "z"),
        )


def test_apply_deadzone_zeroes_small_components():
    result = apply_deadzone(np.array([0.001, -0.01, 0.2]), threshold=0.01)

    assert np.allclose(result, np.array([0.0, 0.0, 0.2]))


def test_limit_components_clamps_each_component():
    result = limit_components(np.array([0.2, -0.2, 0.03]), max_abs=0.05)

    assert np.allclose(result, np.array([0.05, -0.05, 0.03]))


def test_clamp_vector_norm_preserves_direction():
    result = clamp_vector_norm(np.array([3.0, 4.0, 0.0]), max_norm=1.0)

    assert np.allclose(result, np.array([0.6, 0.8, 0.0]))


def test_quaternion_delta_rpy_identity_is_zero():
    delta = quaternion_delta_rpy(
        anchor_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        current_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )

    assert np.allclose(delta, np.zeros(3))
```

- [ ] **Step 2: Run tests and verify they fail because transforms are missing**

Run:

```bash
pytest tests/utils/test_transforms.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing functions.

- [ ] **Step 3: Implement transform utilities**

Create `dex_teleop/utils/transforms.py`:

```python
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def _parse_signed_axis(spec: str) -> tuple[float, str]:
    if not spec:
        raise ValueError("Axis mapping value cannot be empty")
    sign = 1.0
    axis = spec
    if spec[0] in {"+", "-"}:
        sign = -1.0 if spec[0] == "-" else 1.0
        axis = spec[1:]
    if not axis:
        raise ValueError(f"Invalid axis mapping value: {spec!r}")
    return sign, axis


def axis_mapping(
    vector: np.ndarray,
    mapping: Mapping[str, str],
    output_axes: Sequence[str],
    input_axes: Sequence[str],
) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.shape != (len(input_axes),):
        raise ValueError(f"vector must have shape ({len(input_axes)},), got {values.shape}")
    input_index = {axis: i for i, axis in enumerate(input_axes)}
    output = []
    for out_axis in output_axes:
        if out_axis not in mapping:
            raise ValueError(f"Missing mapping for output axis {out_axis!r}")
        sign, in_axis = _parse_signed_axis(mapping[out_axis])
        if in_axis not in input_index:
            raise ValueError(f"Unknown source axis {in_axis!r} for output axis {out_axis!r}")
        output.append(sign * values[input_index[in_axis]])
    return np.asarray(output, dtype=float)


def apply_deadzone(vector: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(vector, dtype=float).copy()
    values[np.abs(values) <= float(threshold)] = 0.0
    return values


def limit_components(vector: np.ndarray, max_abs: float) -> np.ndarray:
    return np.clip(np.asarray(vector, dtype=float), -float(max_abs), float(max_abs))


def clamp_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm == 0.0 or norm <= float(max_norm):
        return values.copy()
    return values * (float(max_norm) / norm)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    values = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Quaternion norm cannot be zero")
    return values / norm


def _quat_to_rpy(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)


def quaternion_delta_rpy(anchor_xyzw: np.ndarray, current_xyzw: np.ndarray) -> np.ndarray:
    anchor = _normalize_quat(anchor_xyzw)
    current = _normalize_quat(current_xyzw)
    delta = _quat_multiply(current, _quat_conjugate(anchor))
    return _quat_to_rpy(_normalize_quat(delta))
```

- [ ] **Step 4: Run transform tests and verify they pass**

Run:

```bash
pytest tests/utils/test_transforms.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit transform utilities**

```bash
git add dex_teleop/utils/transforms.py tests/utils/test_transforms.py
git commit -m "feat: add teleop transform utilities"
```

---

### Task 3: Config loading and example bimanual YAML

**Files:**
- Create: `dex_teleop/core/config.py`
- Create: `configs/oculus_franka_bimanual.yml`
- Test: `tests/core/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/core/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from dex_teleop.core.config import ConfigError, apply_cli_overrides, load_config


def test_load_example_config_has_two_arms_and_safe_defaults():
    cfg = load_config("configs/oculus_franka_bimanual.yml")

    assert cfg["app"]["dry_run"] is True
    assert cfg["input"]["mock_controller"] is True
    assert set(cfg["arms"]) == {"left", "right"}
    assert cfg["arms"]["left"]["controller"] == "left"
    assert cfg["controls"]["clutch_button"] == "grip"


def test_missing_required_section_reports_field_path(tmp_path: Path):
    path = tmp_path / "bad.yml"
    path.write_text("app:\n  rate_hz: 30\n")

    with pytest.raises(ConfigError, match="Missing required config field: input"):
        load_config(path)


def test_missing_arm_interface_cfg_reports_arm_path(tmp_path: Path):
    path = tmp_path / "bad.yml"
    path.write_text(
        """
app:
  rate_hz: 30
  dry_run: true
input:
  type: oculus
  mock_controller: true
controls:
  clutch_button: grip
  gripper_axis: trigger
arms:
  left:
    type: franka
    controller: left
    gripper: left_gripper
grippers:
  left_gripper:
    type: franka
    arm: left
mapping:
  position_axes: {x: '+x', y: '+y', z: '+z'}
  rotation_axes: {roll: '+roll', pitch: '+pitch', yaw: '+yaw'}
scales:
  position: 1.0
  rotation: 1.0
safety:
  max_translation_per_cycle: 0.01
  max_rotation_per_cycle: 0.05
  position_deadzone: 0.002
  rotation_deadzone: 0.01
  send_zero_when_idle: true
""".strip()
    )

    with pytest.raises(ConfigError, match="arms.left.interface_cfg"):
        load_config(path)


def test_cli_overrides_update_dry_run_mock_controller_and_rate():
    cfg = load_config("configs/oculus_franka_bimanual.yml")
    updated = apply_cli_overrides(
        cfg,
        dry_run=False,
        mock_controller=False,
        rate_hz=60.0,
    )

    assert updated["app"]["dry_run"] is False
    assert updated["input"]["mock_controller"] is False
    assert updated["app"]["rate_hz"] == 60.0
```

- [ ] **Step 2: Run config tests and verify they fail**

Run:

```bash
pytest tests/core/test_config.py -v
```

Expected: FAIL because `dex_teleop.core.config` and config file do not exist.

- [ ] **Step 3: Implement config loader and YAML file**

Create `dex_teleop/core/config.py`:

```python
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


REQUIRED_TOP_LEVEL = [
    "app",
    "input",
    "controls",
    "arms",
    "grippers",
    "mapping",
    "scales",
    "safety",
]


def _require(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required config field: {path}.{key}" if path else f"Missing required config field: {key}")
    return mapping[key]


def _validate(cfg: dict[str, Any]) -> None:
    for section in REQUIRED_TOP_LEVEL:
        _require(cfg, section, "")
    for arm_id, arm_cfg in cfg["arms"].items():
        for key in ("type", "interface_cfg", "controller", "gripper"):
            _require(arm_cfg, key, f"arms.{arm_id}")
        gripper_id = arm_cfg["gripper"]
        if gripper_id not in cfg["grippers"]:
            raise ConfigError(f"arms.{arm_id}.gripper references unknown gripper {gripper_id!r}")
    for gripper_id, gripper_cfg in cfg["grippers"].items():
        for key in ("type", "arm"):
            _require(gripper_cfg, key, f"grippers.{gripper_id}")
    for key in ("clutch_button", "gripper_axis"):
        _require(cfg["controls"], key, "controls")
    for key in ("position_axes", "rotation_axes"):
        _require(cfg["mapping"], key, "mapping")
    for key in ("position", "rotation"):
        _require(cfg["scales"], key, "scales")
    for key in (
        "max_translation_per_cycle",
        "max_rotation_per_cycle",
        "position_deadzone",
        "rotation_deadzone",
        "send_zero_when_idle",
    ):
        _require(cfg["safety"], key, "safety")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    _validate(loaded)
    return loaded


def apply_cli_overrides(
    cfg: dict[str, Any],
    *,
    dry_run: bool | None = None,
    mock_controller: bool | None = None,
    rate_hz: float | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    if dry_run is not None:
        updated["app"]["dry_run"] = bool(dry_run)
    if mock_controller is not None:
        updated["input"]["mock_controller"] = bool(mock_controller)
    if rate_hz is not None:
        updated["app"]["rate_hz"] = float(rate_hz)
    return updated


def resolve_repo_path(path: str | Path, repo_root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = repo_root if repo_root is not None else Path.cwd()
    return root / candidate
```

Create `configs/oculus_franka_bimanual.yml`:

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

- [ ] **Step 4: Run config tests and verify they pass**

Run:

```bash
pytest tests/core/test_config.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit config loader and example config**

```bash
git add dex_teleop/core/config.py configs/oculus_franka_bimanual.yml tests/core/test_config.py
git commit -m "feat: add teleop config loading"
```

---

### Task 4: Bimanual clutch policy

**Files:**
- Create: `dex_teleop/policies/bimanual_clutch.py`
- Modify: `dex_teleop/policies/__init__.py`
- Test: `tests/policies/test_bimanual_clutch.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/policies/test_bimanual_clutch.py`:

```python
from __future__ import annotations

import numpy as np

from dex_teleop.core.types import ControllerState, DeviceState, Pose
from dex_teleop.policies.bimanual_clutch import BimanualClutchPolicy


def _state(left_pos=(0.0, 0.0, 0.0), left_grip=False, left_trigger=0.0):
    left = ControllerState(
        name="left",
        pose=Pose(position=left_pos, orientation=[0, 0, 0, 1]),
        buttons={"grip": left_grip},
        axes={"trigger": left_trigger},
    )
    right = ControllerState(
        name="right",
        pose=Pose.identity(),
        buttons={"grip": False},
        axes={"trigger": 0.0},
    )
    return DeviceState(timestamp=1.0, controllers={"left": left, "right": right})


def _policy():
    return BimanualClutchPolicy(
        arm_bindings={
            "left": {"controller": "left", "gripper": "left_gripper"},
            "right": {"controller": "right", "gripper": "right_gripper"},
        },
        clutch_button="grip",
        gripper_axis="trigger",
        position_axes={"x": "+z", "y": "-x", "z": "+y"},
        rotation_axes={"roll": "+roll", "pitch": "+pitch", "yaw": "+yaw"},
        position_scale=1.0,
        rotation_scale=1.0,
        max_translation_per_cycle=0.01,
        max_rotation_per_cycle=0.05,
        position_deadzone=0.0,
        rotation_deadzone=0.0,
        send_zero_when_idle=True,
    )


def test_idle_grip_released_emits_disabled_motion_and_gripper_command():
    batch = _policy().update(_state(left_grip=False, left_trigger=0.25))

    assert batch.motions["left"].enabled is False
    assert batch.grippers["left_gripper"].normalized == 0.25


def test_first_grip_press_establishes_anchor_without_jump():
    batch = _policy().update(_state(left_pos=(1, 2, 3), left_grip=True))

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))


def test_held_grip_outputs_mapped_limited_delta_from_anchor():
    policy = _policy()
    policy.update(_state(left_pos=(0, 0, 0), left_grip=True))
    batch = policy.update(_state(left_pos=(0.2, 0.3, 0.4), left_grip=True))

    assert batch.motions["left"].enabled is True
    assert np.allclose(batch.motions["left"].delta_position, np.array([0.01, -0.01, 0.01]))


def test_grip_release_clears_anchor_and_disables_motion():
    policy = _policy()
    policy.update(_state(left_pos=(0, 0, 0), left_grip=True))
    batch = policy.update(_state(left_pos=(0.2, 0.3, 0.4), left_grip=False))

    assert batch.motions["left"].enabled is False
    assert np.allclose(batch.motions["left"].delta_position, np.zeros(3))


def test_missing_controller_disables_bound_arm():
    policy = _policy()
    state = DeviceState(timestamp=1.0, controllers={})
    batch = policy.update(state)

    assert batch.motions["left"].enabled is False
    assert batch.motions["right"].enabled is False
```

- [ ] **Step 2: Run policy tests and verify they fail**

Run:

```bash
pytest tests/policies/test_bimanual_clutch.py -v
```

Expected: FAIL because `BimanualClutchPolicy` is missing.

- [ ] **Step 3: Implement bimanual clutch policy**

Create `dex_teleop/policies/bimanual_clutch.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dex_teleop.core.types import DeviceState, GripperCommand, MotionCommand, Pose, TeleopCommandBatch
from dex_teleop.utils.transforms import apply_deadzone, axis_mapping, limit_components, quaternion_delta_rpy


@dataclass
class _ArmClutchState:
    active: bool = False
    anchor_pose: Pose | None = None


class BimanualClutchPolicy:
    def __init__(
        self,
        *,
        arm_bindings: dict[str, dict[str, Any]],
        clutch_button: str,
        gripper_axis: str,
        position_axes: dict[str, str],
        rotation_axes: dict[str, str],
        position_scale: float,
        rotation_scale: float,
        max_translation_per_cycle: float,
        max_rotation_per_cycle: float,
        position_deadzone: float,
        rotation_deadzone: float,
        send_zero_when_idle: bool,
    ) -> None:
        self.arm_bindings = arm_bindings
        self.clutch_button = clutch_button
        self.gripper_axis = gripper_axis
        self.position_axes = position_axes
        self.rotation_axes = rotation_axes
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.max_translation_per_cycle = float(max_translation_per_cycle)
        self.max_rotation_per_cycle = float(max_rotation_per_cycle)
        self.position_deadzone = float(position_deadzone)
        self.rotation_deadzone = float(rotation_deadzone)
        self.send_zero_when_idle = bool(send_zero_when_idle)
        self._states = {arm_id: _ArmClutchState() for arm_id in arm_bindings}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BimanualClutchPolicy":
        return cls(
            arm_bindings={
                arm_id: {"controller": arm_cfg["controller"], "gripper": arm_cfg["gripper"]}
                for arm_id, arm_cfg in cfg["arms"].items()
            },
            clutch_button=cfg["controls"]["clutch_button"],
            gripper_axis=cfg["controls"]["gripper_axis"],
            position_axes=cfg["mapping"]["position_axes"],
            rotation_axes=cfg["mapping"]["rotation_axes"],
            position_scale=cfg["scales"]["position"],
            rotation_scale=cfg["scales"]["rotation"],
            max_translation_per_cycle=cfg["safety"]["max_translation_per_cycle"],
            max_rotation_per_cycle=cfg["safety"]["max_rotation_per_cycle"],
            position_deadzone=cfg["safety"]["position_deadzone"],
            rotation_deadzone=cfg["safety"]["rotation_deadzone"],
            send_zero_when_idle=cfg["safety"]["send_zero_when_idle"],
        )

    def update(self, state: DeviceState) -> TeleopCommandBatch:
        motions: dict[str, MotionCommand] = {}
        grippers: dict[str, GripperCommand] = {}
        for arm_id, binding in self.arm_bindings.items():
            controller_name = binding["controller"]
            gripper_id = binding["gripper"]
            controller = state.controllers.get(controller_name)
            if controller is None:
                self._states[arm_id] = _ArmClutchState()
                motions[arm_id] = MotionCommand.disabled(arm_id)
                grippers[gripper_id] = GripperCommand(gripper_id, 0.0)
                continue

            grippers[gripper_id] = GripperCommand(gripper_id, controller.axis(self.gripper_axis))
            clutch_state = self._states[arm_id]
            if not controller.button(self.clutch_button):
                self._states[arm_id] = _ArmClutchState()
                motions[arm_id] = MotionCommand.disabled(arm_id)
                continue

            if not clutch_state.active or clutch_state.anchor_pose is None:
                self._states[arm_id] = _ArmClutchState(active=True, anchor_pose=controller.pose)
                motions[arm_id] = MotionCommand(
                    arm_id=arm_id,
                    delta_position=np.zeros(3),
                    delta_rotation=np.zeros(3),
                    enabled=True,
                )
                continue

            raw_pos_delta = controller.pose.position - clutch_state.anchor_pose.position
            raw_rot_delta = quaternion_delta_rpy(clutch_state.anchor_pose.orientation, controller.pose.orientation)
            mapped_pos = axis_mapping(raw_pos_delta, self.position_axes, ("x", "y", "z"), ("x", "y", "z"))
            mapped_rot = axis_mapping(
                raw_rot_delta,
                self.rotation_axes,
                ("roll", "pitch", "yaw"),
                ("roll", "pitch", "yaw"),
            )
            delta_position = limit_components(
                apply_deadzone(mapped_pos * self.position_scale, self.position_deadzone),
                self.max_translation_per_cycle,
            )
            delta_rotation = limit_components(
                apply_deadzone(mapped_rot * self.rotation_scale, self.rotation_deadzone),
                self.max_rotation_per_cycle,
            )
            motions[arm_id] = MotionCommand(
                arm_id=arm_id,
                delta_position=delta_position,
                delta_rotation=delta_rotation,
                enabled=True,
            )
        return TeleopCommandBatch(motions=motions, grippers=grippers)
```

Modify `dex_teleop/policies/__init__.py`:

```python
from dex_teleop.policies.bimanual_clutch import BimanualClutchPolicy

__all__ = ["BimanualClutchPolicy"]
```

- [ ] **Step 4: Run policy tests and verify they pass**

Run:

```bash
pytest tests/policies/test_bimanual_clutch.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit clutch policy**

```bash
git add dex_teleop/policies tests/policies/test_bimanual_clutch.py
git commit -m "feat: add bimanual clutch policy"
```

---

### Task 5: Device interfaces, mock controller input, and Oculus adapter

**Files:**
- Create: `dex_teleop/devices/base.py`
- Create: `dex_teleop/devices/mock.py`
- Create: `dex_teleop/devices/oculus.py`
- Modify: `dex_teleop/devices/__init__.py`
- Test: `tests/devices/test_mock_controller.py`

- [ ] **Step 1: Write failing mock controller tests**

Create `tests/devices/test_mock_controller.py`:

```python
from __future__ import annotations

from dex_teleop.devices.mock import MockControllerInput


def test_mock_controller_returns_left_and_right_states():
    device = MockControllerInput()
    device.start()
    state = device.read()
    device.stop()

    assert set(state.controllers) == {"left", "right"}
    assert "grip" in state.controllers["left"].buttons
    assert "trigger" in state.controllers["right"].axes


def test_mock_controller_script_progresses_deterministically():
    device = MockControllerInput()
    device.start()
    first = device.read()
    second = device.read()

    assert second.timestamp > first.timestamp
    assert second.controllers["left"].pose.position[0] >= first.controllers["left"].pose.position[0]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/devices/test_mock_controller.py -v
```

Expected: FAIL because device modules are missing.

- [ ] **Step 3: Implement device protocol, mock input, and Oculus adapter**

Create `dex_teleop/devices/base.py`:

```python
from __future__ import annotations

from typing import Protocol

from dex_teleop.core.types import DeviceState


class InputDevice(Protocol):
    def start(self) -> None:
        pass

    def read(self) -> DeviceState:
        pass

    def stop(self) -> None:
        pass
```

Create `dex_teleop/devices/mock.py`:

```python
from __future__ import annotations

import time

import numpy as np

from dex_teleop.core.types import ControllerState, DeviceState, Pose


class MockControllerInput:
    def __init__(self, *, dt: float = 1.0 / 30.0) -> None:
        self.dt = float(dt)
        self._started = False
        self._step = 0
        self._start_time = 0.0

    def start(self) -> None:
        self._started = True
        self._step = 0
        self._start_time = time.monotonic()

    def read(self) -> DeviceState:
        if not self._started:
            self.start()
        t = self._step * self.dt
        left_grip = self._step >= 1
        right_grip = self._step >= 2
        left = ControllerState(
            name="left",
            pose=Pose(position=np.array([0.001 * self._step, 0.0, 0.0]), orientation=[0, 0, 0, 1]),
            buttons={"grip": left_grip},
            axes={"trigger": min(1.0, 0.1 * self._step)},
        )
        right = ControllerState(
            name="right",
            pose=Pose(position=np.array([0.0, 0.001 * self._step, 0.0]), orientation=[0, 0, 0, 1]),
            buttons={"grip": right_grip},
            axes={"trigger": min(1.0, 0.05 * self._step)},
        )
        self._step += 1
        return DeviceState(timestamp=self._start_time + t, controllers={"left": left, "right": right})

    def stop(self) -> None:
        self._started = False
```

Create `dex_teleop/devices/oculus.py`:

```python
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.types import ControllerState, DeviceState, Pose


REPO_ROOT = Path(__file__).resolve().parents[2]
OCULUS_READER_ROOT = Path.home() / "Robots" / "quest" / "oculus_reader"


class OculusDevice:
    def __init__(self, *, ip: str | None = None, port: int = 5555, print_fps: bool = False) -> None:
        self.ip = ip
        self.port = int(port)
        self.print_fps = bool(print_fps)
        self._reader: Any | None = None

    def start(self) -> None:
        if not OCULUS_READER_ROOT.exists():
            raise RuntimeError(f"Missing Oculus Reader dependency: {OCULUS_READER_ROOT}")
        sys.path.insert(0, str(OCULUS_READER_ROOT))
        from oculus_reader.reader import OculusReader  # noqa: PLC0415

        self._reader = OculusReader(
            ip_address=self.ip,
            port=self.port,
            print_FPS=self.print_fps,
            run=True,
        )

    def read(self) -> DeviceState:
        if self._reader is None:
            raise RuntimeError("OculusDevice.start() must be called before read()")
        transforms, buttons = self._reader.get_transformations_and_buttons()
        controllers = self._convert(transforms or {}, buttons or {})
        return DeviceState(timestamp=time.monotonic(), controllers=controllers)

    def stop(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

    def _convert(self, transforms: dict[str, Any], buttons: dict[str, Any]) -> dict[str, ControllerState]:
        controllers: dict[str, ControllerState] = {}
        for name, key in (("left", "l"), ("right", "r")):
            matrix = self._find_transform(transforms, key)
            pose = self._pose_from_matrix(matrix) if matrix is not None else Pose.identity()
            controllers[name] = ControllerState(
                name=name,
                pose=pose,
                buttons={"grip": self._button_bool(buttons, name, "grip")},
                axes={"trigger": self._button_float(buttons, name, "trigger")},
            )
        return controllers

    def _find_transform(self, transforms: dict[str, Any], side_key: str) -> Any | None:
        for key, value in transforms.items():
            lowered = str(key).lower()
            if side_key in lowered or (side_key == "l" and "left" in lowered) or (side_key == "r" and "right" in lowered):
                return value
        return None

    def _pose_from_matrix(self, matrix: Any) -> Pose:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape == (4, 4):
            position = arr[:3, 3]
        else:
            position = np.zeros(3)
        return Pose(position=position, orientation=[0, 0, 0, 1])

    def _button_bool(self, buttons: dict[str, Any], side: str, name: str) -> bool:
        candidates = [f"{side}_{name}", f"{side[0]}_{name}", f"{side}{name}", name]
        for candidate in candidates:
            if candidate in buttons:
                return bool(buttons[candidate])
        return False

    def _button_float(self, buttons: dict[str, Any], side: str, name: str) -> float:
        candidates = [f"{side}_{name}", f"{side[0]}_{name}", f"{side}{name}", name]
        for candidate in candidates:
            if candidate in buttons:
                return float(buttons[candidate])
        return 0.0
```

Modify `dex_teleop/devices/__init__.py`:

```python
from dex_teleop.devices.base import InputDevice
from dex_teleop.devices.mock import MockControllerInput
from dex_teleop.devices.oculus import OculusDevice

__all__ = ["InputDevice", "MockControllerInput", "OculusDevice"]
```

- [ ] **Step 4: Run mock controller tests and verify they pass**

Run:

```bash
pytest tests/devices/test_mock_controller.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit device adapters**

```bash
git add dex_teleop/devices tests/devices/test_mock_controller.py
git commit -m "feat: add oculus and mock controller inputs"
```

---

### Task 6: Robot protocols, mock robot output, and Franka adapter

**Files:**
- Create: `dex_teleop/robots/arm.py`
- Create: `dex_teleop/robots/gripper.py`
- Create: `dex_teleop/robots/mock.py`
- Create: `dex_teleop/robots/franka.py`
- Modify: `dex_teleop/robots/__init__.py`
- Test: `tests/robots/test_mock_robot.py`
- Test: `tests/robots/test_franka_adapter.py`

- [ ] **Step 1: Write failing robot adapter tests**

Create `tests/robots/test_mock_robot.py`:

```python
from __future__ import annotations

import numpy as np

from dex_teleop.core.types import GripperCommand, MotionCommand
from dex_teleop.robots.mock import MockArmController, MockGripperController


def test_mock_arm_records_motion_and_stop():
    arm = MockArmController("left")
    arm.connect()
    arm.send_motion(MotionCommand("left", [0.1, 0, 0], [0, 0, 0], True))
    arm.stop_motion()

    assert arm.connected is True
    assert len(arm.motion_commands) == 2
    assert arm.motion_commands[0].enabled is True
    assert arm.motion_commands[1].enabled is False


def test_mock_gripper_records_commands():
    gripper = MockGripperController("left_gripper")
    gripper.connect()
    gripper.send_gripper(GripperCommand("left_gripper", 0.75))

    assert gripper.connected is True
    assert gripper.gripper_commands[-1].normalized == 0.75
```

Create `tests/robots/test_franka_adapter.py`:

```python
from __future__ import annotations

import numpy as np

from dex_teleop.core.types import GripperCommand, MotionCommand
from dex_teleop.robots.franka import FrankaArmController, FrankaGripperController


class FakeFrankaInterface:
    def __init__(self):
        self.control_calls = []
        self.gripper_calls = []
        self.closed = False

    def control(self, controller_type, action, controller_cfg=None, termination=False):
        self.control_calls.append((controller_type, list(action), controller_cfg, termination))

    def gripper_control(self, action):
        self.gripper_calls.append(action)

    def close(self):
        self.closed = True


def test_franka_arm_sends_osc_pose_action_with_neutral_gripper_slot():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.send_motion(MotionCommand("left", [0.1, 0.2, 0.3], [0.01, 0.02, 0.03], True))

    assert fake.control_calls[-1][0] == "OSC_POSE"
    assert fake.control_calls[-1][1] == [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.0]


def test_franka_arm_stop_sends_zero_termination_action():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    arm.stop_motion()

    assert fake.control_calls[-1][1] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert fake.control_calls[-1][3] is True


def test_franka_gripper_converts_normalized_to_deoxys_action():
    fake = FakeFrankaInterface()
    gripper = FrankaGripperController("left_gripper", interface=fake)
    gripper.connect()
    gripper.send_gripper(GripperCommand("left_gripper", 1.0))
    gripper.send_gripper(GripperCommand("left_gripper", 0.0))

    assert fake.gripper_calls == [1.0, -1.0]
```

- [ ] **Step 2: Run robot tests and verify they fail**

Run:

```bash
pytest tests/robots/test_mock_robot.py tests/robots/test_franka_adapter.py -v
```

Expected: FAIL because robot modules/classes are missing.

- [ ] **Step 3: Implement robot protocols and adapters**

Create `dex_teleop/robots/arm.py`:

```python
from __future__ import annotations

from typing import Protocol

from dex_teleop.core.types import ArmState, MotionCommand


class ArmController(Protocol):
    def connect(self) -> None:
        pass

    def get_state(self) -> ArmState:
        pass

    def send_motion(self, command: MotionCommand) -> None:
        pass

    def stop_motion(self) -> None:
        pass

    def close(self) -> None:
        pass
```

Create `dex_teleop/robots/gripper.py`:

```python
from __future__ import annotations

from typing import Protocol

from dex_teleop.core.types import GripperCommand


class GripperController(Protocol):
    def connect(self) -> None:
        pass

    def send_gripper(self, command: GripperCommand) -> None:
        pass

    def close(self) -> None:
        pass
```

Create `dex_teleop/robots/mock.py`:

```python
from __future__ import annotations

from dex_teleop.core.types import ArmState, GripperCommand, MotionCommand, Pose


class MockArmController:
    def __init__(self, arm_id: str) -> None:
        self.arm_id = arm_id
        self.connected = False
        self.closed = False
        self.motion_commands: list[MotionCommand] = []

    def connect(self) -> None:
        self.connected = True

    def get_state(self) -> ArmState:
        return ArmState(arm_id=self.arm_id, pose=Pose.identity())

    def send_motion(self, command: MotionCommand) -> None:
        self.motion_commands.append(command)

    def stop_motion(self) -> None:
        self.motion_commands.append(MotionCommand.disabled(self.arm_id))

    def close(self) -> None:
        self.closed = True


class MockGripperController:
    def __init__(self, gripper_id: str) -> None:
        self.gripper_id = gripper_id
        self.connected = False
        self.closed = False
        self.gripper_commands: list[GripperCommand] = []

    def connect(self) -> None:
        self.connected = True

    def send_gripper(self, command: GripperCommand) -> None:
        self.gripper_commands.append(command)

    def close(self) -> None:
        self.closed = True
```

Create `dex_teleop/robots/franka.py`:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.types import ArmState, GripperCommand, MotionCommand, Pose


REPO_ROOT = Path(__file__).resolve().parents[2]
DEOXYS_ROOT = Path.home() / "Robots" / "franka" / "deoxys_control" / "deoxys"


class FrankaArmController:
    def __init__(
        self,
        arm_id: str,
        *,
        interface_cfg: str,
        interface: Any | None = None,
        controller_cfg: Any | None = None,
    ) -> None:
        self.arm_id = arm_id
        self.interface_cfg = interface_cfg
        self.interface = interface
        self.controller_cfg = controller_cfg
        self.connected = False

    def connect(self) -> None:
        if self.interface is None:
            self.interface = self._create_interface()
        self.connected = True

    def _create_interface(self) -> Any:
        if not DEOXYS_ROOT.exists():
            raise RuntimeError(f"Missing Deoxys dependency: {DEOXYS_ROOT}")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        sys.path.insert(0, str(DEOXYS_ROOT))
        from deoxys.franka_interface import FrankaInterface  # noqa: PLC0415
        from deoxys.utils.config_utils import get_default_controller_config  # noqa: PLC0415

        if self.controller_cfg is None:
            self.controller_cfg = get_default_controller_config(controller_type="OSC_POSE")
        return FrankaInterface(self.interface_cfg, use_visualizer=False)

    def get_state(self) -> ArmState:
        return ArmState(arm_id=self.arm_id, pose=Pose.identity())

    def send_motion(self, command: MotionCommand) -> None:
        if self.interface is None:
            raise RuntimeError("FrankaArmController.connect() must be called before send_motion()")
        if not command.enabled:
            self.stop_motion()
            return
        action = np.concatenate([command.delta_position, command.delta_rotation, np.array([0.0])]).tolist()
        self.interface.control(
            controller_type="OSC_POSE",
            action=action,
            controller_cfg=self.controller_cfg,
            termination=False,
        )

    def stop_motion(self) -> None:
        if self.interface is None:
            return
        self.interface.control(
            controller_type="OSC_POSE",
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            controller_cfg=self.controller_cfg,
            termination=True,
        )

    def close(self) -> None:
        if self.interface is not None and hasattr(self.interface, "close"):
            self.interface.close()


class FrankaGripperController:
    def __init__(self, gripper_id: str, *, interface: Any) -> None:
        self.gripper_id = gripper_id
        self.interface = interface
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def send_gripper(self, command: GripperCommand) -> None:
        action = 1.0 if command.normalized >= 0.5 else -1.0
        self.interface.gripper_control(action)

    def close(self) -> None:
        pass
```

Modify `dex_teleop/robots/__init__.py`:

```python
from dex_teleop.robots.arm import ArmController
from dex_teleop.robots.franka import FrankaArmController, FrankaGripperController
from dex_teleop.robots.gripper import GripperController
from dex_teleop.robots.mock import MockArmController, MockGripperController

__all__ = [
    "ArmController",
    "FrankaArmController",
    "FrankaGripperController",
    "GripperController",
    "MockArmController",
    "MockGripperController",
]
```

- [ ] **Step 4: Run robot tests and verify they pass**

Run:

```bash
pytest tests/robots/test_mock_robot.py tests/robots/test_franka_adapter.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit robot adapters**

```bash
git add dex_teleop/robots tests/robots/test_mock_robot.py tests/robots/test_franka_adapter.py
git commit -m "feat: add franka and mock robot controllers"
```

---

### Task 7: App loop and CLI entry point

**Files:**
- Create: `dex_teleop/core/loop.py`
- Create: `dex_teleop/apps/oculus_franka.py`
- Test: `tests/apps/test_oculus_franka_app.py`

- [ ] **Step 1: Write failing app smoke tests**

Create `tests/apps/test_oculus_franka_app.py`:

```python
from __future__ import annotations

from dex_teleop.apps.oculus_franka import main


def test_app_dry_run_mock_controller_smoke(capsys):
    code = main(
        [
            "--config",
            "configs/oculus_franka_bimanual.yml",
            "--mock-controller",
            "--dry-run",
            "--duration",
            "0.2",
            "--rate-hz",
            "30",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "dry_run=True" in captured.out
    assert "mock_controller=True" in captured.out
    assert "Teleop shutdown complete" in captured.out
```

- [ ] **Step 2: Run app test and verify it fails**

Run:

```bash
pytest tests/apps/test_oculus_franka_app.py -v
```

Expected: FAIL because app and loop modules are missing.

- [ ] **Step 3: Implement loop helpers and app**

Create `dex_teleop/core/loop.py`:

```python
from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, rate_hz: float) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.period = 1.0 / float(rate_hz)
        self._next_time = time.monotonic()

    def sleep(self) -> None:
        self._next_time += self.period
        remaining = self._next_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self._next_time = time.monotonic()
```

Create `dex_teleop/apps/oculus_franka.py`:

```python
from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from dex_teleop.core.config import ConfigError, apply_cli_overrides, load_config
from dex_teleop.core.loop import RateLimiter
from dex_teleop.devices.mock import MockControllerInput
from dex_teleop.devices.oculus import OculusDevice
from dex_teleop.policies.bimanual_clutch import BimanualClutchPolicy
from dex_teleop.robots.franka import FrankaArmController, FrankaGripperController
from dex_teleop.robots.mock import MockArmController, MockGripperController


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oculus to bimanual Franka teleoperation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Use mock robot outputs")
    parser.add_argument("--real-robot", dest="dry_run", action="store_false", help="Use real robot outputs")
    parser.add_argument("--mock-controller", action="store_true", default=None, help="Use mock controller input")
    parser.add_argument("--real-controller", dest="mock_controller", action="store_false", help="Use real Oculus input")
    parser.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds")
    parser.add_argument("--rate-hz", type=float, default=None, help="Override control-loop rate")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args(argv)


def _make_device(cfg: dict[str, Any]):
    input_cfg = cfg["input"]
    if input_cfg.get("mock_controller", False):
        return MockControllerInput(dt=1.0 / float(cfg["app"]["rate_hz"]))
    return OculusDevice(ip=input_cfg.get("ip"), port=int(input_cfg.get("port", 5555)))


def _make_robots(cfg: dict[str, Any]):
    arms = {}
    grippers = {}
    if cfg["app"].get("dry_run", False):
        for arm_id, arm_cfg in cfg["arms"].items():
            arms[arm_id] = MockArmController(arm_id)
            grippers[arm_cfg["gripper"]] = MockGripperController(arm_cfg["gripper"])
        return arms, grippers

    for arm_id, arm_cfg in cfg["arms"].items():
        arm = FrankaArmController(arm_id, interface_cfg=arm_cfg["interface_cfg"])
        arms[arm_id] = arm
    for gripper_id, gripper_cfg in cfg["grippers"].items():
        arm_id = gripper_cfg["arm"]
        arm = arms[arm_id]
        grippers[gripper_id] = FrankaGripperController(gripper_id, interface=arm.interface)
    return arms, grippers


def run(cfg: dict[str, Any], duration: float | None = None) -> int:
    logger = logging.getLogger(__name__)
    device = _make_device(cfg)
    arms, grippers = _make_robots(cfg)
    policy = BimanualClutchPolicy.from_config(cfg)
    rate = RateLimiter(float(cfg["app"]["rate_hz"]))
    deadline = None if duration is None else time.monotonic() + float(duration)

    print(f"Starting teleop dry_run={cfg['app']['dry_run']} mock_controller={cfg['input']['mock_controller']}")
    try:
        device.start()
        for arm in arms.values():
            arm.connect()
        for gripper in grippers.values():
            gripper.connect()

        while deadline is None or time.monotonic() < deadline:
            state = device.read()
            batch = policy.update(state)
            for arm_id, command in batch.motions.items():
                if command.enabled:
                    arms[arm_id].send_motion(command)
                else:
                    arms[arm_id].stop_motion()
            for gripper_id, command in batch.grippers.items():
                grippers[gripper_id].send_gripper(command)
            rate.sleep()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        for arm in arms.values():
            arm.stop_motion()
        for gripper in grippers.values():
            gripper.close()
        for arm in arms.values():
            arm.close()
        device.stop()
        print("Teleop shutdown complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    try:
        cfg = load_config(args.config)
        cfg = apply_cli_overrides(
            cfg,
            dry_run=args.dry_run,
            mock_controller=args.mock_controller,
            rate_hz=args.rate_hz,
        )
        return run(cfg, duration=args.duration)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2
    except Exception as exc:
        print(f"Teleop error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run app test and verify it passes**

Run:

```bash
pytest tests/apps/test_oculus_franka_app.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full no-hardware smoke command**

Run:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --mock-controller \
  --dry-run \
  --duration 2
```

Expected output includes:

```text
Starting teleop dry_run=True mock_controller=True
Teleop shutdown complete
```

- [ ] **Step 6: Commit app entry point**

```bash
git add dex_teleop/core/loop.py dex_teleop/apps/oculus_franka.py tests/apps/test_oculus_franka_app.py
git commit -m "feat: add oculus franka teleop app"
```

---

### Task 8: Fix real Franka gripper sharing after integration test

**Files:**
- Modify: `dex_teleop/apps/oculus_franka.py`
- Modify: `dex_teleop/robots/franka.py`
- Test: `tests/apps/test_oculus_franka_app.py`
- Test: `tests/robots/test_franka_adapter.py`

- [ ] **Step 1: Write a failing test that exposes real gripper interface sharing**

Append to `tests/robots/test_franka_adapter.py`:

```python

def test_franka_gripper_requires_connected_interface_object():
    fake = FakeFrankaInterface()
    arm = FrankaArmController("left", interface_cfg="unused.yml", interface=fake, controller_cfg={})
    arm.connect()
    gripper = FrankaGripperController("left_gripper", interface=arm.interface)
    gripper.connect()
    gripper.send_gripper(GripperCommand("left_gripper", 0.9))

    assert fake.gripper_calls == [1.0]
```

- [ ] **Step 2: Run the targeted test**

Run:

```bash
pytest tests/robots/test_franka_adapter.py::test_franka_gripper_requires_connected_interface_object -v
```

Expected: PASS if Task 6 already handles connected interface objects; if it fails, the failure should show `NoneType` interface usage.

- [ ] **Step 3: Ensure app connects arms before creating real grippers**

If the test in Step 2 passed, still adjust `_make_robots` so real grippers are created after arm connection in `run`. Replace `_make_robots` and the setup section in `run` with this behavior:

```python
def _make_arms(cfg: dict[str, Any]):
    arms = {}
    if cfg["app"].get("dry_run", False):
        for arm_id in cfg["arms"]:
            arms[arm_id] = MockArmController(arm_id)
        return arms
    for arm_id, arm_cfg in cfg["arms"].items():
        arms[arm_id] = FrankaArmController(arm_id, interface_cfg=arm_cfg["interface_cfg"])
    return arms


def _make_grippers(cfg: dict[str, Any], arms: dict[str, Any]):
    grippers = {}
    if cfg["app"].get("dry_run", False):
        for arm_cfg in cfg["arms"].values():
            grippers[arm_cfg["gripper"]] = MockGripperController(arm_cfg["gripper"])
        return grippers
    for gripper_id, gripper_cfg in cfg["grippers"].items():
        arm = arms[gripper_cfg["arm"]]
        grippers[gripper_id] = FrankaGripperController(gripper_id, interface=arm.interface)
    return grippers
```

In `run`, create arms first, connect them, then create and connect grippers:

```python
arms = _make_arms(cfg)
grippers = {}
policy = BimanualClutchPolicy.from_config(cfg)
rate = RateLimiter(float(cfg["app"]["rate_hz"]))
deadline = None if duration is None else time.monotonic() + float(duration)
device.start()
for arm in arms.values():
    arm.connect()
grippers = _make_grippers(cfg, arms)
for gripper in grippers.values():
    gripper.connect()
```

Initialize `grippers = {}` before the `try` block so `finally` can close it even if startup fails.

- [ ] **Step 4: Run app and robot tests**

Run:

```bash
pytest tests/apps/test_oculus_franka_app.py tests/robots/test_franka_adapter.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit gripper sharing fix**

```bash
git add dex_teleop/apps/oculus_franka.py dex_teleop/robots/franka.py tests/apps/test_oculus_franka_app.py tests/robots/test_franka_adapter.py
git commit -m "fix: create franka grippers after arm connection"
```

---

### Task 9: Documentation and full verification

**Files:**
- Modify: `README.md`
- Test: all tests

- [ ] **Step 1: Update README usage section**

Append this section to `README.md`:

```markdown
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
```

- [ ] **Step 2: Run all unit tests**

Run:

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run dry-run smoke command**

Run:

```bash
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --mock-controller \
  --dry-run \
  --duration 2
```

Expected output includes:

```text
Starting teleop dry_run=True mock_controller=True
Teleop shutdown complete
```

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only README changes if Step 1 has not been committed; otherwise clean.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md
git commit -m "docs: add oculus franka teleop usage"
```

---

## Final verification checklist

Run these commands in the `dex-teleop` conda environment:

```bash
pytest -v
python -m dex_teleop.apps.oculus_franka \
  --config configs/oculus_franka_bimanual.yml \
  --mock-controller \
  --dry-run \
  --duration 2
git status --short
```

Expected:

- `pytest -v` passes.
- dry-run smoke command prints startup and shutdown messages.
- `git status --short` is clean after all commits.

## Spec coverage self-review

- Core data types and interfaces: Tasks 1, 5, and 6.
- Config loading and bimanual YAML: Task 3.
- Axis mapping, scaling, deadzone, limits: Task 2.
- Bimanual grip clutch and trigger gripper policy: Task 4.
- Mock controller and dry-run robot modes: Tasks 5, 6, and 7.
- Oculus and Franka real adapters: Tasks 5 and 6.
- App entry point and CLI overrides: Task 7.
- Deoxys backend remains manually managed: Task 9 documentation and no backend process code in app.
- Safety behavior: Tasks 4, 6, and 7 implement grip gating, disabled motion, termination stop, and Ctrl-C shutdown.
- Tests and smoke path: Tasks 1 through 9.
