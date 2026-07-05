from __future__ import annotations

from pathlib import Path

import pytest

from dex_teleop.core.config import ConfigError, apply_cli_overrides, load_config


_VALID_CONFIG = """
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
    interface_cfg: dependency/deoxys_control/deoxys/config/franka-two.yml
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
    assert cfg["app"]["dry_run"] is True
    assert cfg["input"]["mock_controller"] is True
    assert cfg["app"]["rate_hz"] == 30


def test_unknown_gripper_reference_reports_arm_path(tmp_path: Path):
    path = tmp_path / "bad.yml"
    path.write_text(_VALID_CONFIG.replace("gripper: left_gripper", "gripper: missing_gripper"))

    with pytest.raises(ConfigError, match="arms.left.gripper.*missing_gripper"):
        load_config(path)


def test_unknown_gripper_arm_reference_reports_gripper_path(tmp_path: Path):
    path = tmp_path / "bad.yml"
    path.write_text(_VALID_CONFIG.replace("arm: left", "arm: missing_arm"))

    with pytest.raises(ConfigError, match="grippers.left_gripper.arm.*missing_arm"):
        load_config(path)


def test_mismatched_reciprocal_arm_gripper_ownership_reports_paths(tmp_path: Path):
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
    interface_cfg: dependency/deoxys_control/deoxys/config/franka-two.yml
    controller: left
    gripper: left_gripper
  right:
    type: franka
    interface_cfg: dependency/deoxys_control/deoxys/config/franka-two.yml
    controller: right
    gripper: right_gripper
grippers:
  left_gripper:
    type: franka
    arm: right
  right_gripper:
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

    with pytest.raises(
        ConfigError,
        match=r"arms\.left\.gripper.*left_gripper.*grippers\.left_gripper\.arm.*right.*left",
    ):
        load_config(path)


def test_non_mapping_yaml_root_raises_config_error(tmp_path: Path):
    path = tmp_path / "bad.yml"
    path.write_text("- app\n- input\n")

    with pytest.raises(ConfigError, match="Config root must be a mapping"):
        load_config(path)
