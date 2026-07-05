from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a teleoperation config file is invalid."""


REQUIRED_TOP_LEVEL = (
    "app",
    "input",
    "controls",
    "arms",
    "grippers",
    "mapping",
    "scales",
    "safety",
)

REQUIRED_CONTROLS = ("clutch_button", "gripper_axis")
REQUIRED_MAPPING = ("position_axes", "rotation_axes")
REQUIRED_SCALES = ("position", "rotation")
REQUIRED_SAFETY = (
    "max_translation_per_cycle",
    "max_rotation_per_cycle",
    "position_deadzone",
    "rotation_deadzone",
    "send_zero_when_idle",
)


def _format_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _require(mapping: dict[str, Any], key: str, path: str = "") -> Any:
    field_path = _format_path(path, key)
    if key not in mapping:
        raise ConfigError(f"Missing required config field: {field_path}")
    return mapping[key]


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Config field must be a mapping: {path}")
    return value


def _validate_required_keys(mapping: dict[str, Any], keys: tuple[str, ...], path: str) -> None:
    for key in keys:
        _require(mapping, key, path)


def _validate(cfg: dict[str, Any]) -> None:
    for section in REQUIRED_TOP_LEVEL:
        _require(cfg, section)

    app = _require_mapping(cfg["app"], "app")
    input_cfg = _require_mapping(cfg["input"], "input")
    controls = _require_mapping(cfg["controls"], "controls")
    arms = _require_mapping(cfg["arms"], "arms")
    grippers = _require_mapping(cfg["grippers"], "grippers")
    mapping = _require_mapping(cfg["mapping"], "mapping")
    scales = _require_mapping(cfg["scales"], "scales")
    safety = _require_mapping(cfg["safety"], "safety")

    _validate_required_keys(app, ("rate_hz", "dry_run"), "app")
    _validate_required_keys(input_cfg, ("type", "mock_controller"), "input")
    _validate_required_keys(controls, REQUIRED_CONTROLS, "controls")
    _validate_required_keys(mapping, REQUIRED_MAPPING, "mapping")
    _validate_required_keys(scales, REQUIRED_SCALES, "scales")
    _validate_required_keys(safety, REQUIRED_SAFETY, "safety")

    _require_mapping(mapping["position_axes"], "mapping.position_axes")
    _require_mapping(mapping["rotation_axes"], "mapping.rotation_axes")

    for arm_id, arm_cfg_value in arms.items():
        arm_path = f"arms.{arm_id}"
        arm_cfg = _require_mapping(arm_cfg_value, arm_path)
        for key in ("type", "interface_cfg", "controller", "gripper"):
            _require(arm_cfg, key, arm_path)
        gripper_id = arm_cfg["gripper"]
        if gripper_id not in grippers:
            raise ConfigError(
                f"{arm_path}.gripper references unknown gripper {gripper_id!r}"
            )

    for gripper_id, gripper_cfg_value in grippers.items():
        gripper_path = f"grippers.{gripper_id}"
        gripper_cfg = _require_mapping(gripper_cfg_value, gripper_path)
        for key in ("type", "arm"):
            _require(gripper_cfg, key, gripper_path)
        arm_id = gripper_cfg["arm"]
        if arm_id not in arms:
            raise ConfigError(
                f"{gripper_path}.arm references unknown arm {arm_id!r}"
            )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a teleoperation YAML config file."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

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
    """Return a deep-copied config with optional CLI overrides applied."""

    updated = copy.deepcopy(cfg)
    if dry_run is not None:
        updated["app"]["dry_run"] = bool(dry_run)
    if mock_controller is not None:
        updated["input"]["mock_controller"] = bool(mock_controller)
    if rate_hz is not None:
        updated["app"]["rate_hz"] = float(rate_hz)
    return updated


def resolve_repo_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a config path relative to the repository root unless absolute."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / candidate
