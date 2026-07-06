from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from dex_teleop.core.config import (
    ConfigError,
    apply_cli_overrides,
    load_config,
    resolve_repo_path,
)
from dex_teleop.core.loop import RateLimiter
from dex_teleop.devices.mock import MockControllerInput
from dex_teleop.devices.oculus import OculusDevice
from dex_teleop.policies.bimanual_clutch import BimanualClutchPolicy
from dex_teleop.robots.franka import FrankaArmController, FrankaGripperController
from dex_teleop.robots.mock import MockArmController, MockGripperController

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oculus to bimanual Franka teleoperation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="Use mock robot outputs",
    )
    parser.add_argument(
        "--real-robot",
        dest="dry_run",
        action="store_false",
        help="Use real robot outputs",
    )
    parser.add_argument(
        "--mock-controller",
        dest="mock_controller",
        action="store_true",
        default=None,
        help="Use mock controller input",
    )
    parser.add_argument(
        "--real-controller",
        dest="mock_controller",
        action="store_false",
        help="Use real Oculus input",
    )
    parser.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds")
    parser.add_argument("--rate-hz", type=float, default=None, help="Override control-loop rate")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args(argv)


def _make_device(cfg: dict[str, Any]) -> MockControllerInput | OculusDevice:
    input_cfg = cfg["input"]
    if input_cfg.get("mock_controller", False):
        return MockControllerInput(dt=1.0 / float(cfg["app"]["rate_hz"]))
    return OculusDevice(ip=input_cfg.get("ip"), port=int(input_cfg.get("port", 5555)))


def _is_real_robot(cfg: dict[str, Any]) -> bool:
    return not bool(cfg["app"].get("dry_run", False))


def _resolve_real_interface_cfg(arm_id: str, arm_cfg: dict[str, Any]) -> str:
    interface_cfg = arm_cfg["interface_cfg"]
    resolved = resolve_repo_path(interface_cfg)
    if not resolved.exists():
        raise RuntimeError(
            f"Real arm {arm_id!r} interface_cfg does not exist: {interface_cfg!r} "
            f"(resolved to {resolved}). Check the Franka interface config path "
            "or run with --dry-run to avoid real hardware."
        )
    return str(resolved)


def _validate_real_hardware_config(cfg: dict[str, Any]) -> None:
    if not _is_real_robot(cfg):
        return

    for arm_id, arm_cfg in cfg["arms"].items():
        arm_type = arm_cfg.get("type")
        if arm_type != "franka":
            raise RuntimeError(
                f"Unsupported real arm type for arm {arm_id!r}: {arm_type!r}. "
                "Real robot mode currently supports only type 'franka'; "
                "use --dry-run for mock hardware."
            )
        _resolve_real_interface_cfg(arm_id, arm_cfg)

    for gripper_id, gripper_cfg in cfg["grippers"].items():
        gripper_type = gripper_cfg.get("type")
        if gripper_type != "franka":
            raise RuntimeError(
                f"Unsupported real gripper type for gripper {gripper_id!r}: "
                f"{gripper_type!r}. Real robot mode currently supports only "
                "type 'franka'; use --dry-run for mock hardware."
            )


def _make_arms(cfg: dict[str, Any]) -> dict[str, MockArmController | FrankaArmController]:
    arms: dict[str, MockArmController | FrankaArmController] = {}
    for arm_id, arm_cfg in cfg["arms"].items():
        if cfg["app"].get("dry_run", False):
            arms[arm_id] = MockArmController(arm_id)
        else:
            arm_type = arm_cfg.get("type")
            if arm_type != "franka":
                raise RuntimeError(
                    f"Unsupported real arm type for arm {arm_id!r}: {arm_type!r}. "
                    "Real robot mode currently supports only type 'franka'; "
                    "use --dry-run for mock hardware."
                )
            arms[arm_id] = FrankaArmController(
                arm_id,
                interface_cfg=_resolve_real_interface_cfg(arm_id, arm_cfg),
            )
    return arms


def _make_grippers(
    cfg: dict[str, Any],
    arms: dict[str, MockArmController | FrankaArmController],
) -> dict[str, MockGripperController | FrankaGripperController]:
    grippers: dict[str, MockGripperController | FrankaGripperController] = {}
    for gripper_id, gripper_cfg in cfg["grippers"].items():
        if cfg["app"].get("dry_run", False):
            grippers[gripper_id] = MockGripperController(gripper_id)
        else:
            gripper_type = gripper_cfg.get("type")
            if gripper_type != "franka":
                raise RuntimeError(
                    f"Unsupported real gripper type for gripper {gripper_id!r}: "
                    f"{gripper_type!r}. Real robot mode currently supports only "
                    "type 'franka'; use --dry-run for mock hardware."
                )
            arm = arms[gripper_cfg["arm"]]
            if not isinstance(arm, FrankaArmController):
                raise RuntimeError(f"Real gripper {gripper_id} requires a Franka arm")
            grippers[gripper_id] = FrankaGripperController(gripper_id, interface=arm.interface)
    return grippers


def _safe_call(label: str, func: Any) -> None:
    try:
        func()
    except Exception as exc:  # noqa: BLE001 - cleanup should continue best-effort
        logger.warning("Ignoring %s failure during shutdown: %s", label, exc)


def run(cfg: dict[str, Any], duration: float | None = None) -> int:
    _validate_real_hardware_config(cfg)
    device = _make_device(cfg)
    arms = _make_arms(cfg)
    grippers: dict[str, MockGripperController | FrankaGripperController] = {}
    policy = BimanualClutchPolicy.from_config(cfg)
    rate = RateLimiter(float(cfg["app"]["rate_hz"]))
    deadline = None if duration is None else time.monotonic() + float(duration)

    print(
        f"Starting teleop dry_run={cfg['app']['dry_run']} "
        f"mock_controller={cfg['input']['mock_controller']}"
    )
    try:
        try:
            device.start()
        except Exception as exc:
            if not cfg["input"].get("mock_controller", False):
                raise RuntimeError(
                    "Failed to start real Oculus controller. Check ADB connectivity, "
                    "confirm the Quest teleop app is running, or pass "
                    "--mock-controller for diagnostics without the headset. "
                    f"Original error: {exc}"
                ) from exc
            raise

        for arm_id, arm in arms.items():
            try:
                arm.connect()
            except Exception as exc:
                if _is_real_robot(cfg):
                    raise RuntimeError(
                        f"Failed to connect Franka arm {arm_id!r}. Ensure the "
                        "manually started Deoxys backend for this robot is running "
                        "and reachable before launching teleop. "
                        f"Original error: {exc}"
                    ) from exc
                raise
        arm_moving = {arm_id: False for arm_id in arms}
        grippers = _make_grippers(cfg, arms)
        for gripper_id, gripper in grippers.items():
            try:
                gripper.connect()
            except Exception as exc:
                if _is_real_robot(cfg):
                    raise RuntimeError(
                        f"Failed to connect Franka gripper {gripper_id!r}. Ensure "
                        "the manually started Deoxys backend for the attached arm "
                        "is running and reachable before launching teleop. "
                        f"Original error: {exc}"
                    ) from exc
                raise

        while deadline is None or time.monotonic() < deadline:
            state = device.read()
            batch = policy.update(state)
            for arm_id, command in batch.motions.items():
                arm = arms[arm_id]
                if command.enabled:
                    arm.send_motion(command)
                    arm_moving[arm_id] = True
                else:
                    if arm_moving.get(arm_id, False):
                        arm.stop_motion()
                        arm_moving[arm_id] = False
            for gripper_id, command in batch.grippers.items():
                grippers[gripper_id].send_gripper(command)
            rate.sleep()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        for arm_id, arm in arms.items():
            _safe_call(f"stop_motion({arm_id})", arm.stop_motion)
        for gripper_id, gripper in grippers.items():
            _safe_call(f"close({gripper_id})", gripper.close)
        for arm_id, arm in arms.items():
            _safe_call(f"close({arm_id})", arm.close)
        _safe_call("device.stop", device.stop)
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
    except Exception as exc:  # noqa: BLE001 - CLI entry point should return an error code
        print(f"Teleop error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
