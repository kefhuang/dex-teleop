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


def _make_arms(cfg: dict[str, Any]) -> dict[str, MockArmController | FrankaArmController]:
    arms: dict[str, MockArmController | FrankaArmController] = {}
    for arm_id, arm_cfg in cfg["arms"].items():
        if cfg["app"].get("dry_run", False):
            arms[arm_id] = MockArmController(arm_id)
        else:
            arms[arm_id] = FrankaArmController(arm_id, interface_cfg=arm_cfg["interface_cfg"])
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
        device.start()
        for arm in arms.values():
            arm.connect()
        grippers = _make_grippers(cfg, arms)
        for gripper in grippers.values():
            gripper.connect()

        while deadline is None or time.monotonic() < deadline:
            state = device.read()
            batch = policy.update(state)
            for arm_id, command in batch.motions.items():
                arm = arms[arm_id]
                if command.enabled:
                    arm.send_motion(command)
                else:
                    arm.stop_motion()
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
