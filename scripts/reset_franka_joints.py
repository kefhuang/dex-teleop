#!/usr/bin/env python3
"""Reset one Franka arm to the repository's default joint home pose.

Safety: this script sends JOINT_POSITION commands. Verify that ``--config``
matches the intended robot and keep a hand near the Franka stop.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEOXYS_ROOT = REPO_ROOT / "dependency" / "deoxys_control" / "deoxys"
DEFAULT_CONFIG = "franka-three.yml"
CONTROLLER_TYPE = "JOINT_POSITION"
DEFAULT_TIMEOUT_SECONDS = 7.0
DEFAULT_TOLERANCE = 1e-3

# Keep generated protobuf import compatible with this conda env.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_teleop.robots.franka_constants import FRANKA_HOME_JOINTS  # noqa: E402

# Default Deoxys reset/home pose.
RESET_JOINTS = FRANKA_HOME_JOINTS.tolist()


def resolve_config(config: str) -> Path:
    path = Path(config)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return DEOXYS_ROOT / "config" / config


def import_deoxys() -> tuple[type[Any], Callable[[str], Any]]:
    if not DEOXYS_ROOT.exists():
        raise FileNotFoundError(f"Missing Deoxys dependency: {DEOXYS_ROOT}")
    sys.path.insert(0, str(DEOXYS_ROOT))

    from deoxys.franka_interface import FrankaInterface  # noqa: PLC0415
    from deoxys.utils.config_utils import get_default_controller_config  # noqa: PLC0415

    return FrankaInterface, get_default_controller_config


def build_reset_action(target_joints: Sequence[float]) -> list[float]:
    joints = [float(joint) for joint in target_joints]
    if len(joints) != 7:
        raise ValueError(f"Expected 7 target joints, got {len(joints)}")
    return joints + [-1.0]


def current_joint_error(robot: Any, target_joints: Sequence[float]) -> float | None:
    current_q = getattr(robot, "last_q", None)
    if current_q is None:
        return None
    return float(
        np.max(
            np.abs(
                np.asarray(current_q, dtype=float)
                - np.asarray(target_joints, dtype=float)
            )
        )
    )


def reset_robot_joints(
    robot: Any,
    *,
    controller_cfg: Any,
    target_joints: Sequence[float] = RESET_JOINTS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    tolerance: float = DEFAULT_TOLERANCE,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> bool:
    action = build_reset_action(target_joints)
    start_time = monotonic_fn()

    while monotonic_fn() - start_time <= float(timeout):
        if getattr(robot, "state_buffer_size", 0):
            error = current_joint_error(robot, target_joints)
            if error is not None and error < float(tolerance):
                return True

        robot.control(
            controller_type=CONTROLLER_TYPE,
            action=action,
            controller_cfg=controller_cfg,
            termination=False,
        )
        sleep_fn(0.01)

    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset one Franka arm to a fixed JOINT_POSITION home pose.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Deoxys interface config path/name. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Seconds before giving up. Default: {DEFAULT_TIMEOUT_SECONDS:g}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config(args.config)
    FrankaInterface, get_default_controller_config = import_deoxys()

    robot = FrankaInterface(
        config_path,
        has_gripper=False,
        use_visualizer=False,
    )
    try:
        controller_cfg = get_default_controller_config(CONTROLLER_TYPE)
        reached = reset_robot_joints(
            robot,
            controller_cfg=controller_cfg,
            timeout=args.timeout,
        )
        if reached:
            print(f"[INFO] Reset reached target joints for {config_path}")
            return 0
        print(f"[WARN] Timed out resetting joints for {config_path}")
        return 1
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
