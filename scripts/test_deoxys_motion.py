#!/usr/bin/env python3
"""Small-motion Deoxys control diagnostic for one Franka arm.

Reads the current end-effector pose, then optionally sends 2 cm OSC_POSE delta
motions in robot-base axes: up/down, left/right, forward/back.

Safety: by default this script is dry-run/read-only. Pass --execute to move the
robot. Keep a hand near the Franka stop and verify the selected config matches
the intended robot before using --execute.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEOXYS_ROOT = REPO_ROOT / "dependency" / "deoxys_control" / "deoxys"
DEFAULT_CONFIG = "franka-three.yml"
CONTROLLER_TYPE = "OSC_POSE"
MAX_STEP_METERS = 0.02

# Keep generated protobuf import compatible with this conda env.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_DIRECTION_ALIASES = {
    "up": "up",
    "u": "up",
    "down": "down",
    "d": "down",
    "left": "left",
    "l": "left",
    "right": "right",
    "r": "right",
    "forward": "forward",
    "front": "forward",
    "f": "forward",
    "back": "back",
    "backward": "back",
    "b": "back",
}


def build_direction_deltas(step: float) -> dict[str, np.ndarray]:
    """Return direction -> xyz delta in robot-base frame."""

    step = float(step)
    return {
        "up": np.array([0.0, 0.0, step]),
        "down": np.array([0.0, 0.0, -step]),
        "left": np.array([0.0, step, 0.0]),
        "right": np.array([0.0, -step, 0.0]),
        "forward": np.array([step, 0.0, 0.0]),
        "back": np.array([-step, 0.0, 0.0]),
    }


def clip_delta_norm(delta: np.ndarray, max_norm: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm == 0.0 or norm <= float(max_norm):
        return delta.copy()
    return delta * (float(max_norm) / norm)


def build_motion_action(delta_position: np.ndarray) -> list[float]:
    """Build a 7D OSC_POSE action: xyz, rpy, neutral gripper slot."""

    delta = np.asarray(delta_position, dtype=float)
    if delta.shape != (3,):
        raise ValueError(f"delta_position must be shape (3,), got {delta.shape}")
    return [float(delta[0]), float(delta[1]), float(delta[2]), 0.0, 0.0, 0.0, 0.0]


def parse_direction_sequence(text: str) -> list[str]:
    directions: list[str] = []
    for raw_item in text.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item not in _DIRECTION_ALIASES:
            valid = ", ".join(sorted(set(_DIRECTION_ALIASES.values())))
            raise ValueError(f"Unknown direction {raw_item!r}. Valid directions: {valid}")
        directions.append(_DIRECTION_ALIASES[item])
    if not directions:
        raise ValueError("At least one direction is required")
    return directions


def resolve_config(config: str) -> Path:
    path = Path(config)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return DEOXYS_ROOT / "config" / config


def import_deoxys() -> tuple[type[Any], Any]:
    if not DEOXYS_ROOT.exists():
        raise FileNotFoundError(f"Missing Deoxys dependency: {DEOXYS_ROOT}")
    sys.path.insert(0, str(DEOXYS_ROOT))
    from deoxys.franka_interface import FrankaInterface  # noqa: PLC0415
    from deoxys.utils.config_utils import get_default_controller_config  # noqa: PLC0415

    return FrankaInterface, get_default_controller_config


def wait_for_state(robot: Any, *, timeout: float, label: str) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(robot, "state_buffer_size", 0):
            return True
        time.sleep(0.05)
    print(f"[WARN] Timed out waiting for state after {timeout:.1f}s: {label}")
    return False


def current_eef_pos(robot: Any) -> np.ndarray | None:
    if not getattr(robot, "state_buffer_size", 0):
        return None
    state = robot._state_buffer[-1]
    if len(state.O_T_EE) != 16:
        return None
    eef = np.array(state.O_T_EE).reshape(4, 4).T
    return eef[:3, 3].astype(float)


def scaled_action_for_desired_delta(delta: np.ndarray, controller_cfg: Any) -> list[float]:
    """Invert Deoxys action_scale so the commanded OSC delta is in meters."""

    scale = float(controller_cfg.action_scale.translation)
    if scale <= 0.0:
        raise ValueError(f"Invalid controller action_scale.translation: {scale}")
    return build_motion_action(np.asarray(delta, dtype=float) / scale)


def send_delta(robot: Any, controller_cfg: Any, delta: np.ndarray) -> None:
    action = scaled_action_for_desired_delta(delta, controller_cfg)
    robot.control(
        controller_type=CONTROLLER_TYPE,
        action=action,
        controller_cfg=controller_cfg,
        termination=False,
    )


def move_toward_target(
    robot: Any,
    controller_cfg: Any,
    target_pos: np.ndarray,
    *,
    max_step: float,
    tolerance: float,
    cycles: int,
) -> tuple[np.ndarray | None, int, float | None]:
    last_pos: np.ndarray | None = None
    last_error: float | None = None
    for cycle in range(1, cycles + 1):
        current = current_eef_pos(robot)
        if current is None:
            time.sleep(0.05)
            continue
        last_pos = current
        remaining = np.asarray(target_pos, dtype=float) - current
        last_error = float(np.linalg.norm(remaining))
        if last_error <= tolerance:
            return current, cycle - 1, last_error
        command_delta = clip_delta_norm(remaining, max_step)
        send_delta(robot, controller_cfg, command_delta)
    return last_pos, cycles, last_error


def stop_robot(robot: Any, controller_cfg: Any) -> None:
    robot.control(
        controller_type=CONTROLLER_TYPE,
        action=[0.0] * 7,
        controller_cfg=controller_cfg,
        termination=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move one Franka arm by small 2 cm Deoxys OSC_POSE deltas.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Deoxys config path/name. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--directions",
        default="up,down,left,right,forward,back",
        help="Comma-separated directions. Default: up,down,left,right,forward,back",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=MAX_STEP_METERS,
        help="Step size in meters. Default/max: 0.02",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="Seconds to wait after each motion before reading pose. Default: 1.0",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for initial robot state. Default: 5.0",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=60,
        help="Max closed-loop control cycles per direction. Default: 60",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.003,
        help="Target position tolerance in meters. Default: 0.003",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send control commands. Without this, the script is read-only.",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Construct FrankaInterface with has_gripper=False.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.step <= 0.0:
        print("[ERROR] --step must be positive", file=sys.stderr)
        return 2
    if args.step > MAX_STEP_METERS:
        print("[ERROR] --step is capped at 0.02 m for this safety diagnostic", file=sys.stderr)
        return 2
    if args.settle < 0.0:
        print("[ERROR] --settle must be non-negative", file=sys.stderr)
        return 2
    if args.cycles <= 0:
        print("[ERROR] --cycles must be positive", file=sys.stderr)
        return 2
    if args.tolerance <= 0.0:
        print("[ERROR] --tolerance must be positive", file=sys.stderr)
        return 2
    if args.state_timeout <= 0.0:
        print("[ERROR] --state-timeout must be positive", file=sys.stderr)
        return 2

    try:
        directions = parse_direction_sequence(args.directions)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    cfg_path = resolve_config(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Missing config: {cfg_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Directions: {directions}")
    print(f"[INFO] Step: {args.step:.3f} m")
    if not args.execute:
        print("[DRY-RUN] Not sending robot commands. Re-run with --execute to move.")

    robot = None
    try:
        FrankaInterface, get_default_controller_config = import_deoxys()
        controller_cfg = get_default_controller_config(CONTROLLER_TYPE)
        robot = FrankaInterface(
            str(cfg_path),
            use_visualizer=False,
            has_gripper=not args.no_gripper,
            automatic_gripper_reset=False,
        )

        if not wait_for_state(robot, timeout=args.state_timeout, label="initial"):
            return 1
        start_pos = current_eef_pos(robot)
        if start_pos is None:
            print("[ERROR] Could not read current end-effector position", file=sys.stderr)
            return 1
        print(f"[STATE] start_eef_pos={np.round(start_pos, 5).tolist()}")

        deltas = build_direction_deltas(args.step)
        previous_pos = start_pos
        for index, direction in enumerate(directions, start=1):
            delta = deltas[direction]
            action = scaled_action_for_desired_delta(delta, controller_cfg)
            print(
                f"[PLAN] {index}/{len(directions)} {direction}: "
                f"desired_delta={np.round(delta, 5).tolist()} action={np.round(action, 5).tolist()}"
            )
            if not args.execute:
                continue

            before = current_eef_pos(robot)
            print(f"[STATE] before_{direction}={np.round(before, 5).tolist() if before is not None else None}")
            if before is None:
                print(f"[WARN] Could not read pose before {direction}")
                continue
            target = before + delta
            after, used_cycles, final_error = move_toward_target(
                robot,
                controller_cfg,
                target,
                max_step=args.step,
                tolerance=args.tolerance,
                cycles=args.cycles,
            )
            if args.settle:
                time.sleep(args.settle)
            wait_for_state(robot, timeout=1.0, label=f"after {direction}")
            after = current_eef_pos(robot) if after is None else after
            if after is None:
                print(f"[WARN] Could not read pose after {direction}")
                previous_pos = after
                continue
            measured_delta = after - before
            print(
                f"[STATE] target_{direction}={np.round(target, 5).tolist()} "
                f"after_{direction}={np.round(after, 5).tolist()} "
                f"measured_delta={np.round(measured_delta, 5).tolist()} "
                f"cycles={used_cycles} final_error={None if final_error is None else round(final_error, 5)}"
            )
            previous_pos = after

        if args.execute:
            stop_robot(robot, controller_cfg)
            time.sleep(0.2)
            final_pos = current_eef_pos(robot)
            if final_pos is not None:
                print(
                    f"[STATE] final_eef_pos={np.round(final_pos, 5).tolist()} "
                    f"net_delta={np.round(final_pos - start_pos, 5).tolist()}"
                )
        print("[DONE]")
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001 - diagnostic should show hardware failures clearly.
        print(f"[ERROR] Motion diagnostic failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
