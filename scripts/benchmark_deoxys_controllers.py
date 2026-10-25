#!/usr/bin/env python3
"""Benchmark Deoxys controller response for single-axis Cartesian motions.

The script compares Cartesian OSC controllers, Cartesian velocity control, and
IK-to-joint-space execution. It is dry-run by default; pass --execute to move
hardware. Results are written as JSONL trial records plus a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEOXYS_ROOT = REPO_ROOT / "dependency" / "deoxys_control" / "deoxys"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEOXYS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEOXYS_ROOT))

# Keep generated protobuf import compatible with this conda env.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

OSC_CONTROLLERS = {"OSC_POSE", "OSC_POSITION"}
IK_CONTROLLERS = {"IK_JOINT_IMPEDANCE", "IK_JOINT_POSITION"}
SUPPORTED_CONTROLLERS = (
    "OSC_POSE",
    "OSC_POSITION",
    "CARTESIAN_VELOCITY",
    "IK_JOINT_IMPEDANCE",
    "IK_JOINT_POSITION",
)
DEFAULT_CONTROLLERS = "OSC_POSE,OSC_POSITION,CARTESIAN_VELOCITY,IK_JOINT_IMPEDANCE"
DEFAULT_AXES = "+x,-x,+y,-y,+z,-z"
MAX_STEP_M = 0.10
RESET_JOINTS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]


def parse_csv(text: str) -> list[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated item")
    return items


def axis_delta(axis: str, magnitude: float) -> np.ndarray:
    axis = axis.strip().lower()
    sign = 1.0
    if axis.startswith("+"):
        axis = axis[1:]
    elif axis.startswith("-"):
        axis = axis[1:]
        sign = -1.0
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of x, y, z, +x, -x, +y, -y, +z, -z")
    delta = np.zeros(3, dtype=float)
    delta[{"x": 0, "y": 1, "z": 2}[axis]] = sign * float(magnitude)
    return delta


def axis_name(axis: str) -> str:
    axis = axis.strip().lower()
    if axis[0] not in "+-":
        return "+" + axis
    return axis


def main_axis_index(axis: str) -> int:
    stripped = axis_name(axis)[1:]
    return {"x": 0, "y": 1, "z": 2}[stripped]


def _scale_value(controller_cfg: Any, group: str) -> float:
    action_scale = getattr(controller_cfg, "action_scale", None)
    if action_scale is None and isinstance(controller_cfg, dict):
        action_scale = controller_cfg.get("action_scale")
    if action_scale is None:
        return 1.0
    if isinstance(action_scale, dict):
        value = action_scale.get(group, 1.0)
    else:
        value = getattr(action_scale, group, 1.0)
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"Invalid action_scale.{group}: {value}")
    return value


def scaled_cartesian_action(
    translation: np.ndarray,
    rotation: np.ndarray,
    controller_cfg: Any,
) -> list[float]:
    """Invert Deoxys action_scale so caller supplies SI-unit command values."""

    translation = np.asarray(translation, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if translation.shape != (3,):
        raise ValueError(f"translation must be shape (3,), got {translation.shape}")
    if rotation.shape != (3,):
        raise ValueError(f"rotation must be shape (3,), got {rotation.shape}")
    t_scale = _scale_value(controller_cfg, "translation")
    r_scale = _scale_value(controller_cfg, "rotation")
    return [
        float(translation[0] / t_scale),
        float(translation[1] / t_scale),
        float(translation[2] / t_scale),
        float(rotation[0] / r_scale),
        float(rotation[1] / r_scale),
        float(rotation[2] / r_scale),
        0.0,
    ]


def compute_trial_metrics(
    *,
    axis: str,
    requested_delta: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    duration_s: float,
    controller_calls: int,
    ik_solve_s: float | None,
) -> dict[str, Any]:
    requested_delta = np.asarray(requested_delta, dtype=float)
    start_pos = np.asarray(start_pos, dtype=float)
    end_pos = np.asarray(end_pos, dtype=float)
    actual_delta = end_pos - start_pos
    idx = main_axis_index(axis)
    main_displacement = float(actual_delta[idx])
    requested_main = float(requested_delta[idx])
    drift_vec = actual_delta.copy()
    drift_vec[idx] = 0.0
    duration_s = float(duration_s)
    abs_requested = abs(requested_main)
    return {
        "axis": axis_name(axis),
        "requested_delta_m": requested_delta.tolist(),
        "actual_delta_m": actual_delta.tolist(),
        "main_displacement_m": round(main_displacement, 9),
        "abs_main_displacement_m": round(abs(main_displacement), 9),
        "requested_main_m": round(requested_main, 9),
        "drift_m": round(float(np.linalg.norm(drift_vec)), 9),
        "main_speed_m_s": round(main_displacement / duration_s, 9) if duration_s > 0 else None,
        "abs_main_speed_m_s": round(abs(main_displacement) / duration_s, 9) if duration_s > 0 else None,
        "tracking_efficiency": round(main_displacement / requested_main, 9)
        if abs_requested > 1e-12
        else None,
        "controller_calls": int(controller_calls),
        "avg_call_period_s": round(duration_s / controller_calls, 9) if controller_calls else None,
        "ik_solve_s": None if ik_solve_s is None else round(float(ik_solve_s), 9),
    }


def add_symmetry_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["controller"], row["axis"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        axis = row["axis"]
        if axis.startswith("+"):
            opposite = "-" + axis[1:]
        elif axis.startswith("-"):
            opposite = "+" + axis[1:]
        else:
            opposite = "-" + axis
        other = by_key.get((row["controller"], opposite))
        row = dict(row)
        if other is not None:
            a = abs(float(row.get("main_displacement_m", 0.0)))
            b = abs(float(other.get("main_displacement_m", 0.0)))
            smaller = min(a, b)
            larger = max(a, b)
            row["axis_symmetry_ratio"] = (
                round(larger / smaller, 9) if smaller > 1e-12 else None
            )
        else:
            row["axis_symmetry_ratio"] = None
        output.append(row)
    return output


def resolve_config(config: str) -> Path:
    path = Path(config)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return DEOXYS_ROOT / "config" / config


def default_output_paths() -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("logs") / f"controller_benchmark_{timestamp}.jsonl",
        Path("logs") / f"controller_benchmark_{timestamp}_summary.csv",
    )


def import_deoxys() -> tuple[type[Any], Callable[[str], Any], type[Any]]:
    if not DEOXYS_ROOT.exists():
        raise FileNotFoundError(f"Missing Deoxys dependency: {DEOXYS_ROOT}")
    from deoxys.franka_interface import FrankaInterface  # noqa: PLC0415
    from deoxys.utils.config_utils import get_default_controller_config  # noqa: PLC0415
    from deoxys.utils.yaml_config import YamlConfig  # noqa: PLC0415

    return FrankaInterface, get_default_controller_config, YamlConfig


def current_eef_pos(robot: Any) -> np.ndarray | None:
    pose = getattr(robot, "last_eef_pose", None)
    if pose is not None:
        arr = np.asarray(pose, dtype=float)
        if arr.shape == (4, 4):
            return arr[:3, 3].copy()
    if not getattr(robot, "state_buffer_size", 0):
        return None
    state = robot._state_buffer[-1]
    if len(state.O_T_EE) != 16:
        return None
    eef = np.array(state.O_T_EE).reshape(4, 4).T
    return eef[:3, 3].astype(float)


def current_q(robot: Any) -> list[float] | None:
    q = getattr(robot, "last_q", None)
    if q is not None:
        arr = np.asarray(q, dtype=float)
        if arr.shape == (7,):
            return arr.tolist()
    if not getattr(robot, "state_buffer_size", 0):
        return None
    state = robot._state_buffer[-1]
    q = getattr(state, "q", None)
    if q is None or len(q) != 7:
        return None
    return [float(x) for x in q]


def wait_for_state(robot: Any, timeout: float) -> bool:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if current_eef_pos(robot) is not None and current_q(robot) is not None:
            return True
        time.sleep(0.05)
    return False


def control_loop_sleep(rate_hz: float, started_at: float, call_index: int) -> None:
    next_time = started_at + (call_index + 1) / rate_hz
    remaining = next_time - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def load_controller_cfg(controller: str, get_default_controller_config: Callable[[str], Any]) -> tuple[str, Any]:
    if controller in {"IK_JOINT_IMPEDANCE", "JOINT_IMPEDANCE"}:
        return "JOINT_IMPEDANCE", get_default_controller_config("JOINT_IMPEDANCE")
    if controller in {"IK_JOINT_POSITION", "JOINT_POSITION"}:
        return "JOINT_POSITION", get_default_controller_config("JOINT_POSITION")
    return controller, get_default_controller_config(controller)


def run_osc_trial(
    robot: Any,
    controller_type: str,
    controller_cfg: Any,
    requested_delta: np.ndarray,
    *,
    duration: float,
    rate_hz: float,
    max_delta_per_cycle: float,
) -> int:
    start_time = time.monotonic()
    calls = 0
    target = current_eef_pos(robot) + requested_delta
    while time.monotonic() - start_time < duration:
        current = current_eef_pos(robot)
        if current is None:
            break
        remaining = target - current
        norm = float(np.linalg.norm(remaining))
        if norm > max_delta_per_cycle and norm > 0:
            command_delta = remaining * (max_delta_per_cycle / norm)
        else:
            command_delta = remaining
        action = scaled_cartesian_action(command_delta, np.zeros(3), controller_cfg)
        robot.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)
        calls += 1
        control_loop_sleep(rate_hz, start_time, calls)
    return calls


def run_velocity_trial(
    robot: Any,
    controller_cfg: Any,
    requested_delta: np.ndarray,
    *,
    duration: float,
    rate_hz: float,
    max_linear_velocity: float,
) -> int:
    velocity = requested_delta / max(duration, 1e-9)
    speed = float(np.linalg.norm(velocity))
    if speed > max_linear_velocity and speed > 0:
        velocity = velocity * (max_linear_velocity / speed)
    action = scaled_cartesian_action(velocity, np.zeros(3), controller_cfg)
    start_time = time.monotonic()
    calls = 0
    while time.monotonic() - start_time < duration:
        robot.control(
            controller_type="CARTESIAN_VELOCITY",
            action=action,
            controller_cfg=controller_cfg,
        )
        calls += 1
        control_loop_sleep(rate_hz, start_time, calls)
    zero = scaled_cartesian_action(np.zeros(3), np.zeros(3), controller_cfg)
    robot.control(controller_type="CARTESIAN_VELOCITY", action=zero, controller_cfg=controller_cfg)
    return calls


def solve_ik_delta(requested_delta: np.ndarray, start_q: list[float], *, num_points: int) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    start = time.monotonic()
    from deoxys.utils.ik_utils import IKWrapper  # noqa: PLC0415

    wrapper = IKWrapper()
    joint_traj, debug_info = wrapper.ik_trajectory_delta_position(
        requested_delta,
        start_q,
        num_points=num_points,
        verbose=False,
    )
    dense = wrapper.interpolate_dense_traj(joint_traj)
    solve_s = time.monotonic() - start
    debug = {
        "raw_points": len(joint_traj),
        "dense_points": len(dense),
        "debug_keys": sorted(debug_info.keys()),
    }
    return dense, solve_s, debug


def run_joint_trajectory(
    robot: Any,
    controller_type: str,
    controller_cfg: Any,
    joint_traj: list[np.ndarray],
    *,
    duration: float,
    rate_hz: float,
) -> int:
    max_calls = max(1, int(duration * rate_hz))
    if len(joint_traj) <= 1:
        selected = joint_traj
    else:
        indices = np.linspace(0, len(joint_traj) - 1, max_calls).astype(int)
        selected = [joint_traj[i] for i in indices]
    start_time = time.monotonic()
    calls = 0
    for q in selected:
        action = [float(x) for x in q[:7]] + [-1.0]
        robot.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)
        calls += 1
        control_loop_sleep(rate_hz, start_time, calls)
    return calls



def reset_robot_joints(
    robot: Any,
    controller_cfg: Any,
    timeout: float,
    *,
    tolerance: float = 1e-3,
) -> bool:
    deadline = time.monotonic() + float(timeout)
    action = [float(q) for q in RESET_JOINTS] + [-1.0]
    while time.monotonic() < deadline:
        q = current_q(robot)
        if q is not None and float(np.max(np.abs(np.asarray(q) - np.asarray(RESET_JOINTS)))) <= tolerance:
            return True
        robot.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
            termination=False,
        )
    q = current_q(robot)
    return q is not None and float(np.max(np.abs(np.asarray(q) - np.asarray(RESET_JOINTS)))) <= tolerance


def maybe_reset_before_trial(
    robot: Any,
    reset_controller_cfg: Any,
    args: argparse.Namespace,
    *,
    reset_fn: Callable[[Any, Any, float], bool] = reset_robot_joints,
) -> bool:
    if not bool(getattr(args, "reset_between_trials", False)):
        return False
    ok = reset_fn(robot, reset_controller_cfg, float(args.reset_timeout))
    if not ok:
        raise RuntimeError("Reset before trial did not reach target joints")
    return True


def run_trial(
    robot: Any,
    controller: str,
    requested_delta: np.ndarray,
    axis: str,
    get_default_controller_config: Callable[[str], Any],
    *,
    duration: float,
    rate_hz: float,
    max_delta_per_cycle: float,
    max_linear_velocity: float,
    ik_points: int,
    settle: float,
) -> dict[str, Any]:
    controller_type, controller_cfg = load_controller_cfg(controller, get_default_controller_config)
    start_pos = current_eef_pos(robot)
    start_q = current_q(robot)
    if start_pos is None or start_q is None:
        raise RuntimeError("Could not read robot start state")

    ik_solve_s = None
    ik_debug: dict[str, Any] = {}
    started_at = time.monotonic()
    if controller in OSC_CONTROLLERS:
        calls = run_osc_trial(
            robot,
            controller_type,
            controller_cfg,
            requested_delta,
            duration=duration,
            rate_hz=rate_hz,
            max_delta_per_cycle=max_delta_per_cycle,
        )
    elif controller == "CARTESIAN_VELOCITY":
        calls = run_velocity_trial(
            robot,
            controller_cfg,
            requested_delta,
            duration=duration,
            rate_hz=rate_hz,
            max_linear_velocity=max_linear_velocity,
        )
    elif controller in IK_CONTROLLERS:
        joint_traj, ik_solve_s, ik_debug = solve_ik_delta(
            requested_delta,
            start_q,
            num_points=ik_points,
        )
        calls = run_joint_trajectory(
            robot,
            controller_type,
            controller_cfg,
            joint_traj,
            duration=duration,
            rate_hz=rate_hz,
        )
    else:
        raise ValueError(f"Unsupported controller {controller!r}")
    actual_duration = time.monotonic() - started_at
    if settle > 0:
        time.sleep(settle)
    end_pos = current_eef_pos(robot)
    end_q = current_q(robot)
    if end_pos is None or end_q is None:
        raise RuntimeError("Could not read robot end state")

    metrics = compute_trial_metrics(
        axis=axis,
        requested_delta=requested_delta,
        start_pos=start_pos,
        end_pos=end_pos,
        duration_s=actual_duration,
        controller_calls=calls,
        ik_solve_s=ik_solve_s,
    )
    return {
        "event": "trial",
        "controller": controller,
        "deoxys_controller_type": controller_type,
        "axis": axis_name(axis),
        "started_at_monotonic": started_at,
        "duration_s": round(actual_duration, 9),
        "start_pos": start_pos.tolist(),
        "end_pos": end_pos.tolist(),
        "start_q": start_q,
        "end_q": end_q,
        "ik_debug": ik_debug,
        **metrics,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = add_symmetry_metrics(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Deoxys controller single-axis response.")
    parser.add_argument("--config", default="franka-two.yml", help="Deoxys interface config path/name")
    parser.add_argument("--controllers", default=DEFAULT_CONTROLLERS)
    parser.add_argument("--axes", default=DEFAULT_AXES)
    parser.add_argument("--step", type=float, default=0.03, help="Requested displacement per trial, m")
    parser.add_argument("--duration", type=float, default=2.0, help="Command duration per trial, s")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--max-delta-per-cycle", type=float, default=0.005, help="OSC max delta per control call, m")
    parser.add_argument("--max-linear-velocity", type=float, default=0.03, help="Velocity controller speed cap, m/s")
    parser.add_argument("--ik-points", type=int, default=50)
    parser.add_argument("--settle", type=float, default=0.5, help="Seconds to wait after each trial")
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--reset-between-trials", action="store_true", help="Reset to home joints before each trial")
    parser.add_argument("--reset-timeout", type=float, default=12.0, help="Seconds allowed for each reset")
    parser.add_argument("--jsonl", default=None, help="Output JSONL path")
    parser.add_argument("--csv", default=None, help="Output summary CSV path")
    parser.add_argument("--execute", action="store_true", help="Actually send robot commands")
    parser.add_argument("--no-gripper", action="store_true", help="Construct FrankaInterface with has_gripper=False")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    controllers = parse_csv(args.controllers)
    axes = [axis_name(axis) for axis in parse_csv(args.axes)]
    unsupported = [c for c in controllers if c not in SUPPORTED_CONTROLLERS]
    if unsupported:
        raise ValueError(f"Unsupported controllers: {unsupported}; supported: {SUPPORTED_CONTROLLERS}")
    for axis in axes:
        axis_delta(axis, args.step)
    if args.step <= 0 or args.step > MAX_STEP_M:
        raise ValueError(f"--step must be in (0, {MAX_STEP_M}]")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.max_delta_per_cycle <= 0:
        raise ValueError("--max-delta-per-cycle must be positive")
    if args.max_linear_velocity <= 0:
        raise ValueError("--max-linear-velocity must be positive")
    if args.ik_points <= 0:
        raise ValueError("--ik-points must be positive")
    if getattr(args, "reset_timeout", 12.0) <= 0:
        raise ValueError("--reset-timeout must be positive")
    return controllers, axes


def dry_run_records(args: argparse.Namespace, controllers: list[str], axes: list[str]) -> list[dict[str, Any]]:
    records = [
        {
            "event": "start",
            "dry_run": True,
            "config": str(resolve_config(args.config)),
            "controllers": controllers,
            "axes": axes,
            "step": args.step,
            "duration": args.duration,
            "rate_hz": args.rate_hz,
            "reset_between_trials": bool(getattr(args, "reset_between_trials", False)),
        }
    ]
    for controller in controllers:
        for axis in axes:
            records.append(
                {
                    "event": "planned_trial",
                    "controller": controller,
                    "axis": axis,
                    "requested_delta_m": axis_delta(axis, args.step).tolist(),
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        controllers, axes = validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    default_jsonl, default_csv = default_output_paths()
    jsonl_path = Path(args.jsonl) if args.jsonl else default_jsonl
    csv_path = Path(args.csv) if args.csv else default_csv
    cfg_path = resolve_config(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Missing config: {cfg_path}", file=sys.stderr)
        return 1

    if not args.execute:
        records = dry_run_records(args, controllers, axes)
        write_jsonl(jsonl_path, records)
        print("[DRY-RUN] No robot commands sent. Re-run with --execute to move hardware.")
        print(f"[DRY-RUN] Planned {len(records) - 1} trials")
        print(f"[DRY-RUN] JSONL plan: {jsonl_path}")
        return 0

    robot = None
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    try:
        FrankaInterface, get_default_controller_config, _YamlConfig = import_deoxys()
        robot = FrankaInterface(
            str(cfg_path),
            use_visualizer=False,
            has_gripper=not args.no_gripper,
            automatic_gripper_reset=False,
        )
        if not wait_for_state(robot, args.state_timeout):
            print("[ERROR] Timed out waiting for robot state", file=sys.stderr)
            return 1

        _reset_controller_type, reset_controller_cfg = load_controller_cfg("JOINT_POSITION", get_default_controller_config)

        start_record = {
            "event": "start",
            "dry_run": False,
            "config": str(cfg_path),
            "controllers": controllers,
            "axes": axes,
            "step": args.step,
            "duration": args.duration,
            "rate_hz": args.rate_hz,
            "reset_between_trials": bool(args.reset_between_trials),
            "initial_pos": current_eef_pos(robot).tolist(),
            "initial_q": current_q(robot),
        }
        records.append(start_record)
        print(f"[INFO] Starting {len(controllers) * len(axes)} trials")
        for controller in controllers:
            for axis in axes:
                if args.reset_between_trials:
                    print(f"[RESET] before controller={controller} axis={axis}")
                    maybe_reset_before_trial(robot, reset_controller_cfg, args)
                    time.sleep(args.settle)
                requested = axis_delta(axis, args.step)
                print(f"[TRIAL] controller={controller} axis={axis} requested={requested.tolist()}")
                try:
                    record = run_trial(
                        robot,
                        controller,
                        requested,
                        axis,
                        get_default_controller_config,
                        duration=args.duration,
                        rate_hz=args.rate_hz,
                        max_delta_per_cycle=args.max_delta_per_cycle,
                        max_linear_velocity=args.max_linear_velocity,
                        ik_points=args.ik_points,
                        settle=args.settle,
                    )
                    records.append(record)
                    summary_rows.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key
                            in {
                                "controller",
                                "deoxys_controller_type",
                                "axis",
                                "duration_s",
                                "main_displacement_m",
                                "abs_main_displacement_m",
                                "requested_main_m",
                                "drift_m",
                                "main_speed_m_s",
                                "abs_main_speed_m_s",
                                "tracking_efficiency",
                                "controller_calls",
                                "avg_call_period_s",
                                "ik_solve_s",
                            }
                        }
                    )
                    print(
                        "[RESULT] "
                        f"main={record['main_displacement_m']:.4f}m "
                        f"speed={record['abs_main_speed_m_s']:.4f}m/s "
                        f"drift={record['drift_m']:.4f}m "
                        f"eff={record['tracking_efficiency']} "
                        f"ik={record['ik_solve_s']}"
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark should continue after one failed strategy.
                    error_record = {
                        "event": "trial_error",
                        "controller": controller,
                        "axis": axis,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    records.append(error_record)
                    print(f"[ERROR] {controller} {axis}: {exc}", file=sys.stderr)
        records.append({"event": "shutdown", "final_pos": current_eef_pos(robot).tolist(), "final_q": current_q(robot)})
        write_jsonl(jsonl_path, records)
        write_summary_csv(csv_path, summary_rows)
        print(f"[DONE] JSONL: {jsonl_path}")
        print(f"[DONE] CSV: {csv_path}")
        return 0
    except KeyboardInterrupt:
        records.append({"event": "shutdown", "reason": "keyboard_interrupt"})
        write_jsonl(jsonl_path, records)
        print("\n[INFO] Interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        records.append({"event": "fatal_error", "error_type": type(exc).__name__, "message": str(exc)})
        write_jsonl(jsonl_path, records)
        print(f"[ERROR] Benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
