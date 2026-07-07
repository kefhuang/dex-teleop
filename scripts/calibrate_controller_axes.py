#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_teleop.core.config import apply_cli_overrides, load_config, resolve_repo_path
from dex_teleop.core.types import DeviceState, MotionCommand, Pose
from dex_teleop.devices.mock import MockControllerInput
from dex_teleop.devices.oculus import OculusDevice
from dex_teleop.robots.franka import FrankaArmController
from dex_teleop.robots.mock import MockArmController
from dex_teleop.utils.transforms import quaternion_delta_rpy

POSITION_OUTPUT_AXES = ("x", "y", "z")
POSITION_INPUT_AXES = ("x", "y", "z")
ROTATION_OUTPUT_AXES = ("roll", "pitch", "yaw")
ROTATION_INPUT_AXES = ("roll", "pitch", "yaw")


class AxisCalibrationError(RuntimeError):
    """Raised when an axis calibration measurement is unusable."""


@dataclass(frozen=True)
class AxisCalibrationResult:
    position_axes: dict[str, str]
    rotation_axes: dict[str, str]


def dominant_axis_spec(delta: np.ndarray, axes: Sequence[str], min_delta: float) -> str:
    values = np.asarray(delta, dtype=float)
    if values.shape != (len(axes),):
        raise ValueError(f"delta must have shape ({len(axes)},), got {values.shape}")

    index = int(np.argmax(np.abs(values)))
    magnitude = float(abs(values[index]))
    if magnitude < float(min_delta):
        raise AxisCalibrationError(
            f"Measured delta is too small ({magnitude:.6f}); move or rotate farther and try again."
        )

    sign = "+" if values[index] >= 0.0 else "-"
    return f"{sign}{axes[index]}"


def infer_mapping(
    measurements: Mapping[str, np.ndarray],
    *,
    output_axes: Sequence[str],
    input_axes: Sequence[str],
    min_delta: float,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for output_axis in output_axes:
        if output_axis not in measurements:
            raise AxisCalibrationError(f"Missing measurement for output axis {output_axis!r}")
        mapping[output_axis] = dominant_axis_spec(
            measurements[output_axis],
            input_axes,
            min_delta,
        )
    return mapping


def run_calibration_from_deltas(
    *,
    position_deltas: Mapping[str, np.ndarray],
    rotation_deltas: Mapping[str, np.ndarray],
    min_position_delta: float,
    min_rotation_delta: float,
) -> tuple[dict[str, str], dict[str, str]]:
    position_axes = infer_mapping(
        position_deltas,
        output_axes=POSITION_OUTPUT_AXES,
        input_axes=POSITION_INPUT_AXES,
        min_delta=min_position_delta,
    )
    rotation_axes = infer_mapping(
        rotation_deltas,
        output_axes=ROTATION_OUTPUT_AXES,
        input_axes=ROTATION_INPUT_AXES,
        min_delta=min_rotation_delta,
    )
    return position_axes, rotation_axes


def format_mapping_yaml(position_axes: Mapping[str, str], rotation_axes: Mapping[str, str]) -> str:
    lines = ["mapping:", "  position_axes:"]
    for axis in POSITION_OUTPUT_AXES:
        lines.append(f"    {axis}: \"{position_axes[axis]}\"")
    lines.append("  rotation_axes:")
    for axis in ROTATION_OUTPUT_AXES:
        lines.append(f"    {axis}: \"{rotation_axes[axis]}\"")
    return "\n".join(lines) + "\n"


def _snapshot_pose(pose: Pose) -> Pose:
    return Pose(position=pose.position.copy(), orientation=pose.orientation.copy())


def read_controller_pose(state: DeviceState, controller: str) -> Pose:
    controller_state = state.controllers.get(controller)
    if controller_state is None:
        raise RuntimeError(f"Device state does not contain controller {controller!r}")
    return _snapshot_pose(controller_state.pose)


def _make_device(cfg: dict[str, Any]) -> MockControllerInput | OculusDevice:
    input_cfg = cfg["input"]
    if input_cfg.get("mock_controller", False):
        return MockControllerInput(dt=1.0 / float(cfg["app"].get("rate_hz", 30)))
    return OculusDevice(ip=input_cfg.get("ip"), port=int(input_cfg.get("port", 5555)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guided Oculus controller axis calibration for dex-teleop mappings"
    )
    parser.add_argument("--config", required=True, help="Path to teleop YAML config")
    parser.add_argument("--controller", choices=("left", "right"), required=True)
    parser.add_argument("--arm", required=True, help="Arm id from config to move as calibration reference")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=None, help="Use mock robot")
    parser.add_argument("--real-robot", dest="dry_run", action="store_false", help="Use real robot")
    parser.add_argument("--mock-controller", action="store_true", default=None)
    parser.add_argument("--real-controller", dest="mock_controller", action="store_false")
    parser.add_argument("--samples", type=int, default=5, help="Samples to average for each capture")
    parser.add_argument("--sample-delay", type=float, default=0.02, help="Delay between samples")
    parser.add_argument("--position-step", type=float, default=0.02, help="Robot reference translation in meters")
    parser.add_argument("--rotation-step", type=float, default=0.10, help="Robot reference rotation in radians")
    parser.add_argument("--hold-time", type=float, default=0.8, help="Pause after each robot reference motion")
    parser.add_argument("--min-position-delta", type=float, default=0.02)
    parser.add_argument("--min-rotation-delta", type=float, default=0.15)
    return parser.parse_args(argv)


def _average_quaternion_xyzw(quaternions: list[np.ndarray]) -> np.ndarray:
    if not quaternions:
        raise ValueError("quaternions cannot be empty")
    reference = quaternions[0]
    aligned = []
    for quat in quaternions:
        values = np.asarray(quat, dtype=float)
        aligned.append(-values if float(np.dot(reference, values)) < 0.0 else values)
    average = np.mean(np.asarray(aligned), axis=0)
    norm = float(np.linalg.norm(average))
    if norm == 0.0:
        raise ValueError("average quaternion norm cannot be zero")
    return average / norm


def sample_controller_pose(
    read_state: Callable[[], DeviceState],
    controller: str,
    *,
    samples: int,
    sample_delay: float,
) -> Pose:
    if samples <= 0:
        raise ValueError("samples must be > 0")
    poses: list[Pose] = []
    for index in range(samples):
        poses.append(read_controller_pose(read_state(), controller))
        if index < samples - 1 and sample_delay > 0.0:
            time.sleep(sample_delay)
    return Pose(
        position=np.mean([pose.position for pose in poses], axis=0),
        orientation=_average_quaternion_xyzw([pose.orientation for pose in poses]),
    )


def _make_arm(cfg: dict[str, Any], arm_id: str) -> MockArmController | FrankaArmController:
    if arm_id not in cfg["arms"]:
        raise RuntimeError(f"Arm {arm_id!r} is not configured")
    arm_cfg = cfg["arms"][arm_id]
    if cfg["app"].get("dry_run", False):
        return MockArmController(arm_id)
    if arm_cfg.get("type") != "franka":
        raise RuntimeError(
            f"Unsupported real arm type for arm {arm_id!r}: {arm_cfg.get('type')!r}"
        )
    interface_cfg = arm_cfg["interface_cfg"]
    resolved = resolve_repo_path(interface_cfg)
    if not resolved.exists():
        raise RuntimeError(
            f"Real arm {arm_id!r} interface_cfg does not exist: {interface_cfg!r} "
            f"(resolved to {resolved})"
        )
    return FrankaArmController(arm_id, interface_cfg=str(resolved))


def _motion_command_for_axis(
    arm_id: str,
    *,
    group: str,
    axis: str,
    amount: float,
) -> MotionCommand:
    delta_position = np.zeros(3)
    delta_rotation = np.zeros(3)
    if group == "position":
        delta_position[POSITION_OUTPUT_AXES.index(axis)] = amount
    elif group == "rotation":
        delta_rotation[ROTATION_OUTPUT_AXES.index(axis)] = amount
    else:
        raise ValueError(f"Unsupported motion group {group!r}")
    return MotionCommand(
        arm_id=arm_id,
        delta_position=delta_position,
        delta_rotation=delta_rotation,
        enabled=True,
    )


def move_robot_reference(
    arm: Any,
    *,
    axis: str,
    group: str,
    amount: float,
    hold_time: float,
    sleep_func: Callable[[float], None] = time.sleep,
) -> None:
    arm.send_motion(
        _motion_command_for_axis(
            arm.arm_id,
            group=group,
            axis=axis,
            amount=amount,
        )
    )
    if hold_time > 0.0:
        sleep_func(hold_time)
    arm.stop_motion()


def return_robot_reference(
    arm: Any,
    *,
    axis: str,
    group: str,
    amount: float,
    hold_time: float,
    sleep_func: Callable[[float], None] = time.sleep,
) -> None:
    move_robot_reference(
        arm,
        axis=axis,
        group=group,
        amount=-amount,
        hold_time=hold_time,
        sleep_func=sleep_func,
    )


def capture_delta(
    read_state: Callable[[], DeviceState],
    controller: str,
    *,
    prompt: str,
    samples: int,
    sample_delay: float,
    input_func: Callable[[str], str] = input,
) -> tuple[np.ndarray, np.ndarray]:
    input_func(f"\n{prompt}\nHold the controller at the START pose, then press Enter.")
    start = sample_controller_pose(
        read_state,
        controller,
        samples=samples,
        sample_delay=sample_delay,
    )
    input_func("Move/rotate to the END pose, hold still, then press Enter.")
    end = sample_controller_pose(
        read_state,
        controller,
        samples=samples,
        sample_delay=sample_delay,
    )
    return (
        end.position - start.position,
        quaternion_delta_rpy(start.orientation, end.orientation),
    )


def run_calibration(
    read_state: Callable[[], DeviceState],
    arm: Any,
    controller: str,
    *,
    samples: int = 5,
    sample_delay: float = 0.02,
    position_step: float = 0.02,
    rotation_step: float = 0.10,
    hold_time: float = 0.8,
    min_position_delta: float = 0.02,
    min_rotation_delta: float = 0.15,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
    sleep_func: Callable[[float], None] = time.sleep,
) -> AxisCalibrationResult:
    position_deltas: dict[str, np.ndarray] = {}
    rotation_deltas: dict[str, np.ndarray] = {}

    print_func(
        "Position calibration: the robot moves first; imitate that direction with the controller."
    )
    for axis in POSITION_OUTPUT_AXES:
        input_func(f"\nPress Enter to move robot +{axis} by {position_step:.3f} m.")
        move_robot_reference(
            arm,
            axis=axis,
            group="position",
            amount=position_step,
            hold_time=hold_time,
            sleep_func=sleep_func,
        )
        try:
            position_delta, _ = capture_delta(
                read_state,
                controller,
                prompt=f"Now imitate the robot +{axis} movement with the controller.",
                samples=samples,
                sample_delay=sample_delay,
                input_func=input_func,
            )
            position_deltas[axis] = position_delta
            print_func(f"robot +{axis}: raw controller position delta {np.round(position_delta, 5).tolist()}")
        finally:
            input_func(f"Press Enter to return robot -{axis} to the start pose.")
            return_robot_reference(
                arm,
                axis=axis,
                group="position",
                amount=position_step,
                hold_time=hold_time,
                sleep_func=sleep_func,
            )

    print_func(
        "Rotation calibration: the robot rotates first; imitate that rotation with the controller."
    )
    for axis in ROTATION_OUTPUT_AXES:
        input_func(f"\nPress Enter to rotate robot +{axis} by {rotation_step:.3f} rad.")
        move_robot_reference(
            arm,
            axis=axis,
            group="rotation",
            amount=rotation_step,
            hold_time=hold_time,
            sleep_func=sleep_func,
        )
        try:
            _, rotation_delta = capture_delta(
                read_state,
                controller,
                prompt=f"Now imitate the robot +{axis} rotation with the controller.",
                samples=samples,
                sample_delay=sample_delay,
                input_func=input_func,
            )
            rotation_deltas[axis] = rotation_delta
            print_func(f"robot +{axis}: raw controller rotation delta {np.round(rotation_delta, 5).tolist()}")
        finally:
            input_func(f"Press Enter to return robot -{axis} to the start pose.")
            return_robot_reference(
                arm,
                axis=axis,
                group="rotation",
                amount=rotation_step,
                hold_time=hold_time,
                sleep_func=sleep_func,
            )

    position_axes, rotation_axes = run_calibration_from_deltas(
        position_deltas=position_deltas,
        rotation_deltas=rotation_deltas,
        min_position_delta=min_position_delta,
        min_rotation_delta=min_rotation_delta,
    )
    print_func("\nSuggested config snippet:\n")
    print_func(format_mapping_yaml(position_axes, rotation_axes))
    return AxisCalibrationResult(position_axes=position_axes, rotation_axes=rotation_axes)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(
        cfg,
        dry_run=args.dry_run,
        mock_controller=args.mock_controller,
    )
    device = _make_device(cfg)
    arm = _make_arm(cfg, args.arm)
    try:
        device.start()
        arm.connect()
        run_calibration(
            device.read,
            arm,
            args.controller,
            samples=args.samples,
            sample_delay=args.sample_delay,
            position_step=args.position_step,
            rotation_step=args.rotation_step,
            hold_time=args.hold_time,
            min_position_delta=args.min_position_delta,
            min_rotation_delta=args.min_rotation_delta,
        )
    except AxisCalibrationError as exc:
        print(f"Calibration error: {exc}")
        return 2
    finally:
        try:
            arm.stop_motion()
        except Exception:
            pass
        arm.close()
        device.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
