#!/usr/bin/env python3
"""Continuously draw a rotating diagonal Cartesian figure eight.

This is intentionally separate from ``test_deoxys_joint_control.py``.  The raw
Deoxys diagnostic isolates the backend; this diagnostic exercises the formal
``Robot -> FrankaArm.ik_joint_position -> Deoxys JOINT_POSITION`` path.

The end effector keeps its initial orientation while tracing a diagonal eight
in a plane that starts at base Y-Z.  The plane rotates continuously about the
vertical base-Z line through the trajectory center, advancing 90 degrees every
three eights.  Motion continues until Ctrl-C.  The script never sends a gripper
command.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_teleop.core.types import ArmState, Pose, Twist  # noqa: E402
from dex_teleop.diagnostics import ControlTracer, trace_event  # noqa: E402
from dex_teleop.robots.arm import ArmMotionState  # noqa: E402
from dex_teleop.robots.franka import make_franka_robot  # noqa: E402
from dex_teleop.utils.transforms import (  # noqa: E402
    quaternion_xyzw_to_matrix,
    rotation_error_angle,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "dependency"
    / "deoxys_control"
    / "deoxys"
    / "config"
    / "franka-three.yml"
)
LOG_DIR = REPO_ROOT / "logs"

RATE_HZ = 20.0
EIGHT_DURATION_S = 8.0
DRAW_DURATION_S = EIGHT_DURATION_S
PRE_HOLD_DURATION_S = 1.0
RETURN_HOLD_DURATION_S = 2.0
# Twelve eights advance the plane through one full revolution.  Preview that
# complete repeat period so the operator sees the entire infinite workspace.
PREVIEW_EIGHT_COUNT = 12
SETTLE_TIMEOUT_S = 5.0

# A 16 cm x 24 cm Gerono eight, tilted diagonally inside its initial Y-Z
# plane.  The plane advances 30 degrees per eight about the vertical base-Z
# line through the anchor, hence 90 degrees every three eights.  Tool
# orientation stays fixed, avoiding an unnecessary wrist revolution.
Y_AMPLITUDE_M = 0.08
Z_AMPLITUDE_M = 0.12
FIGURE_EIGHT_TILT_RAD = np.deg2rad(45.0)
PLANE_ROTATION_PER_EIGHT_RAD = np.deg2rad(30.0)
PLANE_ROTATIONS = PLANE_ROTATION_PER_EIGHT_RAD / (2.0 * np.pi)
MAX_PUBLIC_IK_TRANSLATION_STEP_M = 0.01

# StreamingJointPositionTrajInterpolator uses exactly these limits: half of
# libfranka's per-joint velocity limits, plus its own acceleration and jerk
# limits.  Configure FrankaArm's command shaper to the same velocity vector so
# this diagnostic can use the complete accepted range instead of its default
# scalar 1 rad/s cap.
DEOXYS_STREAMING_MAX_VELOCITY_RAD_S = np.array(
    [1.0, 0.5, 0.75, 0.625, 1.5, 0.75, 1.5], dtype=float
)
DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2 = 1.0
DEOXYS_STREAMING_MAX_JERK_RAD_S3 = 100.0
MAX_VELOCITY_LIMIT_RELATIVE_ERROR = 0.01
MAX_JOINT_REFERENCE_INTEGRATION_ERROR_RAD = 1e-8

MIN_COMMAND_RATE_HZ = 18.0
MAX_COMMAND_GAP_S = 0.10
MAX_RMS_TRACKING_ERROR_M = 0.03
MAX_TRACKING_ERROR_M = 0.06
MAX_RETURN_ERROR_M = 0.02
MAX_ORIENTATION_ERROR_RAD = 0.12
MIN_EXTENT_FRACTION = 0.65


@dataclass(frozen=True)
class FigureEightSample:
    phase: str
    iteration: int
    scheduled_elapsed_s: float
    actual_elapsed_s: float
    command_monotonic_s: float
    lateness_s: float
    target_position: np.ndarray
    target_linear_velocity: np.ndarray | None
    measured_position: np.ndarray
    position_error: np.ndarray
    orientation_error_rad: float
    measured_joint_positions: np.ndarray
    measured_joint_velocities: np.ndarray
    ik_joint_positions: np.ndarray | None
    ik_joint_velocities: np.ndarray | None
    commanded_joint_positions: np.ndarray | None
    commanded_joint_velocities: np.ndarray | None
    backend_desired_joint_positions: np.ndarray | None
    backend_desired_joint_velocities: np.ndarray | None
    backend_desired_joint_accelerations: np.ndarray | None


@dataclass(frozen=True)
class FigureEightSummary:
    passed: bool
    command_rate_hz: float
    max_command_gap_s: float
    rms_tracking_error_m: float
    max_tracking_error_m: float
    return_error_m: float
    max_orientation_error_rad: float
    measured_x_extent_m: tuple[float, float]
    measured_y_extent_m: tuple[float, float]
    measured_z_extent_m: tuple[float, float]
    center_crossing_error_m: float
    max_abs_joint_velocity_rad_s: float
    max_commanded_joint_velocity_rad_s: tuple[float, ...]
    commanded_velocity_utilization: tuple[float, ...]
    peak_commanded_velocity_utilization: float
    max_joint_velocity_reference_rad_s: tuple[float, ...]
    velocity_reference_utilization: tuple[float, ...]
    peak_velocity_reference_utilization: float
    max_joint_reference_integration_error_rad: float
    max_backend_desired_joint_velocity_rad_s: tuple[float, ...]
    backend_desired_velocity_utilization: tuple[float, ...]
    peak_backend_desired_velocity_utilization: float
    max_backend_desired_joint_acceleration_rad_s2: tuple[float, ...]
    checks: dict[str, bool]


def figure_eight_offset_for_angle(
    angle_rad: float, plane_angle_rad: float = 0.0
) -> np.ndarray:
    """Return a diagonal Y-Z Gerono-eight after rotating its plane about Z."""

    angle = float(angle_rad)
    plane_angle = float(plane_angle_rad)
    raw_y = Y_AMPLITUDE_M * np.sin(2.0 * angle)
    raw_z = Z_AMPLITUDE_M * np.sin(angle)
    tilt_cosine = np.cos(FIGURE_EIGHT_TILT_RAD)
    tilt_sine = np.sin(FIGURE_EIGHT_TILT_RAD)
    local_y = tilt_cosine * raw_y - tilt_sine * raw_z
    local_z = tilt_sine * raw_y + tilt_cosine * raw_z
    cosine = np.cos(plane_angle)
    sine = np.sin(plane_angle)
    return np.array(
        [
            -sine * local_y,
            cosine * local_y,
            local_z,
        ],
        dtype=float,
    )


def figure_eight_reference(
    anchor: Pose, elapsed_s: float, duration_s: float
) -> tuple[Pose, Twist]:
    """Return a periodic pose/twist pair with no stop at eight boundaries."""

    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    elapsed = float(elapsed_s)
    angle_rate = 2.0 * np.pi / duration
    plane_angle_rate = PLANE_ROTATION_PER_EIGHT_RAD / duration
    angle = angle_rate * elapsed
    plane_angle = plane_angle_rate * elapsed
    raw_y = Y_AMPLITUDE_M * np.sin(2.0 * angle)
    raw_y_rate = 2.0 * Y_AMPLITUDE_M * np.cos(2.0 * angle) * angle_rate
    raw_z_rate = Z_AMPLITUDE_M * np.cos(angle) * angle_rate
    tilt_cosine = np.cos(FIGURE_EIGHT_TILT_RAD)
    tilt_sine = np.sin(FIGURE_EIGHT_TILT_RAD)
    local_y = tilt_cosine * raw_y - tilt_sine * (
        Z_AMPLITUDE_M * np.sin(angle)
    )
    local_y_rate = tilt_cosine * raw_y_rate - tilt_sine * raw_z_rate
    local_z_rate = tilt_sine * raw_y_rate + tilt_cosine * raw_z_rate
    cosine = np.cos(plane_angle)
    sine = np.sin(plane_angle)
    linear_velocity = np.array(
        [
            -cosine * plane_angle_rate * local_y - sine * local_y_rate,
            -sine * plane_angle_rate * local_y + cosine * local_y_rate,
            local_z_rate,
        ],
        dtype=float,
    )
    pose = Pose(
        anchor.position + figure_eight_offset_for_angle(angle, plane_angle),
        anchor.quaternion_xyzw,
    )
    return pose, Twist(linear_velocity, np.zeros(3))


def figure_eight_pose(anchor: Pose, elapsed_s: float, duration_s: float) -> Pose:
    return figure_eight_reference(anchor, elapsed_s, duration_s)[0]


def planned_trajectory_metrics(
    anchor: Pose,
    *,
    eight_duration_s: float = EIGHT_DURATION_S,
    eight_count: int = PREVIEW_EIGHT_COUNT,
    rate_hz: float = RATE_HZ,
) -> dict[str, Any]:
    """Preview several cycles of the exact infinite trajectory."""

    if int(eight_count) <= 0:
        raise ValueError("eight_count must be positive")
    duration = float(eight_duration_s) * int(eight_count)
    count = int(round(duration * float(rate_hz)))
    targets = np.stack(
        [
            figure_eight_pose(
                anchor, index / float(rate_hz), eight_duration_s
            ).position
            for index in range(count + 1)
        ]
    )
    steps = np.linalg.norm(np.diff(targets, axis=0), axis=1)
    return {
        "target_min": np.min(targets, axis=0),
        "target_max": np.max(targets, axis=0),
        "max_target_step_m": float(np.max(steps)),
        "peak_discrete_target_speed_m_s": float(np.max(steps) * rate_hz),
        "target_count": int(targets.shape[0]),
    }


def _optional_vector(value: Any, size: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        return None
    return result.copy()


def _sleep_until(
    deadline: float,
    *,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> None:
    remaining = deadline - monotonic_fn()
    if remaining > 0.0:
        sleep_fn(remaining)


def stream_pose_targets(
    arm: Any,
    target_fn: Callable[[float], Pose],
    *,
    twist_fn: Callable[[float], Twist] | None = None,
    phase: str,
    duration_s: float | None,
    rate_hz: float = RATE_HZ,
    include_endpoint: bool,
    tracer: ControlTracer | None = None,
    sample_callback: Callable[[FigureEightSample], None] | None = None,
    retain_samples: bool = True,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    print_fn: Callable[[str], None] = print,
) -> list[FigureEightSample]:
    """Stream absolute poses through ``Arm.ik_joint_position`` at a fixed rate."""

    rate = float(rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_hz must be finite and positive")
    period = 1.0 / rate
    if duration_s is None:
        duration = None
        sample_count = None
    else:
        duration = float(duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        interval_count = int(round(duration * rate))
        sample_count = interval_count + 1 if include_endpoint else interval_count
        if sample_count <= 0:
            raise ValueError("stream must contain at least one sample")

    started = monotonic_fn()
    report_every = max(1, int(round(0.5 * rate)))
    samples: list[FigureEightSample] = []
    iteration = 0
    while sample_count is None or iteration < sample_count:
        scheduled_elapsed = iteration * period
        if duration is not None:
            scheduled_elapsed = min(scheduled_elapsed, duration)
        deadline = started + scheduled_elapsed
        _sleep_until(deadline, monotonic_fn=monotonic_fn, sleep_fn=sleep_fn)
        command_time = monotonic_fn()
        target = target_fn(scheduled_elapsed)
        twist_reference = (
            None if twist_fn is None else twist_fn(scheduled_elapsed)
        )
        span = (
            tracer.span(
                "ik_figure_eight_cycle",
                phase=phase,
                iteration=iteration,
                scheduled_elapsed_s=scheduled_elapsed,
                target_position=target.position,
            )
            if tracer is not None
            else nullcontext()
        )
        with span:
            state: ArmState = arm.get_state()
            trace_event(
                "diagnostic.ik_figure_eight.target",
                phase=phase,
                target_pose=target,
                target_twist=twist_reference,
                measured_state=state,
            )
            arm.ik_joint_position(
                target,
                twist_ref=twist_reference,
                delta=False,
                dt=period,
            )
            q_ik = _optional_vector(
                getattr(arm, "last_ik_joint_positions", None), arm.dof
            )
            dq_ik = _optional_vector(
                getattr(arm, "last_ik_joint_velocities", None), arm.dof
            )
            q_cmd = _optional_vector(
                getattr(arm, "last_commanded_joint_positions", None), arm.dof
            )
            dq_cmd = _optional_vector(
                getattr(arm, "last_commanded_joint_velocities", None), arm.dof
            )
            interface = getattr(arm, "interface", None)
            q_d = _optional_vector(getattr(interface, "last_q_d", None), arm.dof)
            backend_state = getattr(interface, "last_state", None)
            dq_d = _optional_vector(
                getattr(backend_state, "dq_d", None), arm.dof
            )
            ddq_d = _optional_vector(
                getattr(backend_state, "ddq_d", None), arm.dof
            )
            error = target.position - state.end_effector_pose.position
            orientation_error = rotation_error_angle(
                quaternion_xyzw_to_matrix(target.quaternion_xyzw),
                quaternion_xyzw_to_matrix(
                    state.end_effector_pose.quaternion_xyzw
                ),
            )
            sample = FigureEightSample(
                phase=phase,
                iteration=iteration,
                scheduled_elapsed_s=scheduled_elapsed,
                actual_elapsed_s=command_time - started,
                command_monotonic_s=command_time,
                lateness_s=max(0.0, command_time - deadline),
                target_position=target.position.copy(),
                target_linear_velocity=(None if twist_reference is None else twist_reference.linear.copy()),
                measured_position=state.end_effector_pose.position.copy(),
                position_error=error.copy(),
                orientation_error_rad=orientation_error,
                measured_joint_positions=state.joint_positions.copy(),
                measured_joint_velocities=state.joint_velocities.copy(),
                ik_joint_positions=q_ik,
                ik_joint_velocities=dq_ik,
                commanded_joint_positions=q_cmd,
                commanded_joint_velocities=dq_cmd,
                backend_desired_joint_positions=q_d,
                backend_desired_joint_velocities=dq_d,
                backend_desired_joint_accelerations=ddq_d,
            )
            trace_event("diagnostic.ik_figure_eight.sample", sample=sample)
        if retain_samples:
            samples.append(sample)
        if sample_callback is not None:
            sample_callback(sample)

        is_last = sample_count is not None and iteration == sample_count - 1
        if iteration % report_every == 0 or is_last:
            cycle = scheduled_elapsed / EIGHT_DURATION_S
            plane_angle_deg = np.degrees(
                PLANE_ROTATION_PER_EIGHT_RAD * cycle
            )
            print_fn(
                f"[{phase}] t={scheduled_elapsed:5.2f}s "
                f"eight={cycle:.2f} plane={plane_angle_deg:.1f}deg "
                f"target_xyz={_format_vector(target.position)} "
                f"measured_xyz={_format_vector(state.end_effector_pose.position)} "
                f"error={np.linalg.norm(error):.4f}m "
                f"q_ik={'ok' if q_ik is not None else 'n/a'} "
                f"q_cmd={'ok' if q_cmd is not None else 'n/a'}"
            )
        iteration += 1
    return samples


def run_figure_eight(
    arm: Any,
    anchor: Pose,
    *,
    tracer: ControlTracer | None = None,
    duration_s: float = DRAW_DURATION_S,
    rate_hz: float = RATE_HZ,
    pre_hold_duration_s: float = PRE_HOLD_DURATION_S,
    return_hold_duration_s: float = RETURN_HOLD_DURATION_S,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    print_fn: Callable[[str], None] = print,
) -> list[FigureEightSample]:
    """Arm the formal IK path, draw one eight, and settle at the anchor."""

    samples: list[FigureEightSample] = []
    samples.extend(
        stream_pose_targets(
            arm,
            lambda _elapsed: anchor,
            phase="pre-hold",
            duration_s=pre_hold_duration_s,
            rate_hz=rate_hz,
            include_endpoint=False,
            tracer=tracer,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            print_fn=print_fn,
        )
    )
    samples.extend(
        stream_pose_targets(
            arm,
            lambda elapsed: figure_eight_pose(anchor, elapsed, duration_s),
            twist_fn=lambda elapsed: figure_eight_reference(
                anchor, elapsed, duration_s
            )[1],
            phase="draw-8",
            duration_s=duration_s,
            rate_hz=rate_hz,
            include_endpoint=True,
            tracer=tracer,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            print_fn=print_fn,
        )
    )
    samples.extend(
        stream_pose_targets(
            arm,
            lambda _elapsed: anchor,
            phase="return-hold",
            duration_s=return_hold_duration_s,
            rate_hz=rate_hz,
            include_endpoint=False,
            tracer=tracer,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            print_fn=print_fn,
        )
    )
    return samples


def summarize_run(
    samples: list[FigureEightSample],
    anchor: Pose,
    final_state: ArmState,
) -> FigureEightSummary:
    draw = [sample for sample in samples if sample.phase == "draw-8"]
    if len(draw) < 2:
        raise ValueError("at least two draw-8 samples are required")

    times = np.array([sample.command_monotonic_s for sample in draw])
    gaps = np.diff(times)
    rate = (len(draw) - 1) / (times[-1] - times[0])
    errors = np.array(
        [np.linalg.norm(sample.position_error) for sample in draw], dtype=float
    )
    measured = np.stack([sample.measured_position for sample in draw])
    relative = measured - anchor.position
    target_relative = (
        np.stack([sample.target_position for sample in draw]) - anchor.position
    )
    orientation = np.array(
        [sample.orientation_error_rad for sample in draw], dtype=float
    )
    joint_velocities = np.stack(
        [sample.measured_joint_velocities for sample in draw]
    )
    midpoint = min(
        draw,
        key=lambda sample: abs(
            sample.scheduled_elapsed_s
            - 0.5 * draw[-1].scheduled_elapsed_s
        ),
    )
    center_error = float(
        np.linalg.norm(midpoint.measured_position - anchor.position)
    )
    x_extent = (float(np.min(relative[:, 0])), float(np.max(relative[:, 0])))
    y_extent = (float(np.min(relative[:, 1])), float(np.max(relative[:, 1])))
    z_extent = (float(np.min(relative[:, 2])), float(np.max(relative[:, 2])))
    target_x_extent = (
        float(np.min(target_relative[:, 0])),
        float(np.max(target_relative[:, 0])),
    )
    target_y_extent = (
        float(np.min(target_relative[:, 1])),
        float(np.max(target_relative[:, 1])),
    )
    target_z_extent = (
        float(np.min(target_relative[:, 2])),
        float(np.max(target_relative[:, 2])),
    )

    def extent_reached(
        measured_extent: tuple[float, float],
        target_extent: tuple[float, float],
        direction: int,
    ) -> bool:
        target_magnitude = target_extent[1] if direction > 0 else -target_extent[0]
        measured_magnitude = (
            measured_extent[1] if direction > 0 else -measured_extent[0]
        )
        return bool(
            target_magnitude <= 1e-9
            or measured_magnitude >= MIN_EXTENT_FRACTION * target_magnitude
        )
    return_error = float(
        np.linalg.norm(final_state.end_effector_pose.position - anchor.position)
    )
    rms_error = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    max_orientation = float(np.max(orientation))
    max_joint_velocity = float(np.max(np.abs(joint_velocities)))
    q_commands = [sample.commanded_joint_positions for sample in draw]
    complete_commands = all(command is not None for command in q_commands)
    if complete_commands:
        command_matrix = np.stack(q_commands)
        scheduled_times = np.array(
            [sample.scheduled_elapsed_s for sample in draw], dtype=float
        )
        commanded_velocity = np.abs(
            np.diff(command_matrix, axis=0)
            / np.diff(scheduled_times)[:, np.newaxis]
        )
        max_commanded_velocity = np.max(commanded_velocity, axis=0)
        utilization = (
            max_commanded_velocity / DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
        )
    else:
        max_commanded_velocity = np.full(7, np.nan)
        utilization = np.full(7, np.nan)
    peak_utilization = (
        float(np.max(utilization)) if complete_commands else float("nan")
    )
    velocity_references = [
        sample.commanded_joint_velocities for sample in draw
    ]
    complete_velocity_references = all(
        reference is not None for reference in velocity_references
    )
    if complete_velocity_references:
        velocity_reference_matrix = np.stack(velocity_references)
        max_velocity_reference = np.max(
            np.abs(velocity_reference_matrix), axis=0
        )
        velocity_reference_utilization = (
            max_velocity_reference / DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
        )
        peak_velocity_reference_utilization = float(
            np.max(velocity_reference_utilization)
        )
    else:
        velocity_reference_matrix = None
        max_velocity_reference = np.full(7, np.nan)
        velocity_reference_utilization = np.full(7, np.nan)
        peak_velocity_reference_utilization = float("nan")
    complete_joint_reference = bool(
        complete_commands and complete_velocity_references
    )
    if complete_joint_reference:
        intervals = np.diff(scheduled_times)[:, np.newaxis]
        integrated_displacement = 0.5 * (
            velocity_reference_matrix[:-1] + velocity_reference_matrix[1:]
        ) * intervals
        reference_integration_error = np.diff(command_matrix, axis=0) - integrated_displacement
        max_reference_integration_error = float(
            np.max(np.abs(reference_integration_error))
        )
    else:
        max_reference_integration_error = float("nan")
    backend_velocities = [
        sample.backend_desired_joint_velocities for sample in draw
    ]
    backend_accelerations = [
        sample.backend_desired_joint_accelerations for sample in draw
    ]
    complete_backend_velocities = all(
        velocity is not None for velocity in backend_velocities
    )
    complete_backend_accelerations = all(
        acceleration is not None for acceleration in backend_accelerations
    )
    if complete_backend_velocities:
        max_backend_velocity = np.max(
            np.abs(np.stack(backend_velocities)), axis=0
        )
        backend_utilization = (
            max_backend_velocity / DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
        )
        peak_backend_utilization = float(np.max(backend_utilization))
    else:
        max_backend_velocity = np.full(7, np.nan)
        backend_utilization = np.full(7, np.nan)
        peak_backend_utilization = float("nan")
    if complete_backend_accelerations:
        max_backend_acceleration = np.max(
            np.abs(np.stack(backend_accelerations)), axis=0
        )
    else:
        max_backend_acceleration = np.full(7, np.nan)

    checks = {
        "command_rate": rate >= MIN_COMMAND_RATE_HZ,
        "command_gap": float(np.max(gaps)) <= MAX_COMMAND_GAP_S,
        "rms_tracking": rms_error <= MAX_RMS_TRACKING_ERROR_M,
        "max_tracking": max_error <= MAX_TRACKING_ERROR_M,
        "return_to_anchor": return_error <= MAX_RETURN_ERROR_M,
        "orientation_hold": max_orientation <= MAX_ORIENTATION_ERROR_RAD,
        "positive_x_extent": extent_reached(x_extent, target_x_extent, 1),
        "negative_x_extent": extent_reached(x_extent, target_x_extent, -1),
        "positive_y_extent": extent_reached(y_extent, target_y_extent, 1),
        "negative_y_extent": extent_reached(y_extent, target_y_extent, -1),
        "positive_z_extent": extent_reached(z_extent, target_z_extent, 1),
        "negative_z_extent": extent_reached(z_extent, target_z_extent, -1),
        "commanded_velocity_available": complete_commands,
        "commanded_velocity_within_deoxys_limit": bool(
            complete_commands
            and np.all(
                max_commanded_velocity
                <= DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
                * (1.0 + MAX_VELOCITY_LIMIT_RELATIVE_ERROR)
            )
        ),
        "velocity_reference_available": complete_velocity_references,
        "velocity_reference_within_deoxys_limit": bool(
            complete_velocity_references
            and np.all(
                max_velocity_reference
                <= DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
                * (1.0 + MAX_VELOCITY_LIMIT_RELATIVE_ERROR)
            )
        ),
        "joint_reference_is_coupled": bool(
            complete_joint_reference
            and max_reference_integration_error
            <= MAX_JOINT_REFERENCE_INTEGRATION_ERROR_RAD
        ),
        "backend_desired_velocity_available": complete_backend_velocities,
        "backend_desired_velocity_within_deoxys_limit": bool(
            complete_backend_velocities
            and np.all(
                max_backend_velocity
                <= DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
                * (1.0 + MAX_VELOCITY_LIMIT_RELATIVE_ERROR)
            )
        ),
    }
    return FigureEightSummary(
        passed=all(checks.values()),
        command_rate_hz=float(rate),
        max_command_gap_s=float(np.max(gaps)),
        rms_tracking_error_m=rms_error,
        max_tracking_error_m=max_error,
        return_error_m=return_error,
        max_orientation_error_rad=max_orientation,
        measured_x_extent_m=x_extent,
        measured_y_extent_m=y_extent,
        measured_z_extent_m=z_extent,
        center_crossing_error_m=center_error,
        max_abs_joint_velocity_rad_s=max_joint_velocity,
        max_commanded_joint_velocity_rad_s=tuple(max_commanded_velocity),
        commanded_velocity_utilization=tuple(utilization),
        peak_commanded_velocity_utilization=peak_utilization,
        max_joint_velocity_reference_rad_s=tuple(max_velocity_reference),
        velocity_reference_utilization=tuple(velocity_reference_utilization),
        peak_velocity_reference_utilization=peak_velocity_reference_utilization,
        max_joint_reference_integration_error_rad=max_reference_integration_error,
        max_backend_desired_joint_velocity_rad_s=tuple(max_backend_velocity),
        backend_desired_velocity_utilization=tuple(backend_utilization),
        peak_backend_desired_velocity_utilization=peak_backend_utilization,
        max_backend_desired_joint_acceleration_rad_s2=tuple(
            max_backend_acceleration
        ),
        checks=checks,
    )


def _csv_row(sample: FigureEightSample) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": sample.phase,
        "iteration": sample.iteration,
        "scheduled_elapsed_s": sample.scheduled_elapsed_s,
        "actual_elapsed_s": sample.actual_elapsed_s,
        "command_monotonic_s": sample.command_monotonic_s,
        "lateness_s": sample.lateness_s,
        "position_error_norm_m": float(np.linalg.norm(sample.position_error)),
        "orientation_error_rad": sample.orientation_error_rad,
    }
    for axis, index in zip("xyz", range(3)):
        row[f"target_{axis}_m"] = sample.target_position[index]
        row[f"target_v{axis}_m_s"] = "" if sample.target_linear_velocity is None else sample.target_linear_velocity[index]
        row[f"measured_{axis}_m"] = sample.measured_position[index]
        row[f"error_{axis}_m"] = sample.position_error[index]
    vectors = {
        "q": sample.measured_joint_positions,
        "dq": sample.measured_joint_velocities,
        "q_ik": sample.ik_joint_positions,
        "dq_ik": sample.ik_joint_velocities,
        "q_cmd": sample.commanded_joint_positions,
        "dq_cmd": sample.commanded_joint_velocities,
        "q_d": sample.backend_desired_joint_positions,
        "dq_d": sample.backend_desired_joint_velocities,
        "ddq_d": sample.backend_desired_joint_accelerations,
    }
    for name, values in vectors.items():
        for index in range(7):
            row[f"{name}{index + 1}"] = "" if values is None else values[index]
    return row


def write_csv(path: Path, samples: list[FigureEightSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_csv_row(sample) for sample in samples]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvWriter:
    """Write an unbounded hardware run without retaining it in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="", buffering=1)
        self._writer: csv.DictWriter | None = None

    def write(self, sample: FigureEightSample) -> None:
        row = _csv_row(sample)
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()


def hold_arm_until_settled(
    arm: Any,
    *,
    timeout_s: float = SETTLE_TIMEOUT_S,
    rate_hz: float = RATE_HZ,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Keep feeding the immutable hold until the arm has physically settled."""

    period = 1.0 / float(rate_hz)
    deadline = monotonic_fn() + float(timeout_s)
    while monotonic_fn() < deadline:
        arm.stop()
        if arm.is_settled:
            return True
        sleep_fn(period)
    return False


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values) + "]"


def _print_summary(summary: FigureEightSummary) -> None:
    status = "PASS" if summary.passed else "FAIL"
    failed = [name for name, passed in summary.checks.items() if not passed]
    print("\n=== CARTESIAN FIGURE-EIGHT SUMMARY ===")
    print(
        f"[{status}] rate={summary.command_rate_hz:.1f}Hz "
        f"max_gap={summary.max_command_gap_s:.3f}s "
        f"tracking_rms/max={summary.rms_tracking_error_m:.4f}/"
        f"{summary.max_tracking_error_m:.4f}m "
        f"return_error={summary.return_error_m:.4f}m"
    )
    print(
        f"       measured X extent={summary.measured_x_extent_m[0]:+.4f}/"
        f"{summary.measured_x_extent_m[1]:+.4f}m "
        f"Y extent={summary.measured_y_extent_m[0]:+.4f}/"
        f"{summary.measured_y_extent_m[1]:+.4f}m "
        f"Z extent={summary.measured_z_extent_m[0]:+.4f}/"
        f"{summary.measured_z_extent_m[1]:+.4f}m "
        f"center_crossing_error={summary.center_crossing_error_m:.4f}m"
    )
    print(
        f"       max_orientation_error={summary.max_orientation_error_rad:.4f}rad "
        f"max_abs_joint_velocity={summary.max_abs_joint_velocity_rad_s:.4f}rad/s"
    )
    print(
        "       q_cmd max velocity="
        f"{_format_vector(np.asarray(summary.max_commanded_joint_velocity_rad_s))} "
        "rad/s"
    )
    print(
        "       q_cmd velocity utilization="
        f"{_format_vector(np.asarray(summary.commanded_velocity_utilization))} "
        f"peak={summary.peak_commanded_velocity_utilization:.1%}"
    )
    print(
        "       transmitted dq_ref max="
        f"{_format_vector(np.asarray(summary.max_joint_velocity_reference_rad_s))} "
        "rad/s"
    )
    print(
        "       dq_ref velocity utilization="
        f"{_format_vector(np.asarray(summary.velocity_reference_utilization))} "
        f"peak={summary.peak_velocity_reference_utilization:.1%}"
    )
    print(
        "       q_ref/dq_ref integration error="
        f"{summary.max_joint_reference_integration_error_rad:.3e}rad"
    )
    print(
        "       backend dq_d max="
        f"{_format_vector(np.asarray(summary.max_backend_desired_joint_velocity_rad_s))} "
        "rad/s"
    )
    print(
        "       backend velocity utilization="
        f"{_format_vector(np.asarray(summary.backend_desired_velocity_utilization))} "
        f"peak={summary.peak_backend_desired_velocity_utilization:.1%}"
    )
    print(
        "       backend ddq_d max="
        f"{_format_vector(np.asarray(summary.max_backend_desired_joint_acceleration_rad_s2))} "
        "rad/s^2"
    )
    if failed:
        print(f"       failed_checks={', '.join(failed)}")


def resolve_config(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously draw a diagonal Cartesian figure eight whose Y-Z "
            "plane advances 90 degrees about base Z every three eights."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Deoxys interface config (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config(args.config)
    if not config_path.exists():
        print(f"[ERROR] Missing config: {config_path}", file=sys.stderr)
        return 2

    robot = None
    tracer: ControlTracer | None = None
    csv_writer: StreamingCsvWriter | None = None
    csv_path: Path | None = None
    trace_path: Path | None = None
    try:
        print(f"[INFO] Config: {config_path}")
        print("[INFO] Connecting arm only; gripper control is disabled.")
        robot = make_franka_robot(
            str(config_path),
            with_gripper=False,
            max_velocity=DEOXYS_STREAMING_MAX_VELOCITY_RAD_S,
            max_acceleration=DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2,
        )
        robot.connect()
        anchor_state = robot.arm.get_state()
        anchor = anchor_state.end_effector_pose
        preview = planned_trajectory_metrics(anchor)
        if preview["max_target_step_m"] >= MAX_PUBLIC_IK_TRANSLATION_STEP_M:
            raise RuntimeError(
                "planned Cartesian step exceeds the public IK subtarget limit: "
                f"{preview['max_target_step_m']:.5f} m"
            )

        print(f"[STATE] anchor_position={_format_vector(anchor.position)} m")
        print(f"[STATE] anchor_quaternion_xyzw={_format_vector(anchor.quaternion_xyzw)}")
        print(
            "[PLAN] The figure eight is tilted 45 degrees inside base Y-Z. "
            "Its center lies on the vertical Z rotation axis."
        )
        print(
            "[PLAN] The drawing plane rotates continuously about that axis: "
            "30 degrees per eight, or 90 degrees every three eights."
        )
        print(
            f"[PLAN] duration_per_eight={EIGHT_DURATION_S:.2f}s "
            f"rate={RATE_HZ:.1f}Hz; motion continues until Ctrl-C; "
            f"peak discrete Cartesian speed="
            f"{preview['peak_discrete_target_speed_m_s']:.3f}m/s"
        )
        print(
            f"[PLAN] workspace_min={_format_vector(preview['target_min'])} m "
            f"workspace_max={_format_vector(preview['target_max'])} m"
        )
        print(
            f"[PLAN] discrete max target step={preview['max_target_step_m']:.5f}m "
            f"(< {MAX_PUBLIC_IK_TRANSLATION_STEP_M:.3f}m public IK bound)"
        )
        print(
            f"[PLAN] Deoxys q velocity limits="
            f"{_format_vector(DEOXYS_STREAMING_MAX_VELOCITY_RAD_S)}rad/s; "
            f"acceleration={DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2:.1f}rad/s^2; "
            f"jerk={DEOXYS_STREAMING_MAX_JERK_RAD_S3:.0f}rad/s^3"
        )
        print(
            "[PLAN] This curved path is acceleration-limited; velocity limits "
            "are safety ceilings, not a requirement to saturate a joint."
        )
        print("[WARNING] Clear that full box, verify the robot, and keep the stop reachable.")
        if input("Type DRAW8 to continue: ").strip() != "DRAW8":
            print("[CANCELLED] No figure-eight motion requested.")
            return 0

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_path = LOG_DIR / f"franka_ik_figure_eight_{stamp}.csv"
        trace_path = LOG_DIR / f"franka_ik_figure_eight_{stamp}.jsonl"
        tracer = ControlTracer(trace_path)
        tracer.event(
            "session.start",
            diagnostic="franka_ik_figure_eight",
            config=str(config_path),
            anchor=anchor,
            rate_hz=RATE_HZ,
            infinite=True,
            eight_duration_s=EIGHT_DURATION_S,
            y_amplitude_m=Y_AMPLITUDE_M,
            z_amplitude_m=Z_AMPLITUDE_M,
            figure_eight_tilt_rad=FIGURE_EIGHT_TILT_RAD,
            plane_rotation_per_eight_rad=PLANE_ROTATION_PER_EIGHT_RAD,
            plane_rotations=PLANE_ROTATIONS,
            deoxys_max_velocity_rad_s=DEOXYS_STREAMING_MAX_VELOCITY_RAD_S,
            deoxys_max_acceleration_rad_s2=DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2,
            deoxys_max_jerk_rad_s3=DEOXYS_STREAMING_MAX_JERK_RAD_S3,
            preview=preview,
        )
        csv_writer = StreamingCsvWriter(csv_path)
        stream_pose_targets(
            robot.arm,
            lambda _elapsed: anchor,
            phase="pre-hold",
            duration_s=PRE_HOLD_DURATION_S,
            rate_hz=RATE_HZ,
            include_endpoint=False,
            tracer=tracer,
            sample_callback=csv_writer.write,
            retain_samples=False,
        )
        print("[RUNNING] Drawing continuously. Press Ctrl-C to hold and stop.")
        stream_pose_targets(
            robot.arm,
            lambda elapsed: figure_eight_pose(
                anchor, elapsed, EIGHT_DURATION_S
            ),
            twist_fn=lambda elapsed: figure_eight_reference(
                anchor, elapsed, EIGHT_DURATION_S
            )[1],
            phase="continuous-8",
            duration_s=None,
            rate_hz=RATE_HZ,
            include_endpoint=False,
            tracer=tracer,
            sample_callback=csv_writer.write,
            retain_samples=False,
        )
        raise RuntimeError("infinite figure-eight stream returned unexpectedly")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Holding the latest joint reference before NO_CONTROL.")
        if tracer is not None:
            tracer.event("session.interrupted")
        if robot is not None:
            try:
                settled = hold_arm_until_settled(robot.arm)
                print(
                    "[STOP] Arm reached HOLDING."
                    if settled
                    else "[WARNING] Hold settling timed out; closing with backend safety handling."
                )
            except KeyboardInterrupt:
                print("\n[STOP] Second interrupt; closing immediately.")
            except Exception as exc:  # noqa: BLE001 - preserve final close.
                print(f"[WARNING] Hold settling reported: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - hardware diagnostics must expose failures.
        print(f"[ERROR] IK figure-eight diagnostic failed: {exc}", file=sys.stderr)
        if tracer is not None:
            tracer.event("session.error", exception=exc)
        return 1
    finally:
        if csv_writer is not None:
            csv_writer.close()
            print(f"[TELEMETRY] CSV: {csv_path}")
        if robot is not None:
            try:
                robot.close()
            except Exception as exc:  # noqa: BLE001 - preserve remaining cleanup.
                print(f"[WARNING] Franka shutdown reported: {exc}", file=sys.stderr)
        if tracer is not None:
            tracer.event("session.end")
            tracer.close()
            print(f"[TELEMETRY] full control trace: {trace_path}")


if __name__ == "__main__":
    raise SystemExit(main())
