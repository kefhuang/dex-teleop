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
    """Map vector components between named axes using signed source specs.

    ``mapping`` maps each output axis name to an input axis spec such as ``"+z"``
    or ``"-x"``. Unsigned specs are treated as positive.
    """
    values = np.asarray(vector, dtype=float)
    if values.shape != (len(input_axes),):
        raise ValueError(f"vector must have shape ({len(input_axes)},), got {values.shape}")

    input_index = {axis: i for i, axis in enumerate(input_axes)}
    output: list[float] = []
    for out_axis in output_axes:
        if out_axis not in mapping:
            raise ValueError(f"Missing mapping for output axis {out_axis!r}")
        sign, in_axis = _parse_signed_axis(mapping[out_axis])
        if in_axis not in input_index:
            raise ValueError(f"Unknown source axis {in_axis!r} for output axis {out_axis!r}")
        output.append(sign * values[input_index[in_axis]])
    return np.asarray(output, dtype=float)


def apply_deadzone(vector: np.ndarray, threshold: float) -> np.ndarray:
    """Return a copy with components of absolute value <= threshold zeroed."""
    limit = float(threshold)
    if limit < 0.0:
        raise ValueError("threshold must be >= 0.0")
    values = np.asarray(vector, dtype=float).copy()
    values[np.abs(values) <= limit] = 0.0
    return values


def limit_components(vector: np.ndarray, max_abs: float) -> np.ndarray:
    """Clamp each component to ``[-max_abs, max_abs]``."""
    limit = float(max_abs)
    if limit < 0.0:
        raise ValueError("max_abs must be >= 0.0")
    return np.clip(np.asarray(vector, dtype=float), -limit, limit)


def clamp_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp vector magnitude while preserving direction."""
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    limit = float(max_norm)
    if limit < 0.0:
        raise ValueError("max_norm must be >= 0.0")
    if norm == 0.0 or norm <= limit:
        return values.copy()
    return values * (limit / norm)


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
    if values.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {values.shape}")
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
    """Return roll/pitch/yaw delta from anchor to current for xyzw quaternions."""
    anchor = _normalize_quat(anchor_xyzw)
    current = _normalize_quat(current_xyzw)
    delta = _quat_multiply(current, _quat_conjugate(anchor))
    return _quat_to_rpy(_normalize_quat(delta))
