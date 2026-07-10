"""Velocity- and acceleration-limited seven-joint setpoint shaping."""

from __future__ import annotations

from typing import Any

import numpy as np


_JOINT_COUNT = 7
_HARD_MAX_DT_S = 0.10


def _joint_vector(values: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {_JOINT_COUNT} finite values") from exc
    if vector.shape != (_JOINT_COUNT,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {_JOINT_COUNT} finite values")
    return vector.copy()


def _positive_joint_limit(values: Any, name: str) -> np.ndarray:
    try:
        limit = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive finite scalar or {_JOINT_COUNT}-element vector"
        ) from exc
    if limit.shape == ():
        limit = np.full(_JOINT_COUNT, float(limit), dtype=float)
    elif limit.shape == (_JOINT_COUNT,):
        limit = limit.copy()
    else:
        raise ValueError(
            f"{name} must be a positive finite scalar or {_JOINT_COUNT}-element vector"
        )
    if not np.all(np.isfinite(limit)) or np.any(limit <= 0.0):
        raise ValueError(
            f"{name} must be a positive finite scalar or {_JOINT_COUNT}-element vector"
        )
    return limit


class JointSetpointLimiter:
    """Own and incrementally shape a safe seven-joint position setpoint."""

    def __init__(
        self,
        max_velocity: Any,
        max_acceleration: Any,
        max_dt_s: float,
    ) -> None:
        self.max_velocity = _positive_joint_limit(max_velocity, "max_velocity")
        self.max_acceleration = _positive_joint_limit(
            max_acceleration, "max_acceleration"
        )
        try:
            self.max_dt_s = float(max_dt_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_dt_s must be positive and finite") from exc
        if not np.isfinite(self.max_dt_s) or self.max_dt_s <= 0.0:
            raise ValueError("max_dt_s must be positive and finite")

        self._q_cmd: np.ndarray | None = None
        self._dq_cmd: np.ndarray | None = None

    def reset(self, joint_positions: Any) -> None:
        """Reset the owned command state to measured joint positions at rest."""
        self._q_cmd = _joint_vector(joint_positions, "joint_positions")
        self._dq_cmd = np.zeros(_JOINT_COUNT, dtype=float)

    def step(
        self,
        target: Any,
        dt: float,
        lower_limits: Any,
        upper_limits: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the shaped position and velocity commands by one cycle."""
        if self._q_cmd is None or self._dq_cmd is None:
            raise RuntimeError("reset must be called before step")

        try:
            dt_s = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt must be positive and finite") from exc
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt must be positive and finite")

        target_vector = _joint_vector(target, "target")
        lower_vector = _joint_vector(lower_limits, "lower_limits")
        upper_vector = _joint_vector(upper_limits, "upper_limits")
        if np.any(lower_vector > upper_vector):
            raise ValueError("lower_limits must not exceed upper_limits")

        effective_dt = min(dt_s, self.max_dt_s, _HARD_MAX_DT_S)
        error = target_vector - self._q_cmd
        desired_dq = np.clip(
            error / effective_dt, -self.max_velocity, self.max_velocity
        )
        max_delta_dq = self.max_acceleration * effective_dt
        next_dq = self._dq_cmd + np.clip(
            desired_dq - self._dq_cmd, -max_delta_dq, max_delta_dq
        )
        next_q = self._q_cmd + next_dq * effective_dt

        overshot = np.sign(target_vector - self._q_cmd) != np.sign(
            target_vector - next_q
        )
        next_q[overshot] = target_vector[overshot]
        next_dq[overshot] = 0.0

        next_q = np.clip(next_q, lower_vector, upper_vector)
        next_dq[(next_q <= lower_vector) | (next_q >= upper_vector)] = 0.0

        self._q_cmd = next_q
        self._dq_cmd = next_dq
        return self._q_cmd.copy(), self._dq_cmd.copy()
