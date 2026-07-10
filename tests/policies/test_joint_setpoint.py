import numpy as np
import pytest

from dex_teleop.policies.joint_setpoint import JointSetpointLimiter


JOINT_COUNT = 7
LOWER_LIMITS = -np.ones(JOINT_COUNT) * 2.0
UPPER_LIMITS = np.ones(JOINT_COUNT) * 2.0


def test_limiter_ramps_velocity_without_overshooting():
    limiter = JointSetpointLimiter(0.5, 1.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))

    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT),
        dt=0.05,
        lower_limits=LOWER_LIMITS,
        upper_limits=UPPER_LIMITS,
    )

    np.testing.assert_allclose(dq_cmd, np.ones(JOINT_COUNT) * 0.05)
    np.testing.assert_allclose(q_cmd, np.ones(JOINT_COUNT) * 0.0025)


def test_limiter_caps_velocity():
    limiter = JointSetpointLimiter(0.5, 100.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))

    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT),
        dt=0.1,
        lower_limits=LOWER_LIMITS,
        upper_limits=UPPER_LIMITS,
    )

    np.testing.assert_allclose(dq_cmd, np.ones(JOINT_COUNT) * 0.5)
    np.testing.assert_allclose(q_cmd, np.ones(JOINT_COUNT) * 0.05)


def test_limiter_caps_delayed_dt_and_stops_exactly_at_target():
    limiter = JointSetpointLimiter([0.5] * JOINT_COUNT, [100.0] * JOINT_COUNT, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))

    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT) * 0.01,
        dt=1.0,
        lower_limits=-np.ones(JOINT_COUNT),
        upper_limits=np.ones(JOINT_COUNT),
    )

    np.testing.assert_allclose(q_cmd, np.ones(JOINT_COUNT) * 0.01)
    np.testing.assert_allclose(dq_cmd, np.zeros(JOINT_COUNT))


def test_limiter_enforces_hard_dt_ceiling_above_configured_cap():
    limiter = JointSetpointLimiter(1.0, 1.0, 1.0)
    limiter.reset(np.zeros(JOINT_COUNT))

    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT),
        dt=1.0,
        lower_limits=LOWER_LIMITS,
        upper_limits=UPPER_LIMITS,
    )

    np.testing.assert_allclose(q_cmd, np.ones(JOINT_COUNT) * 0.01)
    np.testing.assert_allclose(dq_cmd, np.ones(JOINT_COUNT) * 0.1)


def test_limiter_does_not_overshoot_when_braking_from_existing_velocity():
    limiter = JointSetpointLimiter(1.0, 10.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))
    limiter.step(
        np.ones(JOINT_COUNT),
        dt=0.1,
        lower_limits=LOWER_LIMITS,
        upper_limits=UPPER_LIMITS,
    )

    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT) * 0.105,
        dt=0.1,
        lower_limits=LOWER_LIMITS,
        upper_limits=UPPER_LIMITS,
    )

    np.testing.assert_allclose(q_cmd, np.ones(JOINT_COUNT) * 0.105)
    np.testing.assert_allclose(dq_cmd, np.zeros(JOINT_COUNT))


def test_limiter_clips_joint_positions_and_stops_at_bounds():
    limiter = JointSetpointLimiter(10.0, 100.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))
    lower_limits = np.array([-0.05, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
    upper_limits = np.array([2.0, 0.05, 2.0, 2.0, 2.0, 2.0, 2.0])
    target = np.array([-1.0, 1.0, 0.2, -0.2, 0.1, -0.1, 0.0])

    q_cmd, dq_cmd = limiter.step(target, 0.1, lower_limits, upper_limits)

    np.testing.assert_allclose(
        q_cmd,
        np.array([-0.05, 0.05, 0.2, -0.2, 0.1, -0.1, 0.0]),
    )
    np.testing.assert_allclose(
        dq_cmd,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize(
    ("max_velocity", "max_acceleration"),
    [
        (0.5, 1.0),
        ([0.5] * JOINT_COUNT, np.ones(JOINT_COUNT)),
    ],
)
def test_limiter_accepts_scalar_or_seven_element_limit_configuration(
    max_velocity, max_acceleration
):
    limiter = JointSetpointLimiter(max_velocity, max_acceleration, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))

    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT), 0.05, LOWER_LIMITS, UPPER_LIMITS
    )

    np.testing.assert_allclose(q_cmd, np.ones(JOINT_COUNT) * 0.0025)
    np.testing.assert_allclose(dq_cmd, np.ones(JOINT_COUNT) * 0.05)


@pytest.mark.parametrize(
    ("max_velocity", "max_acceleration", "max_dt_s"),
    [
        (0.0, 1.0, 0.1),
        (1.0, -1.0, 0.1),
        (1.0, 1.0, 0.0),
        ([1.0] * 6, 1.0, 0.1),
        (1.0, [1.0] * 8, 0.1),
        ([1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0], 1.0, 0.1),
    ],
)
def test_limiter_rejects_invalid_limit_configuration(
    max_velocity, max_acceleration, max_dt_s
):
    with pytest.raises(ValueError):
        JointSetpointLimiter(max_velocity, max_acceleration, max_dt_s)


@pytest.mark.parametrize(
    "joint_positions",
    [
        np.zeros(6),
        np.zeros(8),
        np.array([0.0, 0.0, 0.0, np.inf, 0.0, 0.0, 0.0]),
    ],
)
def test_reset_rejects_non_finite_or_non_seven_joint_positions(joint_positions):
    limiter = JointSetpointLimiter(1.0, 1.0, 0.1)

    with pytest.raises(ValueError):
        limiter.reset(joint_positions)


def test_step_rejects_call_before_reset():
    limiter = JointSetpointLimiter(1.0, 1.0, 0.1)

    with pytest.raises(RuntimeError):
        limiter.step(np.zeros(JOINT_COUNT), 0.1, LOWER_LIMITS, UPPER_LIMITS)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_step_rejects_nonpositive_or_non_finite_dt(dt):
    limiter = JointSetpointLimiter(1.0, 1.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))

    with pytest.raises(ValueError):
        limiter.step(np.zeros(JOINT_COUNT), dt, LOWER_LIMITS, UPPER_LIMITS)


@pytest.mark.parametrize(
    ("target", "lower_limits", "upper_limits"),
    [
        (np.zeros(6), LOWER_LIMITS, UPPER_LIMITS),
        (np.zeros(JOINT_COUNT), np.zeros(6), UPPER_LIMITS),
        (np.zeros(JOINT_COUNT), LOWER_LIMITS, np.zeros(8)),
        (
            np.array([0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0]),
            LOWER_LIMITS,
            UPPER_LIMITS,
        ),
        (
            np.zeros(JOINT_COUNT),
            np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -np.inf, -2.0]),
            UPPER_LIMITS,
        ),
        (
            np.zeros(JOINT_COUNT),
            LOWER_LIMITS,
            np.array([2.0, 2.0, 2.0, 2.0, np.inf, 2.0, 2.0]),
        ),
    ],
)
def test_step_rejects_non_finite_or_non_seven_joint_data(
    target, lower_limits, upper_limits
):
    limiter = JointSetpointLimiter(1.0, 1.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))

    with pytest.raises(ValueError):
        limiter.step(target, 0.1, lower_limits, upper_limits)


def test_step_rejects_inverted_joint_bounds():
    limiter = JointSetpointLimiter(1.0, 1.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))
    lower_limits = LOWER_LIMITS.copy()
    upper_limits = UPPER_LIMITS.copy()
    lower_limits[3] = upper_limits[3] + 0.1

    with pytest.raises(ValueError):
        limiter.step(np.zeros(JOINT_COUNT), 0.1, lower_limits, upper_limits)


def test_step_returns_copies_of_internal_state():
    limiter = JointSetpointLimiter(1.0, 1.0, 0.1)
    limiter.reset(np.zeros(JOINT_COUNT))
    q_cmd, dq_cmd = limiter.step(
        np.ones(JOINT_COUNT), 0.1, LOWER_LIMITS, UPPER_LIMITS
    )
    q_cmd[:] = 99.0
    dq_cmd[:] = 99.0

    next_q_cmd, next_dq_cmd = limiter.step(
        np.ones(JOINT_COUNT), 0.1, LOWER_LIMITS, UPPER_LIMITS
    )

    np.testing.assert_allclose(next_q_cmd, np.ones(JOINT_COUNT) * 0.03)
    np.testing.assert_allclose(next_dq_cmd, np.ones(JOINT_COUNT) * 0.2)
