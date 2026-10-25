from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dex_teleop.core.types import ArmState, Pose, Twist


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "test_franka_ik_figure_eight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_franka_ik_figure_eight_script", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, float(duration))


class TrackingArm:
    dof = 7

    def __init__(self, clock, anchor):
        self.clock = clock
        self.pose = anchor
        self.q = np.array(
            [anchor.position[0], anchor.position[1], anchor.position[2], 0, 0, 0, 0],
            dtype=float,
        )
        self.dq_d = np.zeros(7)
        self.calls = []
        self.last_ik_joint_positions = None
        self.last_ik_joint_velocities = None
        self.last_commanded_joint_positions = None
        self.last_commanded_joint_velocities = None
        self.interface = SimpleNamespace(
            last_q_d=self.q.copy(),
            last_state=SimpleNamespace(
                dq_d=np.zeros(7),
                ddq_d=np.zeros(7),
            ),
        )

    def get_state(self):
        return ArmState(
            self.q,
            np.zeros(7),
            self.pose,
            self.clock.monotonic(),
        )

    def ik_joint_position(self, pose, *, twist_ref=None, delta=False, dt):
        self.calls.append((pose, twist_ref, delta, dt))
        self.pose = pose
        previous_q = self.q.copy()
        previous_velocity_reference = self.dq_d.copy()
        ik_q = np.array(
            [pose.position[0], pose.position[1], pose.position[2], 0, 0, 0, 0],
            dtype=float,
        )
        velocity_reference = np.zeros(7)
        if twist_ref is not None:
            velocity_reference[:3] = twist_ref.linear
        self.q = previous_q + 0.5 * (
            previous_velocity_reference + velocity_reference
        ) * dt
        self.dq_d = velocity_reference.copy()
        self.last_ik_joint_positions = ik_q
        self.last_ik_joint_velocities = velocity_reference.copy()
        self.last_commanded_joint_positions = self.q.copy()
        self.last_commanded_joint_velocities = velocity_reference.copy()
        self.interface.last_q_d = self.q.copy()
        self.interface.last_state = SimpleNamespace(
            dq_d=self.dq_d.copy(),
            ddq_d=(self.dq_d - previous_velocity_reference) / dt,
        )


def test_figure_eight_is_diagonal_in_yz_and_rotates_about_its_center_z_axis():
    module = load_module()

    center = module.figure_eight_offset_for_angle(0.0)
    top = module.figure_eight_offset_for_angle(np.pi / 2.0)
    crossing = module.figure_eight_offset_for_angle(np.pi)
    bottom = module.figure_eight_offset_for_angle(3.0 * np.pi / 2.0)
    end = module.figure_eight_offset_for_angle(2.0 * np.pi)

    np.testing.assert_allclose(center, 0.0, atol=1e-12)
    np.testing.assert_allclose(crossing, 0.0, atol=1e-12)
    np.testing.assert_allclose(end, 0.0, atol=1e-12)
    expected_top = module.Z_AMPLITUDE_M / np.sqrt(2.0)
    assert np.isclose(top[1], -expected_top)
    assert np.isclose(top[2], expected_top)
    assert np.isclose(bottom[1], expected_top)
    assert np.isclose(bottom[2], -expected_top)

    angles = np.linspace(0.0, 2.0 * np.pi, 4001)
    offsets = np.stack(
        [module.figure_eight_offset_for_angle(angle) for angle in angles]
    )
    assert np.allclose(offsets[:, 0], 0.0)

    original = module.figure_eight_offset_for_angle(np.pi / 4.0, 0.0)
    quarter_turn = module.figure_eight_offset_for_angle(
        np.pi / 4.0, np.pi / 2.0
    )
    np.testing.assert_allclose(quarter_turn, [-original[1], 0.0, original[2]], atol=1e-12)
    assert np.isclose(np.linalg.norm(quarter_turn), np.linalg.norm(original))


def test_figure_eight_twist_is_analytic_and_continuous_across_cycle_boundaries():
    module = load_module()
    anchor = Pose([0.5, -0.1, 0.4], [0, 0, 0, 1])
    duration = module.DRAW_DURATION_S

    for elapsed in (0.2 * duration, 0.5 * duration, 0.8 * duration):
        pose, twist = module.figure_eight_reference(anchor, elapsed, duration)
        epsilon = 1e-5
        before = module.figure_eight_pose(anchor, elapsed - epsilon, duration)
        after = module.figure_eight_pose(anchor, elapsed + epsilon, duration)
        finite_difference = (after.position - before.position) / (2.0 * epsilon)
        np.testing.assert_allclose(twist.linear, finite_difference, atol=1e-8)
        np.testing.assert_allclose(twist.angular, 0.0)
        np.testing.assert_allclose(pose.position, module.figure_eight_pose(anchor, elapsed, duration).position)

    for elapsed in (0.0, duration, 2.0 * duration, 3.0 * duration):
        pose, twist = module.figure_eight_reference(anchor, elapsed, duration)
        np.testing.assert_allclose(pose.position, anchor.position, atol=1e-12)
        assert np.linalg.norm(twist.linear) > 0.1

    epsilon = 1e-6
    _, before = module.figure_eight_reference(anchor, duration - epsilon, duration)
    _, after = module.figure_eight_reference(anchor, duration + epsilon, duration)
    np.testing.assert_allclose(before.linear, after.linear, atol=1e-6)

    phase_time = 0.125 * duration
    first, _ = module.figure_eight_reference(anchor, phase_time, duration)
    fourth, _ = module.figure_eight_reference(
        anchor, phase_time + 3.0 * duration, duration
    )
    first_offset = first.position - anchor.position
    fourth_offset = fourth.position - anchor.position
    np.testing.assert_allclose(
        fourth_offset,
        [-first_offset[1], first_offset[0], first_offset[2]],
        atol=1e-12,
    )


def test_discrete_plan_is_closed_and_below_public_ik_step_limit():
    module = load_module()
    anchor = Pose([0.5, -0.1, 0.4], [0, 0, 0, 1])

    preview = module.planned_trajectory_metrics(anchor)
    first = module.figure_eight_pose(anchor, 0.0, module.DRAW_DURATION_S)
    last = module.figure_eight_pose(
        anchor, module.DRAW_DURATION_S, module.DRAW_DURATION_S
    )

    np.testing.assert_allclose(first.position, anchor.position)
    np.testing.assert_allclose(last.position, anchor.position, atol=1e-12)
    np.testing.assert_allclose(first.quaternion_xyzw, anchor.quaternion_xyzw)
    np.testing.assert_allclose(last.quaternion_xyzw, anchor.quaternion_xyzw)
    assert preview["max_target_step_m"] < module.MAX_PUBLIC_IK_TRANSLATION_STEP_M
    assert preview["max_target_step_m"] < 0.01
    assert 0.15 < preview["peak_discrete_target_speed_m_s"] < 0.17
    assert preview["target_min"][0] < anchor.position[0] - 0.03
    assert preview["target_max"][0] > anchor.position[0] + 0.03
    assert preview["target_count"] == int(
        module.PREVIEW_EIGHT_COUNT
        * module.EIGHT_DURATION_S
        * module.RATE_HZ
    ) + 1


def test_infinite_stream_writes_samples_without_retaining_an_unbounded_list():
    module = load_module()
    clock = FakeClock()
    anchor = Pose([0.5, -0.1, 0.4], [0, 0, 0, 1])
    arm = TrackingArm(clock, anchor)
    received = []

    def collect(sample):
        received.append(sample)
        if len(received) == 7:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        module.stream_pose_targets(
            arm,
            lambda elapsed: module.figure_eight_pose(
                anchor, elapsed, module.EIGHT_DURATION_S
            ),
            twist_fn=lambda elapsed: module.figure_eight_reference(
                anchor, elapsed, module.EIGHT_DURATION_S
            )[1],
            phase="continuous-8",
            duration_s=None,
            rate_hz=module.RATE_HZ,
            include_endpoint=False,
            sample_callback=collect,
            retain_samples=False,
            monotonic_fn=clock.monotonic,
            sleep_fn=clock.sleep,
            print_fn=lambda _line: None,
        )

    assert len(received) == 7
    assert len(arm.calls) == 7
    assert received[-1].scheduled_elapsed_s == pytest.approx(0.3)


def test_interrupt_hold_repeats_until_the_arm_is_physically_settled():
    module = load_module()
    clock = FakeClock()

    class SettlingArm:
        def __init__(self):
            self.motion_state = module.ArmMotionState.HOLDING
            self.is_settled = False
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            self.motion_state = module.ArmMotionState.HOLDING
            self.is_settled = self.stop_calls >= 3

    arm = SettlingArm()
    settled = module.hold_arm_until_settled(
        arm,
        timeout_s=1.0,
        rate_hz=20.0,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )

    assert settled is True
    assert arm.stop_calls == 3


def test_stream_uses_public_absolute_ik_and_records_complete_joint_path():
    module = load_module()
    clock = FakeClock()
    anchor = Pose([0.5, -0.1, 0.4], [0, 0, 0, 1])
    arm = TrackingArm(clock, anchor)

    samples = module.stream_pose_targets(
        arm,
        lambda elapsed: module.figure_eight_pose(anchor, elapsed, 0.2),
        phase="draw-8",
        duration_s=0.2,
        rate_hz=20.0,
        include_endpoint=True,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        print_fn=lambda _line: None,
    )

    assert len(samples) == 5
    assert len(arm.calls) == 5
    assert all(delta is False and np.isclose(dt, 0.05) for _, _, delta, dt in arm.calls)
    assert np.isclose(samples[-1].actual_elapsed_s, 0.2)
    np.testing.assert_allclose(samples[-1].target_position, anchor.position, atol=1e-12)
    assert all(sample.ik_joint_positions is not None for sample in samples)
    assert all(sample.ik_joint_velocities is not None for sample in samples)
    assert all(sample.commanded_joint_positions is not None for sample in samples)
    assert all(sample.commanded_joint_velocities is not None for sample in samples)
    assert all(sample.backend_desired_joint_positions is not None for sample in samples)
    assert all(
        sample.backend_desired_joint_velocities is not None for sample in samples
    )
    assert all(
        sample.backend_desired_joint_accelerations is not None for sample in samples
    )


def test_summary_passes_for_exact_tracking_and_csv_contains_all_joint_vectors(tmp_path):
    module = load_module()
    clock = FakeClock()
    anchor = Pose([0.5, -0.1, 0.4], [0, 0, 0, 1])
    arm = TrackingArm(clock, anchor)
    samples = module.stream_pose_targets(
        arm,
        lambda elapsed: module.figure_eight_pose(
            anchor, elapsed, module.DRAW_DURATION_S
        ),
        phase="draw-8",
        duration_s=module.DRAW_DURATION_S,
        rate_hz=module.RATE_HZ,
        include_endpoint=True,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        print_fn=lambda _line: None,
    )
    final_state = arm.get_state()

    summary = module.summarize_run(samples, anchor, final_state)
    path = tmp_path / "figure-eight.csv"
    module.write_csv(path, samples)

    assert summary.passed is True
    assert summary.command_rate_hz == 20.0
    assert summary.max_joint_reference_integration_error_rad < 1e-12
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    for name in ("q1", "dq7", "q_ik7", "dq_ik7", "q_cmd7", "dq_cmd7", "q_d7", "dq_d7", "ddq_d7"):
        assert name in header
