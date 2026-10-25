from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "test_deoxys_joint_control.py"
    spec = importlib.util.spec_from_file_location("test_deoxys_joint_control_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, float(duration))


class FakeErrors:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeInterface:
    def __init__(self):
        self.control_calls = []
        self._state = SimpleNamespace(
            q=[0.0] * 7,
            q_d=[0.0] * 7,
            dq=[0.0] * 7,
            dq_d=[0.0] * 7,
            ddq_d=[0.0] * 7,
            frame=1,
            robot_mode=2,
            control_command_success_rate=1.0,
            current_errors=FakeErrors(),
            last_motion_errors=FakeErrors(),
        )
        self._state_buffer = [self._state]

    @property
    def state_buffer_size(self):
        return len(self._state_buffer)

    def control(self, **kwargs):
        self.control_calls.append(kwargs)
        target = np.asarray(kwargs["action"][:7], dtype=float)
        previous = np.asarray(self._state.q, dtype=float)
        self._state.q = target.tolist()
        self._state.q_d = target.tolist()
        self._state.dq = ((target - previous) / 0.05).tolist()
        self._state.dq_d = ((target - previous) / 0.05).tolist()
        self._state.frame += 1


def test_default_selection_runs_non_disruptive_tests():
    module = load_module()

    assert module.parse_test_selection("") == [
        "state",
        "hold",
        "workspace-sweep",
        "all-joint-large-sweep",
    ]
    assert module.parse_test_selection("all") == [
        "state",
        "hold",
        "workspace-sweep",
        "all-joint-large-sweep",
    ]
    assert module.parse_test_selection("a") == [
        "state",
        "hold",
        "workspace-sweep",
        "all-joint-large-sweep",
    ]


def test_selection_accepts_numbers_names_and_orders_tests():
    module = load_module()

    assert module.parse_test_selection("5,1,bump") == [
        "state",
        "workspace-sweep",
        "watchdog",
    ]
    assert module.parse_test_selection("4,dance") == [
        "all-joint-large-sweep"
    ]


def test_choose_tests_prompts_when_no_cli_selection():
    module = load_module()
    output = []

    selected = module.choose_tests(
        input_fn=lambda _prompt: "2,3",
        print_fn=output.append,
    )

    assert selected == ["hold", "workspace-sweep"]
    assert any("Available raw Deoxys tests" in line for line in output)


def test_configure_joint_position_uses_absolute_full_period_linear_streaming():
    module = load_module()
    config = {
        "is_delta": True,
        "traj_interpolator_cfg": {
            "traj_interpolator_type": "SMOOTH_JOINT_POSITION",
            "time_fraction": 0.3,
        },
    }

    result = module.configure_joint_position(lambda kind: config)

    assert result is config
    assert config == {
        "is_delta": False,
        "traj_interpolator_cfg": {
            "traj_interpolator_type": "LINEAR_JOINT_POSITION",
            "time_fraction": 1.0,
        },
    }


def test_send_joint_position_never_controls_gripper():
    module = load_module()
    interface = FakeInterface()
    target = np.arange(7, dtype=float) / 100.0

    module.send_joint_position(interface, {"controller_type": "JOINT_POSITION"}, target)

    assert len(interface.control_calls) == 1
    call = interface.control_calls[0]
    assert call["controller_type"] == "JOINT_POSITION"
    assert call["action"] == target.tolist() + [0.0]
    assert call["termination"] is False
    assert call["control_gripper"] is False


def test_cosine_bump_starts_and_ends_at_zero_with_midpoint_at_amplitude():
    module = load_module()

    assert module.cosine_bump(0.0, 6.0, 0.01) == 0.0
    assert np.isclose(module.cosine_bump(3.0, 6.0, 0.01), 0.01)
    assert np.isclose(module.cosine_bump(6.0, 6.0, 0.01), 0.0)
    assert module.cosine_bump_velocity(0.0, 6.0, 0.01) == 0.0
    assert np.isclose(module.cosine_bump_velocity(1.5, 6.0, 0.01), np.pi / 600.0)


def test_all_joint_large_sweep_reaches_both_poses_and_stops_at_anchor():
    module = load_module()

    assert np.allclose(module.all_joint_large_sweep_offset(0.0), 0.0)
    assert np.allclose(
        module.all_joint_large_sweep_offset(module.ALL_JOINT_SWEEP_DURATION_S),
        0.0,
        atol=1e-12,
    )
    assert np.allclose(module.all_joint_large_sweep_velocity(0.0), 0.0)
    assert np.allclose(
        module.all_joint_large_sweep_velocity(module.ALL_JOINT_SWEEP_DURATION_S),
        0.0,
        atol=1e-12,
    )
    first_pose = module.all_joint_large_sweep_offset(
        0.25 * module.ALL_JOINT_SWEEP_DURATION_S
    )
    second_pose = module.all_joint_large_sweep_offset(
        0.75 * module.ALL_JOINT_SWEEP_DURATION_S
    )
    expected_pose = (
        module.ALL_JOINT_SWEEP_SIGNS * module.ALL_JOINT_SWEEP_AMPLITUDES_RAD
    )
    assert np.allclose(first_pose, expected_pose)
    assert np.allclose(second_pose, -expected_pose)

    elapsed = np.linspace(0.0, module.ALL_JOINT_SWEEP_DURATION_S, 2001)
    offsets = np.stack(
        [module.all_joint_large_sweep_offset(value) for value in elapsed]
    )
    assert np.allclose(np.max(offsets, axis=0), module.ALL_JOINT_SWEEP_AMPLITUDES_RAD)
    assert np.allclose(-np.min(offsets, axis=0), module.ALL_JOINT_SWEEP_AMPLITUDES_RAD)
    assert np.all(
        np.max(np.abs(offsets), axis=0)
        <= module.ALL_JOINT_SWEEP_AMPLITUDES_RAD + 1e-12
    )
    assert np.all(
        module.ALL_JOINT_SWEEP_COMMAND_PEAK_VELOCITY_RAD_S
        <= module.DEOXYS_STREAMING_MAX_VELOCITY_RAD_S + 1e-12
    )
    assert np.all(
        module.ALL_JOINT_SWEEP_COMMAND_PEAK_ACCELERATION_RAD_S2
        <= module.DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2 + 1e-12
    )
    assert np.all(
        module.ALL_JOINT_SWEEP_COMMAND_PEAK_JERK_RAD_S3
        <= module.DEOXYS_STREAMING_MAX_JERK_RAD_S3 + 1e-12
    )
    assert np.isclose(
        np.max(module.ALL_JOINT_SWEEP_COMMAND_PEAK_ACCELERATION_RAD_S2),
        module.DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2,
    )


def test_stream_runs_at_twenty_hz_and_records_state():
    module = load_module()
    interface = FakeInterface()
    clock = FakeClock()
    target = np.zeros(7)

    result = module.run_stream(
        interface,
        {"controller_type": "JOINT_POSITION"},
        lambda _elapsed: target,
        duration=0.2,
        label="test",
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        print_fn=lambda _line: None,
    )
    rate, max_gap = module._stream_rate(result)

    assert len(result.commands) == 4
    assert np.isclose(rate, 20.0)
    assert np.isclose(max_gap, 0.05)
    assert all(call["control_gripper"] is False for call in interface.control_calls)


def test_hold_and_workspace_sweep_pass_with_tracking_fake():
    module = load_module()
    interface = FakeInterface()
    clock = FakeClock()
    config = {"controller_type": "JOINT_POSITION"}

    hold = module.run_hold_test(
        interface,
        config,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        print_fn=lambda _line: None,
    )
    sweep = module.run_workspace_sweep_test(
        interface,
        config,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        print_fn=lambda _line: None,
    )

    assert hold.passed is True
    assert sweep.passed is True
    assert all(call["control_gripper"] is False for call in interface.control_calls)


def test_all_joint_large_sweep_passes_with_tracking_fake():
    module = load_module()
    interface = FakeInterface()
    clock = FakeClock()
    safe_anchor = [-0.1, -0.2, 0.1, -2.0, 0.1, 2.0, 0.5]
    interface._state.q = safe_anchor.copy()
    interface._state.q_d = safe_anchor.copy()

    result = module.run_all_joint_large_sweep_test(
        interface,
        {"controller_type": "JOINT_POSITION"},
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        print_fn=lambda _line: None,
    )

    assert result.passed is True
    assert "flips expected/dq_d/dq=" in result.details
    assert all(call["control_gripper"] is False for call in interface.control_calls)


def test_all_joint_large_sweep_rejects_insufficient_limit_margin():
    module = load_module()
    anchor = np.zeros(7)
    anchor[1] = module.PANDA_UPPER[1] - 0.1

    try:
        module.require_all_joint_large_sweep_margin(anchor)
    except RuntimeError as exc:
        assert "J2" in str(exc)
    else:
        raise AssertionError("expected joint-limit margin check to reject J2")


def test_hold_fails_on_velocity_spike_even_when_position_tracks():
    module = load_module()
    clock = FakeClock()

    class JitteringInterface(FakeInterface):
        def control(self, **kwargs):
            super().control(**kwargs)
            self._state.dq[0] = 0.1

    result = module.run_hold_test(
        JitteringInterface(),
        {"controller_type": "JOINT_POSITION"},
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        print_fn=lambda _line: None,
    )

    assert result.passed is False
    assert "max_abs_dq=0.10000rad/s" in result.details


def test_workspace_sweep_fails_on_wrong_direction_velocity():
    module = load_module()
    clock = FakeClock()

    class ReversingInterface(FakeInterface):
        def control(self, **kwargs):
            super().control(**kwargs)
            self._state.dq[0] *= -1.0

    result = module.run_workspace_sweep_test(
        ReversingInterface(),
        {"controller_type": "JOINT_POSITION"},
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        print_fn=lambda _line: None,
    )

    assert result.passed is False
    assert "wrong_direction=" in result.details


def test_send_no_control_repeats_explicit_dummy_messages():
    module = load_module()
    sent = []
    clock = FakeClock()

    class FakeAny:
        def __init__(self):
            self.packed = []

        def Pack(self, value):
            self.packed.append(value)

    class FakeControlMessage:
        class ControllerType:
            NO_CONTROL = 0

        def __init__(self):
            self.controller_type = None
            self.control_msg = FakeAny()
            self.timeout = None
            self.termination = None

        def SerializeToString(self):
            assert self.controller_type == self.ControllerType.NO_CONTROL
            assert len(self.control_msg.packed) == 1
            assert self.timeout == 0.2
            assert self.termination is False
            return b"no-control"

    class FakeDummyControllerMessage:
        pass

    protobuf_module = SimpleNamespace(
        FrankaControlMessage=FakeControlMessage,
        FrankaDummyControllerMessage=FakeDummyControllerMessage,
    )
    interface = SimpleNamespace(
        _publisher=SimpleNamespace(send=sent.append),
    )

    module.send_no_control(
        interface,
        protobuf_module,
        repeats=3,
        sleep_fn=clock.sleep,
    )

    assert sent == [b"no-control"] * 3
    assert np.isclose(clock.now, 0.1)


def test_telemetry_collects_full_rate_state_and_separates_direction(tmp_path):
    module = load_module()
    interface = FakeInterface()
    clock = FakeClock()
    collector = module.TelemetryCollector(interface, monotonic_fn=clock.monotonic)
    collector.record_command(
        label="workspace-positive",
        phase_elapsed_s=1.0,
        target=np.full(7, 0.1),
        sent_at=clock.monotonic(),
    )

    state = SimpleNamespace(
        q=[0.01] * 7,
        q_d=[0.01] * 7,
        dq=[-0.05] + [0.0] * 6,
        dq_d=[0.05] + [0.0] * 6,
        ddq_d=[2.0] + [0.0] * 6,
        frame=2,
        robot_mode=2,
        control_command_success_rate=1.0,
        current_errors=FakeErrors(),
        last_motion_errors=FakeErrors(),
    )
    interface._state_buffer.append(state)
    clock.now = 0.01
    collector._drain_available()

    summaries = collector.direction_summaries()
    assert len(collector.states) == 1
    assert len(summaries) == 1
    assert summaries[0].desired_wrong_direction == 0
    assert summaries[0].measured_wrong_direction == 1
    assert summaries[0].max_abs_ddq_d == 2.0

    csv_path, jsonl_path = collector.save(tmp_path)
    assert "ddq_d_1" in csv_path.read_text()
    json_lines = jsonl_path.read_text().splitlines()
    assert len(json_lines) == 2
    assert '"kind":"command"' in json_lines[0]
    assert '"kind":"state"' in json_lines[1]


def test_telemetry_reports_each_large_sweep_joint_direction():
    module = load_module()
    interface = FakeInterface()
    clock = FakeClock()
    collector = module.TelemetryCollector(interface, monotonic_fn=clock.monotonic)
    phase_elapsed = 4.0
    expected = module.all_joint_large_sweep_velocity(phase_elapsed)
    collector.record_command(
        label="all-joint-large-sweep",
        phase_elapsed_s=phase_elapsed,
        target=np.asarray(interface._state.q_d)
        + module.all_joint_large_sweep_offset(phase_elapsed),
        sent_at=clock.monotonic(),
    )
    interface._state_buffer.append(
        SimpleNamespace(
            q=[0.0] * 7,
            q_d=[0.0] * 7,
            dq=expected.tolist(),
            dq_d=expected.tolist(),
            ddq_d=[0.3] * 7,
            frame=2,
            robot_mode=2,
            control_command_success_rate=1.0,
            current_errors=FakeErrors(),
            last_motion_errors=FakeErrors(),
        )
    )
    collector._drain_available()

    summaries = collector.direction_summaries()

    assert len(summaries) == 7
    assert summaries[0].label == "all-joint-large-sweep-J1"
    assert summaries[-1].label == "all-joint-large-sweep-J7"
    assert all(summary.desired_wrong_direction == 0 for summary in summaries)
    assert all(summary.measured_wrong_direction == 0 for summary in summaries)
    assert all(np.isclose(summary.max_abs_ddq_d, 0.3) for summary in summaries)
