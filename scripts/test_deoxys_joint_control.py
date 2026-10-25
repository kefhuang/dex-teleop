#!/usr/bin/env python3
"""Interactive raw-Deoxys JOINT_POSITION hardware diagnostic.

This script deliberately bypasses dex-teleop robot, IK, teleoperation, and VR
layers.  It exercises ``deoxys.franka_interface.FrankaInterface`` directly and
checks state publication, an immutable measured-position hold, a visible
workspace-scale joint sweep, a coordinated seven-joint large sweep, and
(optionally) the backend missing-message watchdog.

The default interactive ``all`` selection runs only the non-disruptive tests.
The watchdog test is separate because the current backend exits after twenty
missing policy messages and may then be restarted by ``auto_arm.sh``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEOXYS_ROOT = REPO_ROOT / "dependency" / "deoxys_control" / "deoxys"
DEFAULT_CONFIG = "franka-three.yml"
TELEMETRY_DIR = REPO_ROOT / "logs"

STATE_TEST = "state"
HOLD_TEST = "hold"
WORKSPACE_SWEEP_TEST = "workspace-sweep"
ALL_JOINT_LARGE_SWEEP_TEST = "all-joint-large-sweep"
WATCHDOG_TEST = "watchdog"
TEST_ORDER = (
    STATE_TEST,
    HOLD_TEST,
    WORKSPACE_SWEEP_TEST,
    ALL_JOINT_LARGE_SWEEP_TEST,
    WATCHDOG_TEST,
)
NON_DISRUPTIVE_TESTS = (
    STATE_TEST,
    HOLD_TEST,
    WORKSPACE_SWEEP_TEST,
    ALL_JOINT_LARGE_SWEEP_TEST,
)
WRITE_TESTS = frozenset(
    (HOLD_TEST, WORKSPACE_SWEEP_TEST, ALL_JOINT_LARGE_SWEEP_TEST, WATCHDOG_TEST)
)

CONTROL_RATE_HZ = 20.0
CONTROL_PERIOD_S = 1.0 / CONTROL_RATE_HZ
STATE_SAMPLE_DURATION_S = 2.0
STATE_TIMEOUT_S = 5.0
HOLD_DURATION_S = 3.0
RETURN_HOLD_DURATION_S = 1.0
WORKSPACE_SWEEP_DURATION_S = 6.0
WORKSPACE_SWEEP_AMPLITUDE_RAD = 0.15
WORKSPACE_SWEEP_INDEX = 0  # zero-based J1; moves the arm through the workspace
ALL_JOINT_SWEEP_AMPLITUDES_RAD = np.array(
    [0.55, 0.40, 0.50, 0.35, 0.50, 0.45, 0.65]
)
ALL_JOINT_SWEEP_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# These are the effective limits in StreamingJointPositionTrajInterpolator:
# 0.5 * libfranka::kMaxJointVelocity, 1 rad/s^2, and 100 rad/s^3.  Derive the
# shortest shared cosine-bump duration that keeps the 20 Hz reference itself
# inside all three bounds.  Acceleration on J7 is the active constraint for the
# current excursions; merely targeting the velocity bounds would ask the
# backend tracker to absorb an acceleration violation as tracking lag.
DEOXYS_STREAMING_MAX_VELOCITY_RAD_S = np.array(
    [1.0, 0.5, 0.75, 0.625, 1.5, 0.75, 1.5]
)
DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2 = 1.0
DEOXYS_STREAMING_MAX_JERK_RAD_S3 = 100.0
ALL_JOINT_SWEEP_HALF_DURATION_S = float(
    max(
        np.max(
            math.pi
            * ALL_JOINT_SWEEP_AMPLITUDES_RAD
            / DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
        ),
        np.sqrt(
            np.max(
                2.0
                * math.pi**2
                * ALL_JOINT_SWEEP_AMPLITUDES_RAD
                / DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2
            )
        ),
        np.cbrt(
            np.max(
                4.0
                * math.pi**3
                * ALL_JOINT_SWEEP_AMPLITUDES_RAD
                / DEOXYS_STREAMING_MAX_JERK_RAD_S3
            )
        ),
    )
)
ALL_JOINT_SWEEP_DURATION_S = 2.0 * ALL_JOINT_SWEEP_HALF_DURATION_S
ALL_JOINT_SWEEP_COMMAND_PEAK_VELOCITY_RAD_S = (
    math.pi
    * ALL_JOINT_SWEEP_AMPLITUDES_RAD
    / ALL_JOINT_SWEEP_HALF_DURATION_S
)
ALL_JOINT_SWEEP_COMMAND_PEAK_ACCELERATION_RAD_S2 = (
    2.0
    * math.pi**2
    * ALL_JOINT_SWEEP_AMPLITUDES_RAD
    / ALL_JOINT_SWEEP_HALF_DURATION_S**2
)
ALL_JOINT_SWEEP_COMMAND_PEAK_JERK_RAD_S3 = (
    4.0
    * math.pi**3
    * ALL_JOINT_SWEEP_AMPLITUDES_RAD
    / ALL_JOINT_SWEEP_HALF_DURATION_S**3
)
ALL_JOINT_SWEEP_RETURN_HOLD_DURATION_S = 1.5
ALL_JOINT_SWEEP_LIMIT_MARGIN_RAD = 0.08
MIN_ALL_JOINT_SWEEP_RESPONSE_FRACTION = 0.75
MAX_ALL_JOINT_SWEEP_DESIRED_SPEED_LIMIT_TOLERANCE_RAD_S = 0.02
MAX_ALL_JOINT_SWEEP_MEASURED_SPEED_LIMIT_TOLERANCE_RAD_S = 0.12
MAX_ALL_JOINT_SWEEP_TRACKING_ERROR_RAD = 0.03
MAX_ALL_JOINT_SWEEP_DESIRED_ACCELERATION_RAD_S2 = 1.05
MAX_ALL_JOINT_SWEEP_RETURN_SPEED_RAD_S = 0.03
MAX_ALL_JOINT_SWEEP_FLIP_DIFFERENCE = 1
WATCHDOG_HOLD_DURATION_S = 1.0
WATCHDOG_SILENCE_S = 1.5
MIN_STATE_RATE_HZ = 50.0
MIN_CONTROL_RATE_HZ = 15.0
MAX_CONTROL_RATE_HZ = 25.0
MAX_CONTROL_GAP_S = 0.15
MIN_CONTROL_SUCCESS_RATE = 0.90
MAX_HOLD_DEVIATION_RAD = 0.01
MAX_HOLD_SPEED_RAD_S = 0.02
MIN_SWEEP_RESPONSE_RAD = 0.12
MAX_SWEEP_RESPONSE_RAD = 0.18
MAX_NON_SWEEP_SPEED_RAD_S = 0.05
SWEEP_SPEED_LIMIT_MULTIPLIER = 2.0
MAX_SWEEP_VELOCITY_ERROR_RAD_S = 0.04
MAX_SWEEP_ACCELERATION_RAD_S2 = 0.75
PANDA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# The vendored protobufs may have been generated by an older protoc.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


class StateSnapshot(NamedTuple):
    received_at: float
    q: np.ndarray
    q_d: np.ndarray
    dq: np.ndarray
    dq_d: np.ndarray
    ddq_d: np.ndarray
    frame: int
    robot_mode: int
    success_rate: float
    current_errors: tuple[str, ...]
    last_motion_errors: tuple[str, ...]


class StreamResult(NamedTuple):
    commands: tuple[np.ndarray, ...]
    samples: tuple[StateSnapshot, ...]
    sent_at: tuple[float, ...]


class TestResult(NamedTuple):
    name: str
    passed: bool
    details: str


class TelemetryCommand(NamedTuple):
    sequence: int
    sent_at: float
    label: str
    phase_elapsed_s: float
    target: tuple[float, ...]


class TelemetryState(NamedTuple):
    received_at: float
    state: Any
    command: TelemetryCommand | None


class DirectionTelemetry(NamedTuple):
    label: str
    samples: int
    expected_flips: int
    desired_flips: int
    measured_flips: int
    desired_wrong_direction: int
    measured_wrong_direction: int
    max_abs_ddq_d: float


class TelemetryCollector:
    """Capture every state-buffer entry and the command active when it arrived."""

    def __init__(
        self,
        interface: Any,
        *,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interface = interface
        self._monotonic_fn = monotonic_fn
        self._next_state_index = int(getattr(interface, "state_buffer_size", 0))
        self._commands: list[TelemetryCommand] = []
        self._states: list[TelemetryState] = []
        self._active_command: TelemetryCommand | None = None
        self._command_sequence = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def commands(self) -> tuple[TelemetryCommand, ...]:
        return tuple(self._commands)

    @property
    def states(self) -> tuple[TelemetryState, ...]:
        return tuple(self._states)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry collector is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="deoxys-state-telemetry",
            daemon=True,
        )
        self._thread.start()

    def record_command(
        self,
        *,
        label: str,
        phase_elapsed_s: float,
        target: np.ndarray,
        sent_at: float,
    ) -> None:
        values = np.asarray(target, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("telemetry command target must be a finite shape-(7,) vector")
        with self._lock:
            self._command_sequence += 1
            command = TelemetryCommand(
                self._command_sequence,
                float(sent_at),
                str(label),
                float(phase_elapsed_s),
                tuple(float(value) for value in values),
            )
            self._commands.append(command)
            self._active_command = command

    def record_marker(self, label: str, *, sent_at: float) -> None:
        with self._lock:
            self._command_sequence += 1
            command = TelemetryCommand(
                self._command_sequence,
                float(sent_at),
                str(label),
                0.0,
                (float("nan"),) * 7,
            )
            self._commands.append(command)
            self._active_command = command

    def _drain_available(self) -> None:
        state_buffer = getattr(self._interface, "_state_buffer", ())
        while self._next_state_index < len(state_buffer):
            state = state_buffer[self._next_state_index]
            self._next_state_index += 1
            with self._lock:
                command = self._active_command
            self._states.append(
                TelemetryState(self._monotonic_fn(), state, command)
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._drain_available()
            self._stop_event.wait(0.001)
        self._drain_available()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._drain_available()

    def save(self, directory: Path = TELEMETRY_DIR) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = directory / f"deoxys_joint_control_{stamp}"
        csv_path = base.with_suffix(".csv")
        jsonl_path = base.with_suffix(".jsonl")
        _write_telemetry_csv(csv_path, self._states)
        _write_telemetry_jsonl(jsonl_path, self._states, self._commands)
        return csv_path, jsonl_path

    def direction_summaries(self) -> tuple[DirectionTelemetry, ...]:
        return _direction_telemetry(self._states)


TELEMETRY_ARRAY_FIELDS = {
    "q": 7,
    "q_d": 7,
    "dq": 7,
    "dq_d": 7,
    "ddq_d": 7,
    "tau_J": 7,
    "tau_J_d": 7,
    "dtau_J": 7,
    "tau_ext_hat_filtered": 7,
    "joint_contact": 7,
    "joint_collision": 7,
    "O_T_EE": 16,
    "O_T_EE_d": 16,
    "O_dP_EE_d": 6,
    "O_F_ext_hat_K": 6,
    "K_F_ext_hat_K": 6,
}


def _telemetry_array(state: Any, name: str, width: int) -> np.ndarray:
    try:
        value = np.asarray(getattr(state, name), dtype=float)
    except (AttributeError, TypeError, ValueError):
        value = np.empty(0, dtype=float)
    if value.shape != (width,):
        return np.full(width, np.nan, dtype=float)
    return value.copy()


def _state_time_seconds(state: Any) -> float:
    try:
        value = float(state.time.toSec)
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _command_json(command: TelemetryCommand | None) -> dict[str, Any] | None:
    if command is None:
        return None
    return {
        "sequence": command.sequence,
        "sent_monotonic_s": command.sent_at,
        "label": command.label,
        "phase_elapsed_s": command.phase_elapsed_s,
        "target": list(command.target),
    }


def _jsonable_fallback(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_fallback(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable_fallback(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _protobuf_json(state: Any) -> dict[str, Any]:
    try:
        from google.protobuf.json_format import MessageToDict  # noqa: PLC0415

        try:
            return MessageToDict(
                state,
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
            )
        except TypeError:
            try:
                return MessageToDict(
                    state,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
            except TypeError:
                return MessageToDict(state, preserving_proto_field_name=True)
    except (AttributeError, ImportError, TypeError, ValueError):
        fallback = _jsonable_fallback(state)
        return fallback if isinstance(fallback, dict) else {"value": fallback}


def _write_telemetry_jsonl(
    path: Path,
    states: list[TelemetryState],
    commands: list[TelemetryCommand],
) -> None:
    events: list[tuple[float, int, dict[str, Any]]] = []
    for command in commands:
        events.append(
            (
                command.sent_at,
                0,
                {"kind": "command", **(_command_json(command) or {})},
            )
        )
    for record in states:
        events.append(
            (
                record.received_at,
                1,
                {
                    "kind": "state",
                    "received_monotonic_s": record.received_at,
                    "active_command": _command_json(record.command),
                    "state": _protobuf_json(record.state),
                },
            )
        )
    events.sort(key=lambda item: (item[0], item[1]))
    with path.open("w", encoding="utf-8") as stream:
        for _, _, event in events:
            stream.write(json.dumps(event, allow_nan=True, separators=(",", ":")))
            stream.write("\n")


def _write_telemetry_csv(path: Path, states: list[TelemetryState]) -> None:
    base_fields = [
        "received_monotonic_s",
        "robot_time_s",
        "frame",
        "robot_mode",
        "control_command_success_rate",
        "command_sequence",
        "command_sent_monotonic_s",
        "command_age_s",
        "phase",
        "phase_elapsed_s",
        "current_errors",
        "last_motion_errors",
    ]
    command_fields = [f"q_cmd_{joint + 1}" for joint in range(7)]
    state_fields = [
        f"{name}_{index + 1}"
        for name, width in TELEMETRY_ARRAY_FIELDS.items()
        for index in range(width)
    ]
    fieldnames = base_fields + command_fields + state_fields

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in states:
            state = record.state
            command = record.command
            row: dict[str, Any] = {
                "received_monotonic_s": record.received_at,
                "robot_time_s": _state_time_seconds(state),
                "frame": int(getattr(state, "frame", -1)),
                "robot_mode": int(getattr(state, "robot_mode", -1)),
                "control_command_success_rate": float(
                    getattr(state, "control_command_success_rate", float("nan"))
                ),
                "command_sequence": "" if command is None else command.sequence,
                "command_sent_monotonic_s": "" if command is None else command.sent_at,
                "command_age_s": "" if command is None else record.received_at - command.sent_at,
                "phase": "" if command is None else command.label,
                "phase_elapsed_s": "" if command is None else command.phase_elapsed_s,
                "current_errors": "|".join(
                    _active_boolean_fields(getattr(state, "current_errors", None))
                ),
                "last_motion_errors": "|".join(
                    _active_boolean_fields(getattr(state, "last_motion_errors", None))
                ),
            }
            command_target = (float("nan"),) * 7 if command is None else command.target
            for joint, value in enumerate(command_target, start=1):
                row[f"q_cmd_{joint}"] = value
            for name, width in TELEMETRY_ARRAY_FIELDS.items():
                for index, value in enumerate(
                    _telemetry_array(state, name, width), start=1
                ):
                    row[f"{name}_{index}"] = value
            writer.writerow(row)


def _count_direction_flips(values: list[float], threshold: float = 0.02) -> int:
    signs = [int(np.sign(value)) for value in values if abs(value) > threshold]
    return sum(previous != current for previous, current in zip(signs, signs[1:]))


def _direction_telemetry(states: list[TelemetryState]) -> tuple[DirectionTelemetry, ...]:
    summaries: list[DirectionTelemetry] = []
    for label, direction in (("workspace-positive", 1.0), ("workspace-negative", -1.0)):
        selected = [
            record
            for record in states
            if record.command is not None and record.command.label == label
        ]
        if not selected:
            continue
        expected: list[float] = []
        desired: list[float] = []
        measured: list[float] = []
        accelerations: list[float] = []
        for record in selected:
            assert record.command is not None
            expected.append(
                direction
                * cosine_bump_velocity(
                    record.command.phase_elapsed_s,
                    WORKSPACE_SWEEP_DURATION_S,
                    WORKSPACE_SWEEP_AMPLITUDE_RAD,
                )
            )
            desired.append(
                float(_telemetry_array(record.state, "dq_d", 7)[WORKSPACE_SWEEP_INDEX])
            )
            measured.append(
                float(_telemetry_array(record.state, "dq", 7)[WORKSPACE_SWEEP_INDEX])
            )
            accelerations.append(
                float(
                    _telemetry_array(record.state, "ddq_d", 7)[
                        WORKSPACE_SWEEP_INDEX
                    ]
                )
            )
        desired_wrong = sum(
            abs(reference) > MAX_HOLD_SPEED_RAD_S
            and abs(value) > MAX_HOLD_SPEED_RAD_S
            and reference * value < 0.0
            for reference, value in zip(expected, desired)
        )
        measured_wrong = sum(
            abs(reference) > MAX_HOLD_SPEED_RAD_S
            and abs(value) > MAX_HOLD_SPEED_RAD_S
            and reference * value < 0.0
            for reference, value in zip(expected, measured)
        )
        finite_accelerations = [value for value in accelerations if np.isfinite(value)]
        summaries.append(
            DirectionTelemetry(
                label=label,
                samples=len(selected),
                expected_flips=_count_direction_flips(expected),
                desired_flips=_count_direction_flips(desired),
                measured_flips=_count_direction_flips(measured),
                desired_wrong_direction=desired_wrong,
                measured_wrong_direction=measured_wrong,
                max_abs_ddq_d=max(
                    (abs(value) for value in finite_accelerations), default=float("nan")
                ),
            )
        )

    large_sweep = [
        record
        for record in states
        if record.command is not None
        and record.command.label == "all-joint-large-sweep"
    ]
    for joint in range(7):
        if not large_sweep:
            break
        expected = []
        desired = []
        measured = []
        accelerations = []
        for record in large_sweep:
            assert record.command is not None
            expected.append(
                float(
                    all_joint_large_sweep_velocity(
                        record.command.phase_elapsed_s
                    )[joint]
                )
            )
            desired.append(float(_telemetry_array(record.state, "dq_d", 7)[joint]))
            measured.append(float(_telemetry_array(record.state, "dq", 7)[joint]))
            accelerations.append(
                float(_telemetry_array(record.state, "ddq_d", 7)[joint])
            )
        desired_wrong = sum(
            abs(reference) > MAX_HOLD_SPEED_RAD_S
            and abs(value) > MAX_HOLD_SPEED_RAD_S
            and reference * value < 0.0
            for reference, value in zip(expected, desired)
        )
        measured_wrong = sum(
            abs(reference) > MAX_HOLD_SPEED_RAD_S
            and abs(value) > MAX_HOLD_SPEED_RAD_S
            and reference * value < 0.0
            for reference, value in zip(expected, measured)
        )
        finite_accelerations = [
            value for value in accelerations if np.isfinite(value)
        ]
        summaries.append(
            DirectionTelemetry(
                label=f"all-joint-large-sweep-J{joint + 1}",
                samples=len(large_sweep),
                expected_flips=_count_direction_flips(expected),
                desired_flips=_count_direction_flips(desired),
                measured_flips=_count_direction_flips(measured),
                desired_wrong_direction=desired_wrong,
                measured_wrong_direction=measured_wrong,
                max_abs_ddq_d=max(
                    (abs(value) for value in finite_accelerations),
                    default=float("nan"),
                ),
            )
        )
    return tuple(summaries)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively test raw Deoxys JOINT_POSITION control without IK or VR.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Deoxys interface config path/name. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help=(
            "Comma-separated tests for non-interactive selection: state, hold, "
            "workspace-sweep, all-joint-large-sweep, watchdog, or all. "
            "Omit to choose interactively."
        ),
    )
    return parser.parse_args(argv)


def resolve_config(config: str) -> Path:
    path = Path(config)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return DEOXYS_ROOT / "config" / config


def import_deoxys() -> tuple[type[Any], Callable[[str], Any], Any]:
    if not DEOXYS_ROOT.exists():
        raise FileNotFoundError(f"Missing Deoxys dependency: {DEOXYS_ROOT}")
    if str(DEOXYS_ROOT) not in sys.path:
        sys.path.insert(0, str(DEOXYS_ROOT))

    from deoxys.franka_interface import FrankaInterface  # noqa: PLC0415
    from deoxys.proto.franka_interface import franka_controller_pb2  # noqa: PLC0415
    from deoxys.utils.config_utils import get_default_controller_config  # noqa: PLC0415

    return FrankaInterface, get_default_controller_config, franka_controller_pb2


def parse_test_selection(value: str) -> list[str]:
    aliases = {
        "1": STATE_TEST,
        "2": HOLD_TEST,
        "3": WORKSPACE_SWEEP_TEST,
        "4": ALL_JOINT_LARGE_SWEEP_TEST,
        "5": WATCHDOG_TEST,
        "state": STATE_TEST,
        "hold": HOLD_TEST,
        "workspace-sweep": WORKSPACE_SWEEP_TEST,
        "workspace_sweep": WORKSPACE_SWEEP_TEST,
        "sweep": WORKSPACE_SWEEP_TEST,
        "joint-bump": WORKSPACE_SWEEP_TEST,
        "bump": WORKSPACE_SWEEP_TEST,
        "all-joint-large-sweep": ALL_JOINT_LARGE_SWEEP_TEST,
        "all_joint_large_sweep": ALL_JOINT_LARGE_SWEEP_TEST,
        "all-joint-sweep": ALL_JOINT_LARGE_SWEEP_TEST,
        "all_joint_sweep": ALL_JOINT_LARGE_SWEEP_TEST,
        "all-joint-figure-eight": ALL_JOINT_LARGE_SWEEP_TEST,
        "figure-eight": ALL_JOINT_LARGE_SWEEP_TEST,
        "figure8": ALL_JOINT_LARGE_SWEEP_TEST,
        "dance": ALL_JOINT_LARGE_SWEEP_TEST,
        "watchdog": WATCHDOG_TEST,
    }
    normalized = value.strip().lower()
    if normalized in ("", "a", "all"):
        return list(NON_DISRUPTIVE_TESTS)

    selected: set[str] = set()
    for raw_item in normalized.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item not in aliases:
            valid = (
                "1/state, 2/hold, 3/workspace-sweep, "
                "4/all-joint-large-sweep, 5/watchdog, all"
            )
            raise ValueError(f"Unknown test {raw_item!r}. Valid selections: {valid}")
        selected.add(aliases[item])
    if not selected:
        raise ValueError("At least one test must be selected")
    return [name for name in TEST_ORDER if name in selected]


def choose_tests(input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print) -> list[str]:
    print_fn("Available raw Deoxys tests:")
    print_fn("  1) state       read-only state rate, shape, finiteness, and error checks")
    print_fn("  2) hold        stream the current desired joints without intentional motion")
    print_fn("  3) workspace-sweep  move J1 through +/-0.15 rad and return to anchor")
    print_fn(
        "  4) all-joint-large-sweep  tracker-limit coordinated +/- motion of all seven joints"
    )
    print_fn("  5) watchdog    intentionally stop messages; backend may exit/restart")
    print_fn("  a) all         run all motion diagnostics above (excludes watchdog)")
    return parse_test_selection(input_fn("Select tests [a]: "))


def confirm_motion(
    config_path: Path,
    tests: list[str],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> bool:
    write_tests = [name for name in tests if name in WRITE_TESTS]
    if not write_tests:
        return True
    print_fn("")
    print_fn("[WARNING] The selected tests send commands to real Franka hardware.")
    print_fn(f"[WARNING] Config: {config_path}")
    print_fn(f"[WARNING] Commanding tests: {', '.join(write_tests)}")
    if WORKSPACE_SWEEP_TEST in write_tests:
        print_fn("[WARNING] J1 will move +/-0.15 rad; the full arm and end effector will move several cm.")
    if ALL_JOINT_LARGE_SWEEP_TEST in write_tests:
        print_fn(
            f"[WARNING] All seven joints will move together through a "
            f"{ALL_JOINT_SWEEP_DURATION_S:.1f} s tracker-limit bidirectional "
            "sweep (up to +/-0.65 rad); expect fast, substantial arm and "
            "end-effector motion."
        )
        print_fn(
            "[WARNING] Reference peak joint speeds: "
            f"{np.round(ALL_JOINT_SWEEP_COMMAND_PEAK_VELOCITY_RAD_S, 3).tolist()} "
            "rad/s."
        )
    print_fn("[WARNING] Verify the robot identity, clear the workspace, and keep the stop reachable.")
    return input_fn("Type MOVE to continue: ").strip() == "MOVE"


def confirm_watchdog(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> bool:
    print_fn("")
    print_fn("[WARNING] The watchdog test intentionally stops policy messages for 1.5 seconds.")
    print_fn("[WARNING] The current backend may terminate and auto_arm.sh may restart it.")
    return input_fn("Type WATCHDOG to run this disruptive test: ").strip() == "WATCHDOG"


def _active_boolean_fields(message: Any) -> tuple[str, ...]:
    if message is None:
        return ()
    if hasattr(message, "ListFields"):
        return tuple(
            field.name
            for field, value in message.ListFields()
            if isinstance(value, bool) and value
        )
    values = vars(message) if hasattr(message, "__dict__") else {}
    return tuple(sorted(name for name, value in values.items() if value is True))


def _state_vector(state: Any, name: str) -> np.ndarray:
    value = np.asarray(getattr(state, name, None), dtype=float)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"Deoxys state {name} must be a finite shape-(7,) vector")
    return value.copy()


def snapshot_state(interface: Any, *, monotonic_fn: Callable[[], float] = time.monotonic) -> StateSnapshot:
    if not getattr(interface, "state_buffer_size", 0):
        raise RuntimeError("Deoxys state is unavailable")
    state = interface._state_buffer[-1]
    success_rate = float(getattr(state, "control_command_success_rate", float("nan")))
    if not np.isfinite(success_rate):
        raise RuntimeError("control_command_success_rate is non-finite")
    return StateSnapshot(
        received_at=monotonic_fn(),
        q=_state_vector(state, "q"),
        q_d=_state_vector(state, "q_d"),
        dq=_state_vector(state, "dq"),
        dq_d=_state_vector(state, "dq_d"),
        ddq_d=_state_vector(state, "ddq_d"),
        frame=int(getattr(state, "frame", -1)),
        robot_mode=int(getattr(state, "robot_mode", -1)),
        success_rate=success_rate,
        current_errors=_active_boolean_fields(getattr(state, "current_errors", None)),
        last_motion_errors=_active_boolean_fields(getattr(state, "last_motion_errors", None)),
    )


def wait_for_state(
    interface: Any,
    *,
    timeout: float = STATE_TIMEOUT_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> StateSnapshot:
    deadline = monotonic_fn() + timeout
    last_error: Exception | None = None
    while monotonic_fn() < deadline:
        if getattr(interface, "state_buffer_size", 0):
            try:
                return snapshot_state(interface, monotonic_fn=monotonic_fn)
            except RuntimeError as exc:
                last_error = exc
        sleep_fn(0.02)
    if last_error is not None:
        raise RuntimeError(f"Timed out waiting for valid Deoxys state: {last_error}")
    raise RuntimeError(f"Timed out waiting {timeout:.1f}s for Deoxys state")


def configure_joint_position(controller_config_loader: Callable[[str], Any]) -> Any:
    config = controller_config_loader("JOINT_POSITION")
    if isinstance(config, dict):
        config["is_delta"] = False
        interpolation = config["traj_interpolator_cfg"]
    else:
        config.is_delta = False
        interpolation = config.traj_interpolator_cfg
    interpolation["traj_interpolator_type"] = "LINEAR_JOINT_POSITION"
    interpolation["time_fraction"] = 1.0
    return config


def send_joint_position(interface: Any, controller_cfg: Any, target: np.ndarray) -> None:
    target = np.asarray(target, dtype=float)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("target must be a finite shape-(7,) vector")
    interface.control(
        controller_type="JOINT_POSITION",
        action=target.tolist() + [0.0],
        controller_cfg=controller_cfg,
        termination=False,
        control_gripper=False,
    )


def send_no_control(
    interface: Any,
    protobuf_module: Any,
    *,
    repeats: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    for index in range(repeats):
        message = protobuf_module.FrankaControlMessage()
        message.controller_type = message.ControllerType.NO_CONTROL
        message.control_msg.Pack(protobuf_module.FrankaDummyControllerMessage())
        message.timeout = 0.2
        message.termination = False
        interface._publisher.send(message.SerializeToString())
        if index + 1 < repeats:
            sleep_fn(CONTROL_PERIOD_S)


def _wait_for_new_frame(
    interface: Any,
    previous_frame: int,
    *,
    timeout: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> StateSnapshot:
    deadline = monotonic_fn() + timeout
    latest = snapshot_state(interface, monotonic_fn=monotonic_fn)
    while monotonic_fn() < deadline:
        latest = snapshot_state(interface, monotonic_fn=monotonic_fn)
        if latest.frame != previous_frame:
            return latest
        sleep_fn(0.01)
    return latest


def enter_joint_position_mode(
    interface: Any,
    controller_cfg: Any,
    *,
    telemetry: TelemetryCollector | None = None,
    label: str = "joint-position-arm",
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> np.ndarray:
    before = snapshot_state(interface, monotonic_fn=monotonic_fn)
    send_joint_position(interface, controller_cfg, before.q_d)
    sent_at = monotonic_fn()
    if telemetry is not None:
        telemetry.record_command(
            label=label,
            phase_elapsed_s=0.0,
            target=before.q_d,
            sent_at=sent_at,
        )
    after = _wait_for_new_frame(
        interface,
        before.frame,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    return after.q_d.copy()


def run_stream(
    interface: Any,
    controller_cfg: Any,
    command_fn: Callable[[float], np.ndarray],
    *,
    duration: float,
    label: str,
    report_joint_index: int = WORKSPACE_SWEEP_INDEX,
    report_all_joints: bool = False,
    report_anchor: np.ndarray | None = None,
    telemetry: TelemetryCollector | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> StreamResult:
    started_at = monotonic_fn()
    commands: list[np.ndarray] = []
    samples: list[StateSnapshot] = []
    sent_at: list[float] = []
    next_report = started_at

    while True:
        elapsed = monotonic_fn() - started_at
        if elapsed >= duration:
            break
        target = np.asarray(command_fn(elapsed), dtype=float)
        send_joint_position(interface, controller_cfg, target)
        sent_time = monotonic_fn()
        if telemetry is not None:
            telemetry.record_command(
                label=label,
                phase_elapsed_s=elapsed,
                target=target,
                sent_at=sent_time,
            )
        sample = snapshot_state(interface, monotonic_fn=monotonic_fn)
        commands.append(target.copy())
        samples.append(sample)
        sent_at.append(sent_time)

        if sent_time >= next_report and report_all_joints:
            reference = (
                np.zeros(7, dtype=float)
                if report_anchor is None
                else np.asarray(report_anchor, dtype=float)
            )
            print_fn(
                f"[{label}] t={elapsed:5.2f}s frame={sample.frame} "
                f"q_cmd_delta={np.round(target - reference, 4).tolist()} "
                f"dq_d={np.round(sample.dq_d, 4).tolist()} "
                f"dq={np.round(sample.dq, 4).tolist()} "
                f"success={sample.success_rate:.4f}"
            )
            next_report = sent_time + 0.5
        elif sent_time >= next_report:
            joint_number = report_joint_index + 1
            print_fn(
                f"[{label}] t={elapsed:5.2f}s frame={sample.frame} "
                f"q{joint_number}_cmd={target[report_joint_index]:+.5f} "
                f"q{joint_number}_d={sample.q_d[report_joint_index]:+.5f} "
                f"q{joint_number}={sample.q[report_joint_index]:+.5f} "
                f"dq{joint_number}_d={sample.dq_d[report_joint_index]:+.5f} "
                f"dq{joint_number}={sample.dq[report_joint_index]:+.5f} "
                f"success={sample.success_rate:.4f}"
            )
            next_report = sent_time + 0.5

        next_deadline = started_at + len(commands) * CONTROL_PERIOD_S
        remaining = next_deadline - monotonic_fn()
        if remaining > 0.0:
            sleep_fn(remaining)

    if not samples:
        raise RuntimeError(f"{label} produced no control samples")
    return StreamResult(tuple(commands), tuple(samples), tuple(sent_at))


def _stream_rate(result: StreamResult) -> tuple[float, float]:
    if len(result.sent_at) < 2:
        return 0.0, float("inf")
    intervals = np.diff(np.asarray(result.sent_at))
    elapsed = result.sent_at[-1] - result.sent_at[0]
    rate = (len(result.sent_at) - 1) / elapsed if elapsed > 0.0 else 0.0
    return float(rate), float(np.max(intervals))


def _stream_errors(result: StreamResult) -> tuple[str, ...]:
    return tuple(sorted({error for sample in result.samples for error in sample.current_errors}))


def _minimum_success_rate(result: StreamResult) -> float:
    samples = result.samples[1:] if len(result.samples) > 1 else result.samples
    return min(sample.success_rate for sample in samples)


def run_state_test(
    interface: Any,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> TestResult:
    before = snapshot_state(interface, monotonic_fn=monotonic_fn)
    before_count = int(interface.state_buffer_size)
    started_at = monotonic_fn()
    sleep_fn(STATE_SAMPLE_DURATION_S)
    elapsed = monotonic_fn() - started_at
    after = snapshot_state(interface, monotonic_fn=monotonic_fn)
    count_delta = int(interface.state_buffer_size) - before_count
    state_rate = count_delta / elapsed if elapsed > 0.0 else 0.0
    passed = (
        count_delta > 0
        and after.frame != before.frame
        and state_rate >= MIN_STATE_RATE_HZ
        and not after.current_errors
    )
    details = (
        f"state_rate={state_rate:.1f}Hz frame={before.frame}->{after.frame} "
        f"success={after.success_rate:.4f} current_errors={list(after.current_errors)} "
        f"last_motion_errors={list(after.last_motion_errors)}"
    )
    return TestResult(STATE_TEST, passed, details)


def run_hold_test(
    interface: Any,
    controller_cfg: Any,
    *,
    telemetry: TelemetryCollector | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> TestResult:
    initial = snapshot_state(interface, monotonic_fn=monotonic_fn)
    anchor = enter_joint_position_mode(
        interface,
        controller_cfg,
        telemetry=telemetry,
        label="hold-arm",
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    result = run_stream(
        interface,
        controller_cfg,
        lambda _elapsed: anchor,
        duration=HOLD_DURATION_S,
        label="hold",
        telemetry=telemetry,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        print_fn=print_fn,
    )
    rate, max_gap = _stream_rate(result)
    max_deviation = max(float(np.max(np.abs(sample.q - initial.q))) for sample in result.samples)
    max_velocity = max(float(np.max(np.abs(sample.dq))) for sample in result.samples)
    max_desired_velocity = max(float(np.max(np.abs(sample.dq_d))) for sample in result.samples)
    errors = _stream_errors(result)
    min_success = _minimum_success_rate(result)
    passed = (
        MIN_CONTROL_RATE_HZ <= rate <= MAX_CONTROL_RATE_HZ
        and max_gap <= MAX_CONTROL_GAP_S
        and max_deviation <= MAX_HOLD_DEVIATION_RAD
        and max_velocity <= MAX_HOLD_SPEED_RAD_S
        and max_desired_velocity <= MAX_HOLD_SPEED_RAD_S
        and min_success >= MIN_CONTROL_SUCCESS_RATE
        and not errors
    )
    details = (
        f"rate={rate:.1f}Hz max_gap={max_gap:.3f}s "
        f"max_q_deviation={max_deviation:.5f}rad max_abs_dq={max_velocity:.5f}rad/s "
        f"max_abs_dq_d={max_desired_velocity:.5f}rad/s "
        f"min_success={min_success:.4f} current_errors={list(errors)}"
    )
    return TestResult(HOLD_TEST, passed, details)


def cosine_bump(elapsed: float, duration: float, amplitude: float) -> float:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    phase = min(max(float(elapsed) / duration, 0.0), 1.0)
    return 0.5 * float(amplitude) * (1.0 - math.cos(2.0 * math.pi * phase))


def cosine_bump_velocity(elapsed: float, duration: float, amplitude: float) -> float:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    phase = min(max(float(elapsed) / duration, 0.0), 1.0)
    return (
        math.pi
        * float(amplitude)
        / duration
        * math.sin(2.0 * math.pi * phase)
    )


def all_joint_large_sweep_offset(
    elapsed: float,
    duration: float = ALL_JOINT_SWEEP_DURATION_S,
) -> np.ndarray:
    """Return a smooth anchor -> pose A -> anchor -> pose B -> anchor sweep."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    clamped = min(max(float(elapsed), 0.0), duration)
    half_duration = 0.5 * duration
    if clamped < half_duration:
        progress = cosine_bump(clamped, half_duration, 1.0)
    else:
        progress = -cosine_bump(
            clamped - half_duration,
            half_duration,
            1.0,
        )
    return ALL_JOINT_SWEEP_SIGNS * ALL_JOINT_SWEEP_AMPLITUDES_RAD * progress


def all_joint_large_sweep_velocity(
    elapsed: float,
    duration: float = ALL_JOINT_SWEEP_DURATION_S,
) -> np.ndarray:
    """Return the analytic velocity of ``all_joint_large_sweep_offset``."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    clamped = min(max(float(elapsed), 0.0), duration)
    half_duration = 0.5 * duration
    if clamped < half_duration:
        progress_velocity = cosine_bump_velocity(
            clamped,
            half_duration,
            1.0,
        )
    else:
        progress_velocity = -cosine_bump_velocity(
            clamped - half_duration,
            half_duration,
            1.0,
        )
    return (
        ALL_JOINT_SWEEP_SIGNS
        * ALL_JOINT_SWEEP_AMPLITUDES_RAD
        * progress_velocity
    )


def _joint_velocity_diagnostics(
    result: StreamResult,
    *,
    direction: float,
    joint_index: int,
) -> tuple[float, float, float, int]:
    started_at = result.sent_at[0]
    expected = np.asarray(
        [
            direction
            * cosine_bump_velocity(
                sent_at - started_at,
                WORKSPACE_SWEEP_DURATION_S,
                WORKSPACE_SWEEP_AMPLITUDE_RAD,
            )
            for sent_at in result.sent_at
        ]
    )
    measured = np.asarray([sample.dq[joint_index] for sample in result.samples])
    desired = np.asarray([sample.dq_d[joint_index] for sample in result.samples])
    measured_error = float(np.max(np.abs(measured - expected)))
    desired_error = float(np.max(np.abs(desired - expected)))
    intervals = np.diff(np.asarray(result.sent_at))
    accelerations = np.diff(measured) / intervals
    max_acceleration = float(np.max(np.abs(accelerations))) if accelerations.size else 0.0
    wrong_direction = int(
        np.count_nonzero(
            (np.abs(expected) > MAX_HOLD_SPEED_RAD_S)
            & (np.abs(measured) > MAX_HOLD_SPEED_RAD_S)
            & (expected * measured < 0.0)
        )
    )
    return measured_error, desired_error, max_acceleration, wrong_direction


def require_bidirectional_sweep_margin(anchor: np.ndarray, joint_index: int, amplitude: float) -> None:
    positive_margin = PANDA_UPPER[joint_index] - anchor[joint_index]
    negative_margin = anchor[joint_index] - PANDA_LOWER[joint_index]
    required_margin = float(amplitude) + 0.05
    if positive_margin < required_margin or negative_margin < required_margin:
        raise RuntimeError(
            f"J{joint_index + 1} needs at least {required_margin:.3f} rad margin "
            f"in both directions for the workspace sweep; margins are "
            f"-{negative_margin:.3f}/+{positive_margin:.3f} rad"
        )


def run_workspace_sweep_test(
    interface: Any,
    controller_cfg: Any,
    *,
    telemetry: TelemetryCollector | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> TestResult:
    anchor = enter_joint_position_mode(
        interface,
        controller_cfg,
        telemetry=telemetry,
        label="workspace-arm",
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    require_bidirectional_sweep_margin(
        anchor,
        WORKSPACE_SWEEP_INDEX,
        WORKSPACE_SWEEP_AMPLITUDE_RAD,
    )

    sweeps: dict[str, StreamResult] = {}
    return_holds: dict[str, StreamResult] = {}
    for direction_name, direction in (("positive", 1.0), ("negative", -1.0)):
        def command(elapsed: float, *, direction: float = direction) -> np.ndarray:
            target = anchor.copy()
            target[WORKSPACE_SWEEP_INDEX] += direction * cosine_bump(
                elapsed,
                WORKSPACE_SWEEP_DURATION_S,
                WORKSPACE_SWEEP_AMPLITUDE_RAD,
            )
            return target

        sweeps[direction_name] = run_stream(
            interface,
            controller_cfg,
            command,
            duration=WORKSPACE_SWEEP_DURATION_S,
            label=f"workspace-{direction_name}",
            report_joint_index=WORKSPACE_SWEEP_INDEX,
            telemetry=telemetry,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
            print_fn=print_fn,
        )
        return_holds[direction_name] = run_stream(
            interface,
            controller_cfg,
            lambda _elapsed: anchor,
            duration=RETURN_HOLD_DURATION_S,
            label=f"{direction_name}-return-hold",
            report_joint_index=WORKSPACE_SWEEP_INDEX,
            telemetry=telemetry,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
            print_fn=print_fn,
        )

    positive = sweeps["positive"]
    negative = sweeps["negative"]
    positive_return = return_holds["positive"]
    negative_return = return_holds["negative"]
    positive_rate, positive_max_gap = _stream_rate(positive)
    negative_rate, negative_max_gap = _stream_rate(negative)
    positive_measured_peak = max(
        sample.q[WORKSPACE_SWEEP_INDEX] - anchor[WORKSPACE_SWEEP_INDEX]
        for sample in positive.samples
    )
    positive_desired_peak = max(
        sample.q_d[WORKSPACE_SWEEP_INDEX] - anchor[WORKSPACE_SWEEP_INDEX]
        for sample in positive.samples
    )
    negative_measured_peak = max(
        anchor[WORKSPACE_SWEEP_INDEX] - sample.q[WORKSPACE_SWEEP_INDEX]
        for sample in negative.samples
    )
    negative_desired_peak = max(
        anchor[WORKSPACE_SWEEP_INDEX] - sample.q_d[WORKSPACE_SWEEP_INDEX]
        for sample in negative.samples
    )
    final_error = abs(
        negative_return.samples[-1].q[WORKSPACE_SWEEP_INDEX]
        - anchor[WORKSPACE_SWEEP_INDEX]
    )
    sweep_samples = positive.samples + negative.samples
    expected_peak_speed = (
        math.pi * WORKSPACE_SWEEP_AMPLITUDE_RAD / WORKSPACE_SWEEP_DURATION_S
    )
    max_allowed_sweep_speed = max(
        MAX_HOLD_SPEED_RAD_S,
        SWEEP_SPEED_LIMIT_MULTIPLIER * expected_peak_speed,
    )
    max_measured_sweep_speed = max(
        abs(sample.dq[WORKSPACE_SWEEP_INDEX]) for sample in sweep_samples
    )
    max_desired_sweep_speed = max(
        abs(sample.dq_d[WORKSPACE_SWEEP_INDEX]) for sample in sweep_samples
    )
    non_sweep_indices = np.arange(7) != WORKSPACE_SWEEP_INDEX
    max_non_sweep_speed = max(
        float(np.max(np.abs(sample.dq[non_sweep_indices])))
        for sample in sweep_samples
    )
    max_non_sweep_desired_speed = max(
        float(np.max(np.abs(sample.dq_d[non_sweep_indices])))
        for sample in sweep_samples
    )
    positive_velocity = _joint_velocity_diagnostics(
        positive,
        direction=1.0,
        joint_index=WORKSPACE_SWEEP_INDEX,
    )
    negative_velocity = _joint_velocity_diagnostics(
        negative,
        direction=-1.0,
        joint_index=WORKSPACE_SWEEP_INDEX,
    )
    max_measured_velocity_error = max(positive_velocity[0], negative_velocity[0])
    max_desired_velocity_error = max(positive_velocity[1], negative_velocity[1])
    max_measured_acceleration = max(positive_velocity[2], negative_velocity[2])
    wrong_direction_samples = positive_velocity[3] + negative_velocity[3]
    return_samples = positive_return.samples + negative_return.samples
    max_return_speed = max(
        max(float(np.max(np.abs(sample.dq))), float(np.max(np.abs(sample.dq_d))))
        for sample in return_samples
    )
    errors = tuple(
        sorted(
            set(_stream_errors(positive))
            | set(_stream_errors(negative))
            | set(_stream_errors(positive_return))
            | set(_stream_errors(negative_return))
        )
    )
    min_success = min(
        _minimum_success_rate(positive),
        _minimum_success_rate(negative),
        _minimum_success_rate(positive_return),
        _minimum_success_rate(negative_return),
    )
    passed = (
        MIN_CONTROL_RATE_HZ <= positive_rate <= MAX_CONTROL_RATE_HZ
        and MIN_CONTROL_RATE_HZ <= negative_rate <= MAX_CONTROL_RATE_HZ
        and max(positive_max_gap, negative_max_gap) <= MAX_CONTROL_GAP_S
        and MIN_SWEEP_RESPONSE_RAD <= positive_measured_peak <= MAX_SWEEP_RESPONSE_RAD
        and MIN_SWEEP_RESPONSE_RAD <= negative_measured_peak <= MAX_SWEEP_RESPONSE_RAD
        and positive_desired_peak >= MIN_SWEEP_RESPONSE_RAD
        and negative_desired_peak >= MIN_SWEEP_RESPONSE_RAD
        and final_error <= MAX_HOLD_DEVIATION_RAD
        and max_measured_sweep_speed <= max_allowed_sweep_speed
        and max_desired_sweep_speed <= max_allowed_sweep_speed
        and max_non_sweep_speed <= MAX_NON_SWEEP_SPEED_RAD_S
        and max_non_sweep_desired_speed <= MAX_NON_SWEEP_SPEED_RAD_S
        and max_measured_velocity_error <= MAX_SWEEP_VELOCITY_ERROR_RAD_S
        and max_desired_velocity_error <= MAX_SWEEP_VELOCITY_ERROR_RAD_S
        and max_measured_acceleration <= MAX_SWEEP_ACCELERATION_RAD_S2
        and wrong_direction_samples == 0
        and max_return_speed <= MAX_HOLD_SPEED_RAD_S
        and min_success >= MIN_CONTROL_SUCCESS_RATE
        and not errors
    )
    details = (
        f"J1 excursion=+/-{WORKSPACE_SWEEP_AMPLITUDE_RAD:.3f}rad "
        f"rate=+{positive_rate:.1f}/-{negative_rate:.1f}Hz "
        f"max_gap={max(positive_max_gap, negative_max_gap):.3f}s "
        f"q_d_peak=+{positive_desired_peak:.5f}/-{negative_desired_peak:.5f}rad "
        f"q_peak=+{positive_measured_peak:.5f}/-{negative_measured_peak:.5f}rad "
        f"max_dq1={max_measured_sweep_speed:.5f}rad/s "
        f"max_dq1_d={max_desired_sweep_speed:.5f}rad/s "
        f"speed_limit={max_allowed_sweep_speed:.5f}rad/s "
        f"max_other_dq={max_non_sweep_speed:.5f}rad/s "
        f"max_other_dq_d={max_non_sweep_desired_speed:.5f}rad/s "
        f"velocity_error={max_measured_velocity_error:.5f}/"
        f"{max_desired_velocity_error:.5f}rad/s "
        f"max_accel={max_measured_acceleration:.5f}rad/s^2 "
        f"wrong_direction={wrong_direction_samples} "
        f"max_return_speed={max_return_speed:.5f}rad/s "
        f"return_error={final_error:.5f}rad min_success={min_success:.4f} "
        f"current_errors={list(errors)}"
    )
    return TestResult(WORKSPACE_SWEEP_TEST, passed, details)


def require_all_joint_large_sweep_margin(anchor: np.ndarray) -> None:
    negative_margin = anchor - PANDA_LOWER
    positive_margin = PANDA_UPPER - anchor
    required_margin = (
        ALL_JOINT_SWEEP_AMPLITUDES_RAD + ALL_JOINT_SWEEP_LIMIT_MARGIN_RAD
    )
    unsafe = np.flatnonzero(
        (negative_margin < required_margin) | (positive_margin < required_margin)
    )
    if unsafe.size:
        joints = ", ".join(f"J{index + 1}" for index in unsafe)
        raise RuntimeError(
            "all-joint large sweep lacks conservative joint-limit margin for "
            f"{joints}; available negative={np.round(negative_margin, 3).tolist()} "
            f"positive={np.round(positive_margin, 3).tolist()} "
            f"required={np.round(required_margin, 3).tolist()}"
        )


def _joint_direction_flips(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _count_direction_flips(values[:, joint].tolist())
            for joint in range(values.shape[1])
        ],
        dtype=int,
    )


def _joint_wrong_direction(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    active = (np.abs(expected) > MAX_HOLD_SPEED_RAD_S) & (
        np.abs(actual) > MAX_HOLD_SPEED_RAD_S
    )
    return np.count_nonzero(active & (expected * actual < 0.0), axis=0)


def run_all_joint_large_sweep_test(
    interface: Any,
    controller_cfg: Any,
    *,
    telemetry: TelemetryCollector | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> TestResult:
    anchor = enter_joint_position_mode(
        interface,
        controller_cfg,
        telemetry=telemetry,
        label="all-joint-large-sweep-arm",
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    require_all_joint_large_sweep_margin(anchor)

    result = run_stream(
        interface,
        controller_cfg,
        lambda elapsed: anchor + all_joint_large_sweep_offset(elapsed),
        duration=ALL_JOINT_SWEEP_DURATION_S,
        label="all-joint-large-sweep",
        report_all_joints=True,
        report_anchor=anchor,
        telemetry=telemetry,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        print_fn=print_fn,
    )
    return_hold = run_stream(
        interface,
        controller_cfg,
        lambda _elapsed: anchor,
        duration=ALL_JOINT_SWEEP_RETURN_HOLD_DURATION_S,
        label="all-joint-sweep-return-hold",
        report_all_joints=True,
        report_anchor=anchor,
        telemetry=telemetry,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        print_fn=print_fn,
    )

    rate, max_gap = _stream_rate(result)
    command_offsets = np.stack(result.commands) - anchor
    desired_offsets = np.stack([sample.q_d for sample in result.samples]) - anchor
    measured_offsets = np.stack([sample.q for sample in result.samples]) - anchor
    desired_velocities = np.stack([sample.dq_d for sample in result.samples])
    measured_velocities = np.stack([sample.dq for sample in result.samples])
    desired_accelerations = np.stack([sample.ddq_d for sample in result.samples])
    elapsed = np.asarray(result.sent_at) - result.sent_at[0]
    expected_velocities = np.stack(
        [all_joint_large_sweep_velocity(value) for value in elapsed]
    )

    command_positive = np.max(command_offsets, axis=0)
    command_negative = -np.min(command_offsets, axis=0)
    desired_positive = np.max(desired_offsets, axis=0)
    desired_negative = -np.min(desired_offsets, axis=0)
    measured_positive = np.max(measured_offsets, axis=0)
    measured_negative = -np.min(measured_offsets, axis=0)
    expected_flips = _joint_direction_flips(expected_velocities)
    desired_flips = _joint_direction_flips(desired_velocities)
    measured_flips = _joint_direction_flips(measured_velocities)
    desired_wrong = _joint_wrong_direction(expected_velocities, desired_velocities)
    measured_wrong = _joint_wrong_direction(expected_velocities, measured_velocities)
    max_desired_speed = np.max(np.abs(desired_velocities), axis=0)
    max_measured_speed = np.max(np.abs(measured_velocities), axis=0)
    max_desired_velocity_error = float(
        np.max(np.abs(desired_velocities - expected_velocities))
    )
    max_measured_velocity_error = float(
        np.max(np.abs(measured_velocities - expected_velocities))
    )
    max_tracking_error = float(
        np.max(np.abs(np.stack([sample.q for sample in result.samples]) -
                      np.stack([sample.q_d for sample in result.samples])))
    )
    max_desired_acceleration = float(np.max(np.abs(desired_accelerations)))
    settle_start = min(
        len(return_hold.samples) - 1,
        int(0.5 * CONTROL_RATE_HZ),
    )
    settled_return_samples = return_hold.samples[settle_start:]
    return_speed = max(
        max(float(np.max(np.abs(sample.dq))), float(np.max(np.abs(sample.dq_d))))
        for sample in settled_return_samples
    )
    return_error = float(
        np.max(np.abs(return_hold.samples[-1].q - anchor))
    )
    desired_positions = np.stack([sample.q_d for sample in result.samples])
    measured_positions = np.stack([sample.q for sample in result.samples])
    within_joint_limits = bool(
        np.all(desired_positions >= PANDA_LOWER)
        and np.all(desired_positions <= PANDA_UPPER)
        and np.all(measured_positions >= PANDA_LOWER)
        and np.all(measured_positions <= PANDA_UPPER)
    )
    errors = tuple(
        sorted(set(_stream_errors(result)) | set(_stream_errors(return_hold)))
    )
    min_success = min(
        _minimum_success_rate(result), _minimum_success_rate(return_hold)
    )

    desired_response = (
        (
            desired_positive
            >= MIN_ALL_JOINT_SWEEP_RESPONSE_FRACTION * command_positive
        )
        & (
            desired_negative
            >= MIN_ALL_JOINT_SWEEP_RESPONSE_FRACTION * command_negative
        )
    )
    measured_response = (
        (
            measured_positive
            >= MIN_ALL_JOINT_SWEEP_RESPONSE_FRACTION * command_positive
        )
        & (
            measured_negative
            >= MIN_ALL_JOINT_SWEEP_RESPONSE_FRACTION * command_negative
        )
    )
    command_profile_within_tracker_limits = bool(
        np.all(
            ALL_JOINT_SWEEP_COMMAND_PEAK_VELOCITY_RAD_S
            <= DEOXYS_STREAMING_MAX_VELOCITY_RAD_S + 1e-12
        )
        and np.all(
            ALL_JOINT_SWEEP_COMMAND_PEAK_ACCELERATION_RAD_S2
            <= DEOXYS_STREAMING_MAX_ACCELERATION_RAD_S2 + 1e-12
        )
        and np.all(
            ALL_JOINT_SWEEP_COMMAND_PEAK_JERK_RAD_S3
            <= DEOXYS_STREAMING_MAX_JERK_RAD_S3 + 1e-12
        )
    )
    passed = (
        MIN_CONTROL_RATE_HZ <= rate <= MAX_CONTROL_RATE_HZ
        and max_gap <= MAX_CONTROL_GAP_S
        and command_profile_within_tracker_limits
        and bool(np.all(desired_response))
        and bool(np.all(measured_response))
        and bool(
            np.all(
                max_desired_speed
                <= DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
                + MAX_ALL_JOINT_SWEEP_DESIRED_SPEED_LIMIT_TOLERANCE_RAD_S
            )
        )
        and bool(
            np.all(
                max_measured_speed
                <= DEOXYS_STREAMING_MAX_VELOCITY_RAD_S
                + MAX_ALL_JOINT_SWEEP_MEASURED_SPEED_LIMIT_TOLERANCE_RAD_S
            )
        )
        and max_tracking_error <= MAX_ALL_JOINT_SWEEP_TRACKING_ERROR_RAD
        and max_desired_acceleration
        <= MAX_ALL_JOINT_SWEEP_DESIRED_ACCELERATION_RAD_S2
        and bool(
            np.all(
                np.abs(desired_flips - expected_flips)
                <= MAX_ALL_JOINT_SWEEP_FLIP_DIFFERENCE
            )
        )
        and bool(
            np.all(
                np.abs(measured_flips - expected_flips)
                <= MAX_ALL_JOINT_SWEEP_FLIP_DIFFERENCE
            )
        )
        and return_speed <= MAX_ALL_JOINT_SWEEP_RETURN_SPEED_RAD_S
        and return_error <= MAX_HOLD_DEVIATION_RAD
        and within_joint_limits
        and min_success >= MIN_CONTROL_SUCCESS_RATE
        and not errors
    )

    def values(vector: np.ndarray) -> list[float]:
        return np.round(vector, 4).tolist()

    details = (
        f"duration={ALL_JOINT_SWEEP_DURATION_S:.1f}s rate={rate:.1f}Hz "
        f"max_gap={max_gap:.3f}s "
        f"q_cmd_excursion=+{values(command_positive)}/-{values(command_negative)}rad "
        f"q_d_excursion=+{values(desired_positive)}/-{values(desired_negative)}rad "
        f"q_excursion=+{values(measured_positive)}/-{values(measured_negative)}rad "
        f"command_max_dq={values(ALL_JOINT_SWEEP_COMMAND_PEAK_VELOCITY_RAD_S)}rad/s "
        f"tracker_max_dq={values(DEOXYS_STREAMING_MAX_VELOCITY_RAD_S)}rad/s "
        f"command_max_ddq={float(np.max(ALL_JOINT_SWEEP_COMMAND_PEAK_ACCELERATION_RAD_S2)):.4f}rad/s^2 "
        f"command_max_jerk={float(np.max(ALL_JOINT_SWEEP_COMMAND_PEAK_JERK_RAD_S3)):.4f}rad/s^3 "
        f"max_dq_d={values(max_desired_speed)}rad/s "
        f"max_dq={values(max_measured_speed)}rad/s "
        f"max_ddq_d={max_desired_acceleration:.4f}rad/s^2 "
        f"velocity_lag_peak={max_desired_velocity_error:.4f}/"
        f"{max_measured_velocity_error:.4f}rad/s "
        f"tracking_error={max_tracking_error:.5f}rad "
        f"flips expected/dq_d/dq={expected_flips.tolist()}/"
        f"{desired_flips.tolist()}/{measured_flips.tolist()} "
        f"phase_lag_samples_vs_raw dq_d/dq={desired_wrong.tolist()}/"
        f"{measured_wrong.tolist()} "
        f"return_speed={return_speed:.4f}rad/s "
        f"return_error={return_error:.5f}rad "
        f"profile_limits_ok={command_profile_within_tracker_limits} "
        f"joint_limits_ok={within_joint_limits} min_success={min_success:.4f} "
        f"current_errors={list(errors)}"
    )
    return TestResult(ALL_JOINT_LARGE_SWEEP_TEST, passed, details)


def run_watchdog_test(
    interface: Any,
    controller_cfg: Any,
    *,
    telemetry: TelemetryCollector | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    print_fn: Callable[[str], None] = print,
) -> TestResult:
    anchor = enter_joint_position_mode(
        interface,
        controller_cfg,
        telemetry=telemetry,
        label="watchdog-arm",
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    run_stream(
        interface,
        controller_cfg,
        lambda _elapsed: anchor,
        duration=WATCHDOG_HOLD_DURATION_S,
        label="watchdog-hold",
        telemetry=telemetry,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
        print_fn=print_fn,
    )
    before = snapshot_state(interface, monotonic_fn=monotonic_fn)
    before_count = int(interface.state_buffer_size)
    print_fn(f"[watchdog] No policy messages for {WATCHDOG_SILENCE_S:.1f}s...")
    sleep_fn(WATCHDOG_SILENCE_S)
    after = snapshot_state(interface, monotonic_fn=monotonic_fn)
    count_delta = int(interface.state_buffer_size) - before_count
    deviation = float(np.max(np.abs(after.q - before.q)))
    passed = deviation <= MAX_HOLD_DEVIATION_RAD and not after.current_errors
    details = (
        f"silence={WATCHDOG_SILENCE_S:.1f}s state_messages_during_silence={count_delta} "
        f"frame={before.frame}->{after.frame} max_q_deviation={deviation:.5f}rad "
        f"current_errors={list(after.current_errors)}; confirm backend log contains "
        "'No valid messages received in 20 steps' or the expected restart"
    )
    return TestResult(WATCHDOG_TEST, passed, details)


def _gripper_width(interface: Any) -> float | None:
    if not getattr(interface, "gripper_state_buffer_size", 0):
        return None
    value = float(interface._gripper_state_buffer[-1].width)
    return value if np.isfinite(value) else None


def print_initial_state(snapshot: StateSnapshot, print_fn: Callable[[str], None] = print) -> None:
    print_fn(f"[STATE] frame={snapshot.frame} robot_mode={snapshot.robot_mode}")
    print_fn(f"[STATE] q={np.round(snapshot.q, 5).tolist()}")
    print_fn(f"[STATE] q_d={np.round(snapshot.q_d, 5).tolist()}")
    print_fn(f"[STATE] dq={np.round(snapshot.dq, 5).tolist()}")
    print_fn(
        f"[STATE] success={snapshot.success_rate:.4f} "
        f"current_errors={list(snapshot.current_errors)} "
        f"last_motion_errors={list(snapshot.last_motion_errors)}"
    )


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    args = parse_args(argv)
    config_path = resolve_config(args.config)
    if not config_path.exists():
        print(f"[ERROR] Missing config: {config_path}", file=sys.stderr)
        return 1

    try:
        tests = parse_test_selection(args.tests) if args.tests is not None else choose_tests(input_fn, print_fn)
    except (EOFError, ValueError) as exc:
        print(f"[ERROR] Could not select tests: {exc}", file=sys.stderr)
        return 2

    print_fn(f"[INFO] Config: {config_path}")
    print_fn(f"[INFO] Selected tests: {', '.join(tests)}")
    interface = None
    protobuf_module = None
    telemetry: TelemetryCollector | None = None
    commands_sent = False
    results: list[TestResult] = []

    try:
        FrankaInterface, controller_config_loader, protobuf_module = import_deoxys()
        interface = FrankaInterface(
            str(config_path),
            control_freq=CONTROL_RATE_HZ,
            state_freq=100.0,
            has_gripper=True,
            use_visualizer=False,
            automatic_gripper_reset=False,
        )
        initial = wait_for_state(interface)
        print_initial_state(initial, print_fn)
        telemetry = TelemetryCollector(interface)
        telemetry.start()
        print_fn("[INFO] Capturing every published state and command to telemetry files.")
        starting_gripper_width = _gripper_width(interface)
        if starting_gripper_width is not None:
            print_fn(f"[STATE] gripper_width={starting_gripper_width:.5f}m (arm tests will not command it)")

        if STATE_TEST in tests:
            print_fn("\n=== STATE TEST ===")
            result = run_state_test(interface)
            results.append(result)
            print_fn(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.details}")

        if any(name in WRITE_TESTS for name in tests):
            try:
                confirmed = confirm_motion(config_path, tests, input_fn, print_fn)
            except EOFError:
                confirmed = False
            if not confirmed:
                print_fn("[CANCELLED] No control commands were sent.")
                return 0
            controller_cfg = configure_joint_position(controller_config_loader)

            if HOLD_TEST in tests:
                print_fn("\n=== HOLD TEST ===")
                commands_sent = True
                result = run_hold_test(
                    interface,
                    controller_cfg,
                    telemetry=telemetry,
                    print_fn=print_fn,
                )
                results.append(result)
                print_fn(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.details}")

            if WORKSPACE_SWEEP_TEST in tests:
                print_fn("\n=== WORKSPACE SWEEP TEST ===")
                commands_sent = True
                result = run_workspace_sweep_test(
                    interface,
                    controller_cfg,
                    telemetry=telemetry,
                    print_fn=print_fn,
                )
                results.append(result)
                print_fn(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.details}")

            if ALL_JOINT_LARGE_SWEEP_TEST in tests:
                print_fn("\n=== ALL-JOINT LARGE SWEEP TEST ===")
                commands_sent = True
                result = run_all_joint_large_sweep_test(
                    interface,
                    controller_cfg,
                    telemetry=telemetry,
                    print_fn=print_fn,
                )
                results.append(result)
                print_fn(
                    f"[{'PASS' if result.passed else 'FAIL'}] "
                    f"{result.name}: {result.details}"
                )

            if WATCHDOG_TEST in tests:
                try:
                    watchdog_confirmed = confirm_watchdog(input_fn, print_fn)
                except EOFError:
                    watchdog_confirmed = False
                if watchdog_confirmed:
                    print_fn("\n=== WATCHDOG TEST ===")
                    commands_sent = True
                    result = run_watchdog_test(
                        interface,
                        controller_cfg,
                        telemetry=telemetry,
                        print_fn=print_fn,
                    )
                    results.append(result)
                    print_fn(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.details}")
                else:
                    print_fn("[SKIP] watchdog was not explicitly confirmed.")

        ending_gripper_width = _gripper_width(interface)
        if starting_gripper_width is not None and ending_gripper_width is not None:
            gripper_delta = ending_gripper_width - starting_gripper_width
            print_fn(f"[GRIPPER] width_delta={gripper_delta:+.5f}m; no arm call enabled gripper control")

        print_fn("\n=== SUMMARY ===")
        for result in results:
            print_fn(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.details}")
        return 0 if results and all(result.passed for result in results) else 1
    except KeyboardInterrupt:
        print_fn("\n[INFO] Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - hardware diagnostic must report the actual failure.
        print(f"[ERROR] Raw Deoxys control diagnostic failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if interface is not None:
            if commands_sent and protobuf_module is not None:
                try:
                    if telemetry is not None:
                        telemetry.record_marker("no-control", sent_at=time.monotonic())
                    send_no_control(interface, protobuf_module)
                    print_fn("[STOP] Sent repeated NO_CONTROL messages.")
                except Exception as exc:  # noqa: BLE001 - still close the Python interface.
                    print_fn(f"[WARN] Failed to send NO_CONTROL: {exc}")
            if telemetry is not None:
                try:
                    telemetry.stop()
                    csv_path, jsonl_path = telemetry.save()
                    print_fn(
                        f"[TELEMETRY] saved {len(telemetry.states)} states and "
                        f"{len(telemetry.commands)} commands"
                    )
                    print_fn(f"[TELEMETRY] CSV: {csv_path}")
                    print_fn(f"[TELEMETRY] full JSONL: {jsonl_path}")
                    for summary in telemetry.direction_summaries():
                        direction_metric = (
                            "phase_lag_samples_vs_raw"
                            if summary.label.startswith("all-joint-large-sweep-")
                            else "wrong_direction"
                        )
                        print_fn(
                            f"[TELEMETRY] {summary.label}: samples={summary.samples} "
                            f"flips expected/dq_d/dq={summary.expected_flips}/"
                            f"{summary.desired_flips}/{summary.measured_flips} "
                            f"{direction_metric} dq_d/dq="
                            f"{summary.desired_wrong_direction}/"
                            f"{summary.measured_wrong_direction} "
                            f"max_abs_ddq_d={summary.max_abs_ddq_d:.5f}rad/s^2"
                        )
                except Exception as exc:  # noqa: BLE001 - still close the interface.
                    print_fn(f"[WARN] Failed to save telemetry: {exc}")
            try:
                interface.close()
            except Exception as exc:  # noqa: BLE001 - diagnostic cleanup should remain best effort.
                print_fn(f"[WARN] Failed to close FrankaInterface: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
