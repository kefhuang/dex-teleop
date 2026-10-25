from __future__ import annotations

import json

import numpy as np
import pytest

from dex_teleop.core.types import Pose
from dex_teleop.diagnostics import ControlTracer, trace_event, trace_scope


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_control_tracer_writes_structured_scoped_cycle(tmp_path):
    path = tmp_path / "trace.jsonl"

    with ControlTracer(path) as tracer:
        tracer.event("session.start", config={"rate_hz": 20})
        with tracer.span("control_cycle", iteration=3, configured_period_s=0.05):
            trace_event("device.read.result", values=np.array([1.0, 2.0]))
            with trace_scope(arm_id="right"):
                trace_event("teleop.target", pose=Pose.identity())

    records = _records(path)
    cycle = next(record for record in records if record["record_type"] == "control_cycle")

    assert cycle["schema_version"] == 1
    assert cycle["iteration"] == 3
    assert cycle["status"] == "ok"
    assert [event["name"] for event in cycle["events"]] == [
        "device.read.result",
        "teleop.target",
    ]
    assert cycle["events"][0]["data"]["values"] == [1.0, 2.0]
    assert cycle["events"][1]["scope"] == {"arm_id": "right"}
    assert cycle["events"][1]["data"]["pose"]["quaternion_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert records[-1]["record_type"] == "trace_summary"
    assert records[-1]["written_records_before_summary"] == len(records) - 1
    assert records[-1]["dropped_records"] == 0


def test_control_tracer_expands_protobuf_style_backend_state(tmp_path):
    class Field:
        def __init__(self, name):
            self.name = name

    class Descriptor:
        fields = [Field("q"), Field("dq"), Field("robot_mode")]

    class BackendState:
        DESCRIPTOR = Descriptor()
        q = [0.1, 0.2]
        dq = [0.3, 0.4]
        robot_mode = 2

    path = tmp_path / "trace.jsonl"
    with ControlTracer(path) as tracer:
        with tracer.span("control_cycle", iteration=0):
            trace_event("franka.backend.state.read", raw_state=BackendState())

    cycle = next(
        record for record in _records(path) if record["record_type"] == "control_cycle"
    )
    raw = cycle["events"][0]["data"]["raw_state"]
    assert raw == {"q": [0.1, 0.2], "dq": [0.3, 0.4], "robot_mode": 2}


def test_control_tracer_preserves_span_exception_and_traceback(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = ControlTracer(path)

    with pytest.raises(ValueError, match="broken command"):
        with tracer.span("control_cycle", iteration=9):
            trace_event("arm.command.request")
            raise ValueError("broken command")
    tracer.close()

    cycle = next(
        record for record in _records(path) if record["record_type"] == "control_cycle"
    )
    assert cycle["status"] == "error"
    assert cycle["exception"]["type"] == "builtins.ValueError"
    assert cycle["exception"]["message"] == "broken command"
    assert any("raise ValueError" in line for line in cycle["exception"]["traceback"])
