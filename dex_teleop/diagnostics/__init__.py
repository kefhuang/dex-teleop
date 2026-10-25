"""Structured diagnostic tracing for live control paths."""

from dex_teleop.diagnostics.control_trace import (
    ControlTracer,
    trace_event,
    trace_scope,
)

__all__ = ["ControlTracer", "trace_event", "trace_scope"]
