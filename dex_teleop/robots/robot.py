from __future__ import annotations
from dataclasses import dataclass
from dex_teleop.diagnostics import trace_event, trace_scope
from dex_teleop.robots.arm import Arm
from dex_teleop.robots.gripper import EndEffector

@dataclass
class Robot:
    arm: Arm
    end_effector: EndEffector | None = None
    _connection: object | None = None
    _closed: bool = False

    def connect(self) -> None:
        trace_event("robot.lifecycle.connect.request")
        try:
            with trace_scope(component="arm"):
                self.arm.connect()
            if self.end_effector is not None:
                with trace_scope(component="end_effector"):
                    self.end_effector.connect()
        except Exception as exc:
            trace_event("robot.lifecycle.connect.failed", exception=exc)
            try:
                self.stop()
            except Exception as cleanup_exc:
                trace_event("robot.lifecycle.connect_cleanup.failed", exception=cleanup_exc)
            raise
        trace_event("robot.lifecycle.connect.completed")

    def stop(self) -> None:
        trace_event("robot.lifecycle.stop.request")
        errors = []
        for component_name, component in (("arm", self.arm), ("end_effector", self.end_effector)):
            if component is not None:
                with trace_scope(component=component_name):
                    try:
                        component.stop()
                    except Exception as exc:
                        trace_event("robot.lifecycle.component_stop.failed", exception=exc)
                        errors.append(exc)
        if errors:
            raise errors[0]
        trace_event("robot.lifecycle.stop.completed")

    def close(self) -> None:
        if self._closed:
            trace_event("robot.lifecycle.close.noop", reason="already_closed")
            return
        trace_event("robot.lifecycle.close.request")
        self._closed = True
        error = None
        try: self.stop()
        except Exception as exc: error = exc
        try:
            if self._connection is not None and hasattr(self._connection, "close"):
                with trace_scope(component="connection"):
                    self._connection.close()
        except Exception as exc:
            if error is None: error = exc
        if error is not None:
            trace_event("robot.lifecycle.close.failed", exception=error)
            raise error
        trace_event("robot.lifecycle.close.completed")
