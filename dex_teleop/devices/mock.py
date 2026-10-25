from __future__ import annotations

import time

import numpy as np

from dex_teleop.core.types import (
    ControllerAction,
    ControllerState,
    DeviceState,
    Pose,
    Twist,
)
from dex_teleop.diagnostics import trace_event


class MockControllerInput:
    """Deterministic bimanual controller input for tests and dry runs."""

    def __init__(self, *, dt: float = 1.0 / 30.0) -> None:
        self.dt = float(dt)
        self._started = False
        self._step = 0
        self._start_time = 0.0

    def start(self) -> None:
        self._started = True
        self._step = 0
        self._start_time = time.monotonic()
        trace_event("mock_controller.start", start_monotonic_s=self._start_time)

    def read(self) -> DeviceState:
        if not self._started:
            self.start()

        step = self._step
        timestamp = self._start_time + step * self.dt
        left = ControllerState(
            name="left",
            pose=Pose(
                position=np.array([0.001 * step, 0.0, 0.0]),
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            buttons={"grip": step >= 2},
            axes={"trigger": min(1.0, 0.1 * step)},
            twist=Twist([0.001 / self.dt, 0.0, 0.0], np.zeros(3)),
            timestamp=timestamp,
            actions=(
                {ControllerAction.CALIBRATE} if step == 0 else frozenset()
            ),
            pointer_direction=np.array([0.0, 0.0, 1.0]),
        )
        right = ControllerState(
            name="right",
            pose=Pose(
                position=np.array([0.0, 0.001 * step, 0.0]),
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            buttons={"grip": step >= 3},
            axes={"trigger": min(1.0, 0.05 * step)},
            twist=Twist([0.0, 0.001 / self.dt, 0.0], np.zeros(3)),
            timestamp=timestamp,
            actions=(
                {ControllerAction.CALIBRATE} if step == 1 else frozenset()
            ),
            pointer_direction=np.array([0.0, 0.0, 1.0]),
        )
        self._step += 1
        state = DeviceState(
            timestamp=timestamp, controllers={"left": left, "right": right}
        )
        trace_event("mock_controller.state.result", step=step, state=state)
        return state

    def stop(self) -> None:
        self._started = False
        trace_event("mock_controller.stop")
