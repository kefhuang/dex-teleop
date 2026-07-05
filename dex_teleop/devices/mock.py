from __future__ import annotations

import time

import numpy as np

from dex_teleop.core.types import ControllerState, DeviceState, Pose


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

    def read(self) -> DeviceState:
        if not self._started:
            self.start()

        step = self._step
        timestamp = self._start_time + step * self.dt
        left = ControllerState(
            name="left",
            pose=Pose(
                position=np.array([0.001 * step, 0.0, 0.0]),
                orientation=[0.0, 0.0, 0.0, 1.0],
            ),
            buttons={"grip": step >= 1},
            axes={"trigger": min(1.0, 0.1 * step)},
        )
        right = ControllerState(
            name="right",
            pose=Pose(
                position=np.array([0.0, 0.001 * step, 0.0]),
                orientation=[0.0, 0.0, 0.0, 1.0],
            ),
            buttons={"grip": step >= 2},
            axes={"trigger": min(1.0, 0.05 * step)},
        )
        self._step += 1
        return DeviceState(timestamp=timestamp, controllers={"left": left, "right": right})

    def stop(self) -> None:
        self._started = False
