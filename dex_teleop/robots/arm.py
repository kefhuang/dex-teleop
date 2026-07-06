from __future__ import annotations

from typing import Protocol

from dex_teleop.core.types import ArmState, MotionCommand


class ArmController(Protocol):
    def connect(self) -> None:
        pass

    def get_state(self) -> ArmState:
        pass

    def send_motion(self, command: MotionCommand) -> None:
        pass

    def stop_motion(self) -> None:
        pass

    def close(self) -> None:
        pass
