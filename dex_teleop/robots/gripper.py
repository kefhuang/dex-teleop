from __future__ import annotations

from typing import Protocol

from dex_teleop.core.types import GripperCommand


class GripperController(Protocol):
    def connect(self) -> None:
        pass

    def send_gripper(self, command: GripperCommand) -> None:
        pass

    def close(self) -> None:
        pass
