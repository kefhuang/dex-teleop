from __future__ import annotations

from typing import Protocol

from dex_teleop.core.types import DeviceState


class InputDevice(Protocol):
    """Common protocol for controller input devices."""

    def start(self) -> None:
        """Initialize the device and begin producing readings."""
        ...

    def read(self) -> DeviceState:
        """Return the latest controller state."""
        ...

    def stop(self) -> None:
        """Stop the device and release resources."""
        ...
