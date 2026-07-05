from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dex_teleop.core.types import ControllerState, DeviceState, Pose


REPO_ROOT = Path(__file__).resolve().parents[2]
OCULUS_READER_ROOT = REPO_ROOT / "dependency" / "oculus_reader"


class OculusDevice:
    """Thin adapter around the vendored Oculus Reader package."""

    def __init__(self, *, ip: str | None = None, port: int = 5555, print_fps: bool = False) -> None:
        self.ip = ip
        self.port = int(port)
        self.print_fps = bool(print_fps)
        self._reader: Any | None = None

    def start(self) -> None:
        if not OCULUS_READER_ROOT.exists():
            raise RuntimeError(f"Missing Oculus Reader dependency: {OCULUS_READER_ROOT}")
        reader_root = str(OCULUS_READER_ROOT)
        if reader_root not in sys.path:
            sys.path.insert(0, reader_root)

        from oculus_reader.reader import OculusReader  # noqa: PLC0415

        self._reader = OculusReader(
            ip_address=self.ip,
            port=self.port,
            print_FPS=self.print_fps,
            run=True,
        )

    def read(self) -> DeviceState:
        if self._reader is None:
            raise RuntimeError("OculusDevice.start() must be called before read()")
        transforms, buttons = self._reader.get_transformations_and_buttons()
        return DeviceState(
            timestamp=time.monotonic(),
            controllers=self._convert(transforms or {}, buttons or {}),
        )

    def stop(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

    def _convert(self, transforms: dict[str, Any], buttons: dict[str, Any]) -> dict[str, ControllerState]:
        controllers: dict[str, ControllerState] = {}
        for name in ("left", "right"):
            matrix = self._find_transform(transforms, name)
            pose = self._pose_from_matrix(matrix) if matrix is not None else Pose.identity()
            controllers[name] = ControllerState(
                name=name,
                pose=pose,
                buttons={"grip": self._button_bool(buttons, name, "grip")},
                axes={"trigger": self._button_float(buttons, name, "trigger")},
            )
        return controllers

    def _find_transform(self, transforms: dict[str, Any], side: str) -> Any | None:
        side_key = side[0]
        for key, value in transforms.items():
            lowered = str(key).lower()
            if lowered == side_key or lowered == side:
                return value
            if side in lowered:
                return value
        return None

    def _pose_from_matrix(self, matrix: Any) -> Pose:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape == (4, 4):
            position = arr[:3, 3]
        elif arr.shape == (3,):
            position = arr
        else:
            position = np.zeros(3)
        return Pose(position=position, orientation=[0.0, 0.0, 0.0, 1.0])

    def _button_bool(self, buttons: dict[str, Any], side: str, name: str) -> bool:
        for candidate in self._button_candidates(side, name):
            if candidate in buttons:
                value = buttons[candidate]
                if isinstance(value, tuple):
                    return bool(value[0]) if value else False
                return bool(value)
        return False

    def _button_float(self, buttons: dict[str, Any], side: str, name: str) -> float:
        for candidate in self._button_candidates(side, name):
            if candidate in buttons:
                value = buttons[candidate]
                if isinstance(value, tuple):
                    value = value[0] if value else 0.0
                return float(min(1.0, max(0.0, float(value))))
        return 0.0

    def _button_candidates(self, side: str, name: str) -> list[str]:
        side_initial = side[0]
        oculus_side = "left" if side == "left" else "right"
        oculus_initial = "L" if side == "left" else "R"
        if name == "grip":
            oculus_names = ["Grip", "G"]
        elif name == "trigger":
            oculus_names = ["Trig", "Tr"]
        else:
            oculus_names = [name]
        return [
            f"{side}_{name}",
            f"{side_initial}_{name}",
            f"{side}{name}",
            f"{oculus_side}{oculus_names[0]}",
            f"{oculus_initial}{oculus_names[-1]}",
            name,
        ]
