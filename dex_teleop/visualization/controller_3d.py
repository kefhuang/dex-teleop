from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from dex_teleop.core.types import DeviceState


def _project_point(
    point: np.ndarray,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    extent_m: float,
    zoom: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Project a tracking-frame point into the visualization canvas."""

    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_azimuth = math.cos(azimuth)
    sin_azimuth = math.sin(azimuth)
    cos_elevation = math.cos(elevation)
    sin_elevation = math.sin(elevation)

    x, y, z = np.asarray(point, dtype=float)
    view_x = cos_azimuth * x - sin_azimuth * y
    view_depth = sin_azimuth * x + cos_azimuth * y
    view_y = cos_elevation * z - sin_elevation * view_depth
    depth = cos_elevation * view_depth + sin_elevation * z

    camera_distance = 4.0 * extent_m
    perspective = camera_distance / max(camera_distance - depth, 0.1 * extent_m)
    pixels_per_metre = 0.42 * min(width, height) * zoom / extent_m
    return (
        width / 2.0 + view_x * pixels_per_metre * perspective,
        height / 2.0 - view_y * pixels_per_metre * perspective,
    )


class Controller3DVisualizer:
    """Live 3D view of controller positions in the Oculus tracking frame."""

    _CONTROLLER_COLORS = {
        "left": "#ffb347",
        "right": "#45d6ff",
    }
    _AXIS_COLORS = {
        "X": "#ff6868",
        "Y": "#63dd7a",
        "Z": "#6b9fff",
    }

    def __init__(
        self,
        *,
        trail_length: int = 120,
        width: int = 720,
        height: int = 720,
        min_extent_m: float = 0.5,
    ) -> None:
        if trail_length <= 0:
            raise ValueError("trail_length must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if not math.isfinite(min_extent_m) or min_extent_m <= 0.0:
            raise ValueError("min_extent_m must be finite and positive")

        try:
            import tkinter as tk
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "Controller visualization requires Python tkinter support"
            ) from exc

        self._tk = tk
        self._root: Any | None = None
        self._closed = False
        try:
            self._root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - depends on display server
            raise RuntimeError(
                "Cannot open the controller visualization window; "
                "check that a graphical display is available"
            ) from exc

        self._root.title("Oculus controller positions")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._canvas = tk.Canvas(
            self._root,
            width=width,
            height=height,
            background="#10151d",
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        self._width = int(width)
        self._height = int(height)
        self._extent_m = float(min_extent_m)
        self._zoom = 1.0
        self._azimuth_deg = -45.0
        self._elevation_deg = 25.0
        self._drag_at: tuple[int, int] | None = None
        self._trail_length = int(trail_length)
        self._trails: dict[str, deque[np.ndarray]] = {}
        self._current: dict[str, np.ndarray] = {}

        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self._canvas.bind("<Button-4>", lambda _event: self._change_zoom(1.1))
        self._canvas.bind("<Button-5>", lambda _event: self._change_zoom(1 / 1.1))
        self._redraw()
        self._pump_events()

    @property
    def closed(self) -> bool:
        return self._closed

    def update(self, state: DeviceState) -> None:
        """Display one device sample without blocking the control loop."""

        if self._closed:
            return

        current: dict[str, np.ndarray] = {}
        for name, controller in state.controllers.items():
            position = np.asarray(controller.pose.position, dtype=float).copy()
            current[name] = position
            trail = self._trails.setdefault(
                name, deque(maxlen=self._trail_length)
            )
            trail.append(position)
            required_extent = 1.2 * float(np.max(np.abs(position)))
            self._extent_m = max(self._extent_m, required_extent)
        self._current = current

        self._redraw()
        self._pump_events()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        root, self._root = self._root, None
        if root is not None:
            try:
                root.destroy()
            except self._tk.TclError:
                pass

    def _project(self, point: np.ndarray) -> tuple[float, float]:
        return _project_point(
            point,
            azimuth_deg=self._azimuth_deg,
            elevation_deg=self._elevation_deg,
            extent_m=self._extent_m,
            zoom=self._zoom,
            width=self._width,
            height=self._height,
        )

    def _pump_events(self) -> None:
        if self._closed or self._root is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
            self._root = None

    def _redraw(self) -> None:
        if self._closed:
            return
        canvas = self._canvas
        try:
            canvas.delete("all")
        except self._tk.TclError:
            self._closed = True
            return

        self._draw_grid()
        self._draw_axes()
        self._draw_controller_trails()

        canvas.create_text(
            14,
            12,
            anchor="nw",
            text="Oculus tracking frame  |  position in metres",
            fill="#d8e1eb",
            font=("TkDefaultFont", 12, "bold"),
        )
        canvas.create_text(
            14,
            34,
            anchor="nw",
            text="Drag to rotate  |  Scroll to zoom",
            fill="#8493a5",
        )
        if not self._current:
            canvas.create_text(
                self._width / 2,
                self._height - 30,
                text="Waiting for a fresh controller sample...",
                fill="#8493a5",
            )

    def _draw_grid(self) -> None:
        extent = self._extent_m
        for index in range(-4, 5):
            offset = extent * index / 4.0
            shade = "#34404e" if index == 0 else "#222c37"
            x_start = self._project(np.array([-extent, offset, 0.0]))
            x_end = self._project(np.array([extent, offset, 0.0]))
            y_start = self._project(np.array([offset, -extent, 0.0]))
            y_end = self._project(np.array([offset, extent, 0.0]))
            self._canvas.create_line(*x_start, *x_end, fill=shade)
            self._canvas.create_line(*y_start, *y_end, fill=shade)

    def _draw_axes(self) -> None:
        origin = np.zeros(3)
        extent = self._extent_m
        endpoints = {
            "X": np.array([extent, 0.0, 0.0]),
            "Y": np.array([0.0, extent, 0.0]),
            "Z": np.array([0.0, 0.0, extent]),
        }
        origin_2d = self._project(origin)
        for label, endpoint in endpoints.items():
            endpoint_2d = self._project(endpoint)
            color = self._AXIS_COLORS[label]
            self._canvas.create_line(
                *origin_2d,
                *endpoint_2d,
                fill=color,
                width=3,
                arrow="last",
            )
            self._canvas.create_text(
                endpoint_2d[0],
                endpoint_2d[1] - 12,
                text=label,
                fill=color,
                font=("TkDefaultFont", 11, "bold"),
            )

    def _draw_controller_trails(self) -> None:
        fallback_colors = ("#e970ff", "#ffe66d", "#b8f28b", "#ff8d6b")
        for color_index, name in enumerate(sorted(self._trails)):
            color = self._CONTROLLER_COLORS.get(
                name, fallback_colors[color_index % len(fallback_colors)]
            )
            trail = self._trails[name]
            if len(trail) >= 2:
                coords = [
                    coordinate
                    for point in trail
                    for coordinate in self._project(point)
                ]
                self._canvas.create_line(*coords, fill=color, width=2)

            position = self._current.get(name)
            if position is None:
                continue
            x, y = self._project(position)
            radius = 7
            self._canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="#ffffff",
                width=2,
            )
            values = ", ".join(f"{value:+.3f}" for value in position)
            self._canvas.create_text(
                x + 11,
                y - 11,
                anchor="sw",
                text=f"{name}: [{values}] m",
                fill=color,
                font=("TkFixedFont", 10),
            )

    def _on_resize(self, event: Any) -> None:
        self._width = max(1, int(event.width))
        self._height = max(1, int(event.height))
        self._redraw()

    def _on_drag_start(self, event: Any) -> None:
        self._drag_at = (int(event.x), int(event.y))

    def _on_drag(self, event: Any) -> None:
        if self._drag_at is None:
            return
        previous_x, previous_y = self._drag_at
        current_x, current_y = int(event.x), int(event.y)
        self._azimuth_deg += 0.5 * (current_x - previous_x)
        self._elevation_deg = min(
            85.0,
            max(-85.0, self._elevation_deg + 0.5 * (previous_y - current_y)),
        )
        self._drag_at = (current_x, current_y)
        self._redraw()

    def _on_mouse_wheel(self, event: Any) -> None:
        self._change_zoom(1.1 if event.delta > 0 else 1 / 1.1)

    def _change_zoom(self, factor: float) -> None:
        self._zoom = min(3.0, max(0.4, self._zoom * factor))
        self._redraw()
