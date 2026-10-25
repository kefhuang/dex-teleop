#!/usr/bin/env python3
"""Visualize Oculus controller positions without constructing any robots."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_teleop.core.loop import RateLimiter
from dex_teleop.devices.mock import MockControllerInput
from dex_teleop.devices.oculus import OculusDevice
from dex_teleop.visualization import Controller3DVisualizer


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show Oculus controller positions in 3D without connecting to a robot"
        )
    )
    parser.add_argument(
        "--ip",
        help="Quest headset IP address for network mode; omit for USB mode",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=5555,
        help="ADB TCP port for network mode (default: 5555)",
    )
    parser.add_argument(
        "--stale-timeout",
        type=_positive_float,
        default=0.15,
        help="Hide controller samples older than this many seconds (default: 0.15)",
    )
    parser.add_argument(
        "--rate-hz",
        type=_positive_float,
        default=30.0,
        help="Visualization refresh rate (default: 30)",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        help="Optional run duration in seconds; otherwise run until closed",
    )
    parser.add_argument(
        "--trail-length",
        type=_positive_int,
        default=120,
        help="Number of recent positions retained per controller (default: 120)",
    )
    parser.add_argument(
        "--mock-controller",
        action="store_true",
        help="Use deterministic simulated controller positions instead of Quest input",
    )
    return parser.parse_args(argv)


def _make_device(args: argparse.Namespace) -> MockControllerInput | OculusDevice:
    if args.mock_controller:
        return MockControllerInput(dt=1.0 / args.rate_hz)
    return OculusDevice(
        ip=args.ip,
        port=args.port,
        stale_timeout=args.stale_timeout,
    )


def run(
    device: Any,
    visualizer: Any,
    *,
    rate_hz: float,
    duration: float | None = None,
) -> int:
    """Run the read-only controller visualization loop."""

    rate = RateLimiter(rate_hz)
    deadline = None if duration is None else time.monotonic() + duration
    try:
        device.start()
        while (
            (deadline is None or time.monotonic() < deadline)
            and not visualizer.closed
        ):
            visualizer.update(device.read())
            rate.sleep()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        try:
            visualizer.close()
        finally:
            device.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "mock" if args.mock_controller else (
        f"network {args.ip}:{args.port}" if args.ip else "USB"
    )
    print(f"[INFO] Starting controller-only visualization ({mode} input).")
    print("[INFO] Close the window or press Ctrl-C to stop; no robot is connected.")

    try:
        device = _make_device(args)
        visualizer = Controller3DVisualizer(trail_length=args.trail_length)
        return run(
            device,
            visualizer,
            rate_hz=args.rate_hz,
            duration=args.duration,
        )
    except Exception as exc:
        print(f"[ERROR] Controller visualization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
