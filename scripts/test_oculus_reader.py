#!/usr/bin/env python3
"""Quick diagnostic for the shared Oculus Reader.

This script checks whether the Oculus/Quest reader can connect through ADB and
receive controller transforms/buttons from the headset app.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether shared oculus_reader can read Quest controller data.",
    )
    parser.add_argument(
        "--ip",
        default=None,
        help="Quest headset IP address for network mode. Omit for USB mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ADB TCP port for network mode. Default: 5555.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to sample Oculus Reader output. Default: 5.0.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Seconds between samples. Default: 0.25.",
    )
    parser.add_argument(
        "--skip-adb-check",
        action="store_true",
        help="Skip the preliminary `adb devices` check.",
    )
    parser.add_argument(
        "--print-fps",
        action="store_true",
        help="Enable OculusReader's internal FPS printout.",
    )
    return parser.parse_args()


def run_adb_devices() -> bool:
    if shutil.which("adb") is None:
        print("[WARN] `adb` executable not found on PATH.")
        print("       Install android-tools-adb or add adb to PATH before hardware testing.")
        return False

    print("[INFO] Running `adb devices`...")
    result = subprocess.run(
        ["adb", "devices"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0:
        print(f"[WARN] `adb devices` exited with code {result.returncode}.")
        return False

    device_lines = [
        line
        for line in result.stdout.splitlines()[1:]
        if line.strip() and not line.strip().startswith("*")
    ]
    ready_devices = [line for line in device_lines if "\tdevice" in line]
    if not ready_devices:
        print("[WARN] No ready ADB device found. OculusReader may still fail to connect.")
        print("       Put on the headset and accept USB debugging if prompted.")
        return False

    print(f"[INFO] Found {len(ready_devices)} ready ADB device(s).")
    return True


def import_oculus_reader() -> type[Any]:
    from oculus_reader.reader import OculusReader  # noqa: PLC0415

    return OculusReader


def has_data(transforms: Any, buttons: Any) -> bool:
    return bool(transforms) or bool(buttons)


def summarize_sample(transforms: Any, buttons: Any) -> str:
    transform_keys = sorted(transforms.keys()) if isinstance(transforms, dict) else []
    button_keys = sorted(buttons.keys()) if isinstance(buttons, dict) else []
    return f"transforms={transform_keys}, buttons={button_keys}"


def main() -> int:
    args = parse_args()

    if args.duration <= 0:
        print("[ERROR] --duration must be positive.", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("[ERROR] --interval must be positive.", file=sys.stderr)
        return 2

    if not args.skip_adb_check:
        run_adb_devices()

    try:
        OculusReader = import_oculus_reader()
    except Exception as exc:  # noqa: BLE001 - diagnostic should show import failures clearly.
        print(f"[ERROR] Failed to import OculusReader: {exc}", file=sys.stderr)
        return 1

    print("[INFO] Starting OculusReader...")
    if args.ip:
        print(f"[INFO] Network mode: {args.ip}:{args.port}")
    else:
        print("[INFO] USB mode")

    reader = None
    samples = []
    try:
        reader = OculusReader(
            ip_address=args.ip,
            port=args.port,
            print_FPS=args.print_fps,
            run=True,
        )
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            transforms, buttons = reader.get_transformations_and_buttons()
            if has_data(transforms, buttons):
                samples.append((transforms, buttons))
                print(f"[DATA] {summarize_sample(transforms, buttons)}")
            else:
                print("[WAIT] No transforms/buttons yet...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print(f"[ERROR] OculusReader exited early with code {code}.", file=sys.stderr)
        return code
    except Exception as exc:  # noqa: BLE001 - keep this useful as a hardware diagnostic.
        print(f"[ERROR] OculusReader failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if reader is not None:
            print("[INFO] Stopping OculusReader...")
            reader.stop()

    if samples:
        transforms, buttons = samples[-1]
        print("[PASS] Received Oculus Reader data.")
        print(f"[PASS] Last sample: {summarize_sample(transforms, buttons)}")
        return 0

    print("[FAIL] Did not receive Oculus Reader data within the requested duration.")
    print("       Check headset app launch, USB debugging permission, and controller tracking.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
