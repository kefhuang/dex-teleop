from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from dex_teleop.devices.mock import MockControllerInput


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "visualize_oculus_controllers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "visualize_oculus_controllers_script", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mock_mode_constructs_only_a_mock_controller_device():
    module = load_module()
    args = module.parse_args(["--mock-controller", "--rate-hz", "50"])

    device = module._make_device(args)

    assert isinstance(device, MockControllerInput)
    assert device.dt == pytest.approx(0.02)


def test_run_updates_visualizer_and_closes_both_components():
    module = load_module()

    class FakeDevice:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.read_count = 0

        def start(self):
            self.started = True

        def read(self):
            self.read_count += 1
            if self.read_count == 3:
                raise KeyboardInterrupt
            return f"state-{self.read_count}"

        def stop(self):
            self.stopped = True

    class FakeVisualizer:
        def __init__(self):
            self.closed = False
            self.states = []

        def update(self, state):
            self.states.append(state)

        def close(self):
            self.closed = True

    device = FakeDevice()
    visualizer = FakeVisualizer()

    assert module.run(device, visualizer, rate_hz=1000) == 0
    assert device.started
    assert device.stopped
    assert visualizer.closed
    assert visualizer.states == ["state-1", "state-2"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("--rate-hz", "0"),
        ("--duration", "nan"),
        ("--stale-timeout", "-1"),
        ("--trail-length", "0"),
        ("--port", "70000"),
    ],
)
def test_parse_args_rejects_invalid_numeric_options(option, value):
    module = load_module()

    with pytest.raises(SystemExit):
        module.parse_args([option, value])
