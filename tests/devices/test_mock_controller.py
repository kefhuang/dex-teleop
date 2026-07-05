from __future__ import annotations

from dex_teleop.devices.mock import MockControllerInput


def test_mock_controller_returns_left_and_right_states():
    device = MockControllerInput()
    device.start()
    state = device.read()
    device.stop()

    assert set(state.controllers) == {"left", "right"}
    assert "grip" in state.controllers["left"].buttons
    assert "grip" in state.controllers["right"].buttons
    assert "trigger" in state.controllers["left"].axes
    assert "trigger" in state.controllers["right"].axes


def test_mock_controller_script_progresses_deterministically():
    device = MockControllerInput()
    device.start()
    first = device.read()
    second = device.read()

    assert second.timestamp > first.timestamp
    assert second.controllers["left"].pose.position[0] >= first.controllers["left"].pose.position[0]


def test_mock_controller_stop_restart_resets_script():
    device = MockControllerInput(dt=0.5)
    device.start()
    first = device.read()
    _ = device.read()
    device.stop()

    device.start()
    restarted = device.read()

    assert restarted.timestamp != first.timestamp
    assert restarted.controllers["left"].pose.position[0] == first.controllers["left"].pose.position[0]
    assert restarted.controllers["left"].buttons["grip"] == first.controllers["left"].buttons["grip"]
    assert restarted.controllers["right"].axes["trigger"] == first.controllers["right"].axes["trigger"]


def test_mock_controller_read_auto_starts():
    device = MockControllerInput()

    state = device.read()

    assert set(state.controllers) == {"left", "right"}
    assert state.timestamp > 0.0
