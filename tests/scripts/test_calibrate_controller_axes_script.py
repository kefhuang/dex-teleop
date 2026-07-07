from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dex_teleop.core.types import ControllerState, DeviceState, Pose
from scripts.calibrate_controller_axes import (
    AxisCalibrationError,
    dominant_axis_spec,
    format_mapping_yaml,
    infer_mapping,
    run_calibration_from_deltas,
)


def test_dominant_axis_spec_returns_signed_largest_component():
    assert dominant_axis_spec(np.array([0.01, -0.05, 0.02]), ("x", "y", "z"), 0.01) == "-y"


def test_dominant_axis_spec_rejects_too_small_delta():
    with pytest.raises(AxisCalibrationError, match="too small"):
        dominant_axis_spec(np.array([0.001, 0.0, 0.0]), ("x", "y", "z"), 0.01)


def test_infer_mapping_maps_each_output_axis_to_dominant_input_axis():
    mapping = infer_mapping(
        {
            "x": np.array([0.0, 0.0, 0.04]),
            "y": np.array([-0.03, 0.0, 0.0]),
            "z": np.array([0.0, 0.02, 0.0]),
        },
        output_axes=("x", "y", "z"),
        input_axes=("x", "y", "z"),
        min_delta=0.01,
    )

    assert mapping == {"x": "+z", "y": "-x", "z": "+y"}


def test_format_mapping_yaml_prints_position_and_rotation_sections():
    text = format_mapping_yaml(
        {"x": "+z", "y": "-x", "z": "+y"},
        {"roll": "+pitch", "pitch": "-roll", "yaw": "+yaw"},
    )

    assert "position_axes:" in text
    assert "  x: \"+z\"" in text
    assert "rotation_axes:" in text
    assert "  pitch: \"-roll\"" in text


def test_run_calibration_from_deltas_returns_both_mappings():
    position, rotation = run_calibration_from_deltas(
        position_deltas={
            "x": np.array([0.0, 0.0, 0.04]),
            "y": np.array([-0.03, 0.0, 0.0]),
            "z": np.array([0.0, 0.02, 0.0]),
        },
        rotation_deltas={
            "roll": np.array([0.0, 0.2, 0.0]),
            "pitch": np.array([-0.3, 0.0, 0.0]),
            "yaw": np.array([0.0, 0.0, 0.4]),
        },
        min_position_delta=0.01,
        min_rotation_delta=0.05,
    )

    assert position == {"x": "+z", "y": "-x", "z": "+y"}
    assert rotation == {"roll": "+pitch", "pitch": "-roll", "yaw": "+yaw"}


def test_read_controller_pose_raises_for_missing_controller():
    from scripts.calibrate_controller_axes import read_controller_pose

    state = DeviceState(timestamp=1.0, controllers={})

    with pytest.raises(RuntimeError, match="controller 'left'"):
        read_controller_pose(state, "left")


def test_read_controller_pose_returns_selected_pose_copy():
    from scripts.calibrate_controller_axes import read_controller_pose

    pose = Pose(position=[1.0, 2.0, 3.0], orientation=[0.0, 0.0, 0.0, 1.0])
    state = DeviceState(
        timestamp=1.0,
        controllers={"left": ControllerState(name="left", pose=pose)},
    )

    result = read_controller_pose(state, "left")
    pose.position[:] = 0.0

    assert np.allclose(result.position, [1.0, 2.0, 3.0])


def _state_with_pose(position, orientation=(0.0, 0.0, 0.0, 1.0)):
    return DeviceState(
        timestamp=1.0,
        controllers={
            "left": ControllerState(
                name="left",
                pose=Pose(position=position, orientation=orientation),
            )
        },
    )


def test_run_calibration_guides_position_then_rotation_and_prints_yaml():
    from scripts.calibrate_controller_axes import run_calibration

    yaw = (0.0, 0.0, np.sin(0.2 / 2.0), np.cos(0.2 / 2.0))
    pitch_negative = (0.0, np.sin(-0.25 / 2.0), 0.0, np.cos(-0.25 / 2.0))
    roll = (np.sin(0.3 / 2.0), 0.0, 0.0, np.cos(0.3 / 2.0))
    states = iter(
        [
            _state_with_pose([0.0, 0.0, 0.0]),
            _state_with_pose([0.0, 0.0, 0.04]),
            _state_with_pose([0.0, 0.0, 0.0]),
            _state_with_pose([-0.03, 0.0, 0.0]),
            _state_with_pose([0.0, 0.0, 0.0]),
            _state_with_pose([0.0, 0.02, 0.0]),
            _state_with_pose([0.0, 0.0, 0.0]),
            _state_with_pose([0.0, 0.0, 0.0], orientation=pitch_negative),
            _state_with_pose([0.0, 0.0, 0.0]),
            _state_with_pose([0.0, 0.0, 0.0], orientation=roll),
            _state_with_pose([0.0, 0.0, 0.0]),
            _state_with_pose([0.0, 0.0, 0.0], orientation=yaw),
        ]
    )
    output: list[str] = []

    result = run_calibration(
        lambda: next(states),
        "left",
        samples=1,
        sample_delay=0.0,
        min_position_delta=0.01,
        min_rotation_delta=0.05,
        input_func=lambda prompt: "",
        print_func=output.append,
    )

    assert result.position_axes == {"x": "+z", "y": "-x", "z": "+y"}
    assert result.rotation_axes == {"roll": "-pitch", "pitch": "+roll", "yaw": "+yaw"}
    assert "Suggested config snippet" in "\n".join(output)


def test_script_help_runs_when_invoked_by_path():
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_controller_axes.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Guided Oculus controller axis calibration" in result.stdout
