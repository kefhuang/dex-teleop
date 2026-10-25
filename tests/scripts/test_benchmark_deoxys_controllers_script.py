from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_deoxys_controllers.py"
    spec = importlib.util.spec_from_file_location("benchmark_deoxys_controllers_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_csv_preserves_order_and_trims_items():
    module = load_module()

    assert module.parse_csv("OSC_POSE, CARTESIAN_VELOCITY,IK_JOINT_IMPEDANCE") == [
        "OSC_POSE",
        "CARTESIAN_VELOCITY",
        "IK_JOINT_IMPEDANCE",
    ]


def test_axis_delta_vectors_use_robot_base_axes():
    module = load_module()

    assert np.allclose(module.axis_delta("x", 0.03), [0.03, 0.0, 0.0])
    assert np.allclose(module.axis_delta("-y", 0.03), [0.0, -0.03, 0.0])
    assert np.allclose(module.axis_delta("+z", 0.03), [0.0, 0.0, 0.03])


def test_scaled_osc_action_inverts_deoxys_action_scale():
    module = load_module()

    class Cfg:
        class action_scale:
            translation = 0.05
            rotation = 1.0

    action = module.scaled_cartesian_action(
        np.array([0.01, -0.02, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        Cfg(),
    )

    assert np.allclose(action, [0.2, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_velocity_action_is_not_divided_when_action_scale_is_one():
    module = load_module()

    class Cfg:
        class action_scale:
            translation = 1.0
            rotation = 1.0

    action = module.scaled_cartesian_action(
        np.array([0.03, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        Cfg(),
    )

    assert np.allclose(action, [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_compute_trial_metrics_reports_main_speed_drift_and_efficiency():
    module = load_module()

    metrics = module.compute_trial_metrics(
        axis="x",
        requested_delta=np.array([0.03, 0.0, 0.0]),
        start_pos=np.array([0.5, 0.0, 0.2]),
        end_pos=np.array([0.515, 0.004, 0.197]),
        duration_s=2.0,
        controller_calls=60,
        ik_solve_s=None,
    )

    assert metrics["main_displacement_m"] == 0.015
    assert round(metrics["main_speed_m_s"], 4) == 0.0075
    assert round(metrics["drift_m"], 4) == 0.005
    assert round(metrics["tracking_efficiency"], 4) == 0.5
    assert metrics["controller_calls"] == 60
    assert metrics["ik_solve_s"] is None


def test_summary_rows_include_plus_minus_symmetry():
    module = load_module()
    rows = [
        {"controller": "OSC_POSE", "axis": "+x", "main_displacement_m": 0.02},
        {"controller": "OSC_POSE", "axis": "-x", "main_displacement_m": -0.01},
    ]

    summary = module.add_symmetry_metrics(rows)

    assert summary[0]["axis_symmetry_ratio"] == 2.0
    assert summary[1]["axis_symmetry_ratio"] == 2.0


def test_validate_args_allows_ten_centimeter_step_but_rejects_larger():
    import argparse

    module = load_module()
    ok = argparse.Namespace(
        controllers="OSC_POSE",
        axes="+x",
        step=0.10,
        duration=1.0,
        rate_hz=30.0,
        max_delta_per_cycle=0.005,
        max_linear_velocity=0.03,
        ik_points=10,
    )
    controllers, axes = module.validate_args(ok)
    assert controllers == ["OSC_POSE"]
    assert axes == ["+x"]

    too_large = argparse.Namespace(**{**vars(ok), "step": 0.11})
    try:
        module.validate_args(too_large)
    except ValueError as exc:
        assert "--step" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_args_supports_reset_between_trials_flag():
    module = load_module()

    args = module.parse_args([
        "--config",
        "franka-two.yml",
        "--reset-between-trials",
    ])

    assert args.reset_between_trials is True


def test_maybe_reset_before_trial_invokes_reset_when_enabled():
    module = load_module()
    calls = []

    class Robot:
        pass

    class Args:
        reset_between_trials = True
        reset_timeout = 3.0

    module.maybe_reset_before_trial(
        Robot(),
        {"controller_type": "JOINT_POSITION"},
        Args(),
        reset_fn=lambda robot, controller_cfg, timeout: calls.append((robot, controller_cfg, timeout)) or True,
    )

    assert len(calls) == 1
    assert calls[0][1] == {"controller_type": "JOINT_POSITION"}
    assert calls[0][2] == 3.0


def test_maybe_reset_before_trial_skips_reset_when_disabled():
    module = load_module()
    calls = []

    class Args:
        reset_between_trials = False
        reset_timeout = 3.0

    module.maybe_reset_before_trial(
        object(),
        {"controller_type": "JOINT_POSITION"},
        Args(),
        reset_fn=lambda robot, controller_cfg, timeout: calls.append(True) or True,
    )

    assert calls == []
