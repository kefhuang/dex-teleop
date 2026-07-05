from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "test_deoxys_control.py"
    spec = importlib.util.spec_from_file_location("test_deoxys_control_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_plan_defaults_to_no_backend_start():
    module = load_module()

    plan = module.build_backend_plan(start_backend=False, configs=["franka-two.yml", "franka-three.yml"])

    assert plan is None


def test_backend_plan_uses_auto_bimanual_for_two_configs():
    module = load_module()

    plan = module.build_backend_plan(start_backend=True, configs=["franka-two.yml", "franka-three.yml"])

    assert plan is not None
    assert plan.session_name == "bimanual"
    assert plan.command[:2] == ["bash", "auto_scripts/auto_bimanual.sh"]
    assert plan.command[-2:] == ["config/franka-two.yml", "config/franka-three.yml"]


def test_backend_plan_rejects_start_backend_without_two_configs():
    module = load_module()

    try:
        module.build_backend_plan(start_backend=True, configs=["franka-two.yml"])
    except ValueError as exc:
        assert "requires exactly two" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_read_only_guard_rejects_control_methods():
    module = load_module()

    class FakeInterface:
        def control(self):
            raise RuntimeError("should not be called")

        def preprocess(self):
            raise RuntimeError("should not be called")

        def gripper_control(self):
            raise RuntimeError("should not be called")

    assert module.assert_read_only_interface_usage(FakeInterface()) is True
