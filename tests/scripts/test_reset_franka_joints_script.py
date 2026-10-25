from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "reset_franka_joints.py"
    spec = importlib.util.spec_from_file_location("reset_franka_joints_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRobot:
    def __init__(self, q_sequence):
        self._q_sequence = [list(q) for q in q_sequence]
        self.control_calls = []
        self.closed = False

    @property
    def state_buffer_size(self):
        return 1 if self._q_sequence else 0

    @property
    def last_q(self):
        return self._q_sequence[0] if self._q_sequence else None

    def control(self, controller_type, action, controller_cfg=None, termination=False):
        self.control_calls.append(
            (controller_type, list(action), controller_cfg, termination)
        )
        if len(self._q_sequence) > 1:
            self._q_sequence.pop(0)

    def close(self):
        self.closed = True


def test_build_reset_action_appends_neutral_gripper_slot():
    module = load_module()

    action = module.build_reset_action([1, 2, 3, 4, 5, 6, 7])

    assert action == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -1.0]


def test_reset_loop_uses_joint_position_default_config_until_target_reached():
    module = load_module()
    target = [0.1, -0.2, 0.0, -2.4, 0.0, 2.3, 0.8]
    robot = FakeRobot([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target,
    ])
    monotonic_values = iter([0.0, 0.01, 0.02])

    reached = module.reset_robot_joints(
        robot,
        controller_cfg={"controller_type": "JOINT_POSITION"},
        target_joints=target,
        timeout=1.0,
        tolerance=1e-3,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: next(monotonic_values),
    )

    assert reached is True
    assert robot.control_calls == [
        (
            "JOINT_POSITION",
            target + [-1.0],
            {"controller_type": "JOINT_POSITION"},
            False,
        )
    ]


def test_main_resolves_config_creates_interface_and_closes_robot(monkeypatch):
    module = load_module()
    constructed = []

    class ConstructedFakeRobot(FakeRobot):
        def __init__(self, config_path, **kwargs):
            super().__init__([module.RESET_JOINTS])
            self.config_path = config_path
            self.kwargs = kwargs
            constructed.append(self)

    monkeypatch.setattr(
        module,
        "import_deoxys",
        lambda: (
            ConstructedFakeRobot,
            lambda controller_type: SimpleNamespace(controller_type=controller_type),
        ),
    )

    exit_code = module.main(["--config", "franka-three.yml", "--timeout", "0.5"])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config_path == (
        module.DEOXYS_ROOT / "config" / "franka-three.yml"
    )
    assert constructed[0].kwargs == {"has_gripper": False, "use_visualizer": False}
    assert constructed[0].closed is True
