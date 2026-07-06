from __future__ import annotations

from pathlib import Path

import pytest

from dex_teleop.apps import oculus_franka
from dex_teleop.apps.oculus_franka import main
from dex_teleop.core.types import GripperCommand, MotionCommand, TeleopCommandBatch


def test_app_dry_run_mock_controller_smoke(capsys):
    code = main(
        [
            "--config",
            "configs/oculus_franka_bimanual.yml",
            "--mock-controller",
            "--dry-run",
            "--duration",
            "0.2",
            "--rate-hz",
            "30",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "dry_run=True" in captured.out
    assert "mock_controller=True" in captured.out
    assert "Teleop shutdown complete" in captured.out


def test_app_missing_config_returns_config_error(capsys):
    code = main(
        [
            "--config",
            "configs/does-not-exist.yml",
            "--mock-controller",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "Config error" in captured.out


def test_repeated_disabled_motion_sends_zero_keepalive_when_configured(monkeypatch):
    class FakeDevice:
        def __init__(self) -> None:
            self.reads = 0

        def start(self) -> None:
            pass

        def read(self):
            self.reads += 1
            if self.reads > 3:
                raise KeyboardInterrupt
            return object()

        def stop(self) -> None:
            pass

    class FakeArm:
        def __init__(self) -> None:
            self.stop_count = 0
            self.sent_commands: list[MotionCommand] = []

        def connect(self) -> None:
            pass

        def send_motion(self, command: MotionCommand) -> None:
            self.sent_commands.append(command)

        def stop_motion(self) -> None:
            self.stop_count += 1

        def close(self) -> None:
            pass

    class FakePolicy:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def update(self, state):
            return TeleopCommandBatch(motions={"left": MotionCommand.disabled("left")})

    arm = FakeArm()
    cfg = {
        "app": {"dry_run": True, "rate_hz": 1000},
        "input": {"mock_controller": True},
        "arms": {"left": {}},
        "grippers": {},
        "safety": {"send_zero_when_idle": True},
    }
    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "_make_arms", lambda cfg: {"left": arm})
    monkeypatch.setattr(oculus_franka, "_make_grippers", lambda cfg, arms: {})
    monkeypatch.setattr(oculus_franka, "BimanualClutchPolicy", FakePolicy)

    assert oculus_franka.run(cfg) == 0

    assert arm.sent_commands == []
    assert arm.stop_count == 4


def test_run_ignores_gripper_command_for_unconfigured_gripper(monkeypatch):
    class FakeDevice:
        def __init__(self) -> None:
            self.reads = 0

        def start(self) -> None:
            pass

        def read(self):
            self.reads += 1
            if self.reads > 1:
                raise KeyboardInterrupt
            return object()

        def stop(self) -> None:
            pass

    class FakeArm:
        def connect(self) -> None:
            pass

        def send_motion(self, command: MotionCommand) -> None:
            pass

        def stop_motion(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakePolicy:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def update(self, state):
            return TeleopCommandBatch(
                motions={"left": MotionCommand.disabled("left")},
                grippers={"missing_gripper": GripperCommand("missing_gripper", 0.5)},
            )

    cfg = {
        "app": {"dry_run": True, "rate_hz": 1000},
        "input": {"mock_controller": True},
        "arms": {"left": {"gripper": None}},
        "grippers": {},
    }
    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "_make_arms", lambda cfg: {"left": FakeArm()})
    monkeypatch.setattr(oculus_franka, "_make_grippers", lambda cfg, arms: {})
    monkeypatch.setattr(oculus_franka, "BimanualClutchPolicy", FakePolicy)

    assert oculus_franka.run(cfg) == 0


def test_real_app_connects_arms_before_creating_grippers(monkeypatch):
    events: list[str] = []

    class FakeDevice:
        def start(self) -> None:
            events.append("device.start")

        def read(self):
            raise KeyboardInterrupt

        def stop(self) -> None:
            events.append("device.stop")

    class FakeArm:
        def __init__(self, arm_id: str, interface_cfg: str) -> None:
            self.arm_id = arm_id
            self.interface_cfg = interface_cfg
            self.interface = None

        def connect(self) -> None:
            events.append(f"{self.arm_id}.connect")
            self.interface = {"arm_id": self.arm_id}

        def stop_motion(self) -> None:
            events.append(f"{self.arm_id}.stop")

        def close(self) -> None:
            events.append(f"{self.arm_id}.close")

    class FakeGripper:
        def __init__(self, gripper_id: str, *, interface) -> None:
            events.append(f"{gripper_id}.created")
            assert interface == {"arm_id": "left"}
            self.gripper_id = gripper_id
            self.interface = interface

        def connect(self) -> None:
            events.append(f"{self.gripper_id}.connect")

        def send_gripper(self, command) -> None:
            raise AssertionError("loop should stop before commands are sent")

        def close(self) -> None:
            events.append(f"{self.gripper_id}.close")

    class FakePolicy:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def update(self, state):
            raise AssertionError("device read interrupts before policy update")

    cfg = {
        "app": {"dry_run": False, "rate_hz": 1000},
        "input": {"mock_controller": True},
        "arms": {
            "left": {
                "type": "franka",
                "interface_cfg": "dependency/deoxys_control/deoxys/config/franka-two.yml",
            }
        },
        "grippers": {"left_gripper": {"type": "franka", "arm": "left"}},
    }
    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "FrankaArmController", FakeArm)
    monkeypatch.setattr(oculus_franka, "FrankaGripperController", FakeGripper)
    monkeypatch.setattr(oculus_franka, "BimanualClutchPolicy", FakePolicy)

    assert oculus_franka.run(cfg) == 0

    assert events.index("left.connect") < events.index("left_gripper.created")


def test_dry_run_make_grippers_uses_mock_grippers_without_arm_interface():
    class FakeArmWithoutInterface:
        interface = None

    cfg = {
        "app": {"dry_run": True},
        "grippers": {"left_gripper": {"arm": "left"}},
    }

    grippers = oculus_franka._make_grippers(cfg, {"left": FakeArmWithoutInterface()})

    assert isinstance(grippers["left_gripper"], oculus_franka.MockGripperController)


def test_real_run_missing_interface_cfg_fails_before_franka_construction(monkeypatch):
    constructed: list[str] = []
    started: list[str] = []

    class FakeDevice:
        def start(self) -> None:
            started.append("device.start")

        def stop(self) -> None:
            started.append("device.stop")

    def fail_if_franka_constructed(arm_id: str, interface_cfg: str):
        constructed.append(f"{arm_id}:{interface_cfg}")
        raise AssertionError("FrankaArmController should not be constructed")

    cfg = {
        "app": {"dry_run": False, "rate_hz": 1000},
        "input": {"mock_controller": True},
        "arms": {
            "left": {
                "type": "franka",
                "interface_cfg": "configs/does-not-exist-franka.yml",
            }
        },
        "grippers": {"left_gripper": {"type": "franka", "arm": "left"}},
    }
    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "FrankaArmController", fail_if_franka_constructed)

    with pytest.raises(RuntimeError) as excinfo:
        oculus_franka.run(cfg)

    message = str(excinfo.value)
    assert "interface_cfg" in message
    assert "does not exist" in message
    assert "configs/does-not-exist-franka.yml" in message
    assert str(Path("configs/does-not-exist-franka.yml")) in message
    assert constructed == []
    assert started == []


def test_real_run_rejects_unsupported_arm_and_gripper_types():
    cfg = {
        "app": {"dry_run": False, "rate_hz": 1000},
        "input": {"mock_controller": True},
        "arms": {
            "left": {
                "type": "mock",
                "interface_cfg": "dependency/deoxys_control/deoxys/config/franka-two.yml",
            }
        },
        "grippers": {"left_gripper": {"type": "mock", "arm": "left"}},
    }

    with pytest.raises(RuntimeError) as excinfo:
        oculus_franka.run(cfg)

    message = str(excinfo.value)
    assert "Unsupported real arm type" in message
    assert "left" in message
    assert "franka" in message


def test_real_oculus_startup_error_mentions_adb_quest_or_mock_controller(
    monkeypatch, capsys
):
    class FailingDevice:
        def start(self) -> None:
            raise RuntimeError("oculus unavailable")

        def stop(self) -> None:
            pass

    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FailingDevice())

    code = main(
        [
            "--config",
            "configs/oculus_franka_bimanual.yml",
            "--real-robot",
            "--real-controller",
            "--duration",
            "0.01",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "Teleop error" in captured.out
    assert (
        "--mock-controller" in captured.out
        or "ADB" in captured.out
        or "Quest" in captured.out
    )


def test_real_franka_connect_error_mentions_manually_started_deoxys_backend(
    monkeypatch, capsys
):
    class FakeDevice:
        def start(self) -> None:
            pass

        def read(self):
            raise AssertionError("loop should not run after Franka connect failure")

        def stop(self) -> None:
            pass

    class FailingArm:
        def __init__(self, arm_id: str, interface_cfg: str) -> None:
            self.arm_id = arm_id
            self.interface_cfg = interface_cfg
            self.interface = None

        def connect(self) -> None:
            raise RuntimeError("cannot reach robot")

        def stop_motion(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "FrankaArmController", FailingArm)

    code = main(
        [
            "--config",
            "configs/oculus_franka_bimanual.yml",
            "--real-robot",
            "--mock-controller",
            "--duration",
            "0.01",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "Teleop error" in captured.out
    assert "Deoxys backend" in captured.out
    assert "manually started" in captured.out


def test_app_print_commands_outputs_dry_run_commands(capsys):
    code = main(
        [
            "--config",
            "configs/oculus_franka_bimanual.yml",
            "--mock-controller",
            "--dry-run",
            "--duration",
            "0.2",
            "--rate-hz",
            "30",
            "--print-commands",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "controllers=" in captured.out
    assert "motion[left]" in captured.out
    assert "gripper[left_gripper]" in captured.out
