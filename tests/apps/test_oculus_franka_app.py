from __future__ import annotations

from dex_teleop.apps import oculus_franka
from dex_teleop.apps.oculus_franka import main
from dex_teleop.core.types import MotionCommand, TeleopCommandBatch


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


def test_repeated_disabled_motion_stops_only_during_cleanup(monkeypatch):
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
    }
    monkeypatch.setattr(oculus_franka, "_make_device", lambda cfg: FakeDevice())
    monkeypatch.setattr(oculus_franka, "_make_arms", lambda cfg: {"left": arm})
    monkeypatch.setattr(oculus_franka, "_make_grippers", lambda cfg, arms: {})
    monkeypatch.setattr(oculus_franka, "BimanualClutchPolicy", FakePolicy)

    assert oculus_franka.run(cfg) == 0

    assert arm.sent_commands == []
    assert arm.stop_count == 1
