from __future__ import annotations

from dex_teleop.apps.oculus_franka import main


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
