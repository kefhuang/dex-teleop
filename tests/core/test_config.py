from pathlib import Path
import pytest
from dex_teleop.core.config import ConfigError, apply_cli_overrides, load_config

def test_load_configs_use_direct_ik_architecture():
    cfg=load_config('configs/oculus_franka_bimanual.yml')
    assert set(cfg['arms'])=={'left','right'}
    assert 'cartesian_velocity' not in cfg
    assert cfg['safety']['max_translation_from_anchor_m']>0
    assert 'mapping' not in cfg

def test_cli_overrides_do_not_mutate_input():
    cfg=load_config('configs/oculus_franka_right.yml'); out=apply_cli_overrides(cfg,dry_run=False,rate_hz=50)
    assert out['app']['dry_run'] is False and cfg['app']['dry_run'] is True

def test_missing_required_section_reports_path(tmp_path):
    path=tmp_path/'bad.yml'; path.write_text('app: {}')
    with pytest.raises(ConfigError,match='input'): load_config(path)

def test_unknown_gripper_reference_is_rejected(tmp_path):
    text=Path('configs/oculus_franka_right.yml').read_text().replace('gripper: right_gripper','gripper: missing')
    path=tmp_path/'bad.yml'; path.write_text(text)
    with pytest.raises(ConfigError,match='unknown gripper'): load_config(path)
