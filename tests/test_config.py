from pathlib import Path
import pytest
from src.config import Config, SubProblemConfig, SweepConfig, MipConfig, load_config


def test_load_config_minimal(tmp_path):
    yaml_content = """
sub_problems:
  jia_gong_le_20:
    wave_type: 加工
    size_class: le_20
    N_max: 50
sweep:
  num_pickers: 4
  pick_time_per_order: 120
  num_machines: 2
  process_time_per_jian: 36
  K_ws: 8
  daily_cutoff: "14:00"
  daily_deadline: "18:00"
  picker_shift_start: "08:00"
  picker_shift_end: "18:00"
  machine_shift_start: "08:00"
  machine_shift_end: "18:00"
candidate_W: ["1h", "2h", "4h"]
mip:
  solver: gurobi
  time_limit: 60
  mip_gap: 0.05
  fallback: greedy
validation:
  recomputed_hit_rate: true
  boundary_check: true
  inventory_consistency_sample: 5
  stockout_filter_sample: 10
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(p)

    assert isinstance(cfg, Config)
    assert "jia_gong_le_20" in cfg.sub_problems
    assert cfg.sub_problems["jia_gong_le_20"].N_max == 50
    assert cfg.sweep.num_pickers == 4
    assert cfg.candidate_W == ["1h", "2h", "4h"]
    assert cfg.mip.solver == "gurobi"
    assert cfg.mip.mip_gap == 0.05


def test_load_config_missing_field_raises(tmp_path):
    yaml_content = """
sub_problems:
  jia_gong_le_20:
    wave_type: 加工
    size_class: le_20
    N_max: 50
sweep:
  num_pickers: 4
candidate_W: ["1h"]
mip:
  solver: gurobi
  time_limit: 60
  mip_gap: 0.05
  fallback: greedy
validation:
  recomputed_hit_rate: true
  boundary_check: true
  inventory_consistency_sample: 5
  stockout_filter_sample: 10
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="sweep"):
        load_config(p)
