"""Config 数据类 + YAML 加载。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class SubProblemConfig:
    wave_type: str  # "加工" / "非加工"
    size_class: str  # "le_20" / "gt_20"
    N_max: int


@dataclass(frozen=True)
class SweepConfig:
    num_pickers: int
    pick_time_per_order: float  # 秒/单
    num_machines: int
    process_time_per_jian: float  # 秒/件
    K_ws: int
    daily_cutoff: str  # "HH:MM"
    daily_deadline: str
    picker_shift_start: str
    picker_shift_end: str
    machine_shift_start: str
    machine_shift_end: str


@dataclass(frozen=True)
class MipConfig:
    solver: Literal["gurobi", "scip", "cbc"]
    time_limit: int  # 秒
    mip_gap: float
    fallback: Literal["greedy"]


@dataclass(frozen=True)
class ValidationConfig:
    recomputed_hit_rate: bool
    boundary_check: bool
    inventory_consistency_sample: int
    stockout_filter_sample: int


@dataclass(frozen=True)
class Config:
    sub_problems: dict[str, SubProblemConfig]
    sweep: SweepConfig
    candidate_W: list[str]
    mip: MipConfig
    validation: ValidationConfig


_REQUIRED_SWEEP_FIELDS = [
    "num_pickers", "pick_time_per_order", "num_machines",
    "process_time_per_jian", "K_ws",
    "daily_cutoff", "daily_deadline",
    "picker_shift_start", "picker_shift_end",
    "machine_shift_start", "machine_shift_end",
]


def load_config(path: Path | str) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"config 根节点必须是 dict,实际是 {type(raw).__name__}")

    sub_raw = raw.get("sub_problems", {})
    if not isinstance(sub_raw, dict) or not sub_raw:
        raise ValueError("sub_problems 必须非空 dict")
    sub_problems = {
        name: SubProblemConfig(
            wave_type=sp["wave_type"],
            size_class=sp["size_class"],
            N_max=int(sp["N_max"]),
        )
        for name, sp in sub_raw.items()
    }

    sweep_raw = raw.get("sweep", {})
    missing = [f for f in _REQUIRED_SWEEP_FIELDS if f not in sweep_raw]
    if missing:
        raise ValueError(f"sweep 缺字段:{missing}")

    sweep = SweepConfig(
        num_pickers=int(sweep_raw["num_pickers"]),
        pick_time_per_order=float(sweep_raw["pick_time_per_order"]),
        num_machines=int(sweep_raw["num_machines"]),
        process_time_per_jian=float(sweep_raw["process_time_per_jian"]),
        K_ws=int(sweep_raw["K_ws"]),
        daily_cutoff=str(sweep_raw["daily_cutoff"]),
        daily_deadline=str(sweep_raw["daily_deadline"]),
        picker_shift_start=str(sweep_raw["picker_shift_start"]),
        picker_shift_end=str(sweep_raw["picker_shift_end"]),
        machine_shift_start=str(sweep_raw["machine_shift_start"]),
        machine_shift_end=str(sweep_raw["machine_shift_end"]),
    )

    candidate_W = raw.get("candidate_W", [])
    if not isinstance(candidate_W, list) or not candidate_W:
        raise ValueError("candidate_W 必须是非空 list")

    mip_raw = raw.get("mip", {})
    mip = MipConfig(
        solver=mip_raw.get("solver", "gurobi"),
        time_limit=int(mip_raw.get("time_limit", 60)),
        mip_gap=float(mip_raw.get("mip_gap", 0.05)),
        fallback=mip_raw.get("fallback", "greedy"),
    )

    val_raw = raw.get("validation", {})
    validation = ValidationConfig(
        recomputed_hit_rate=bool(val_raw.get("recomputed_hit_rate", True)),
        boundary_check=bool(val_raw.get("boundary_check", True)),
        inventory_consistency_sample=int(val_raw.get("inventory_consistency_sample", 5)),
        stockout_filter_sample=int(val_raw.get("stockout_filter_sample", 10)),
    )

    return Config(
        sub_problems=sub_problems,
        sweep=sweep,
        candidate_W=candidate_W,
        mip=mip,
        validation=validation,
    )
