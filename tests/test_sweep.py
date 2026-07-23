from pathlib import Path
import datetime

import pytest

from src.config import (
    Config,
    SubProblemConfig,
    SweepConfig,
    MipConfig,
    ValidationConfig,
)
from src.data_loader import load_orders, load_inventory_snapshots, Order, OrderLine
from src.sweep import simulate_window, WindowResult


def _make_config(N_max=2):
    return Config(
        sub_problems={
            "fei_jia_gong_le_20": SubProblemConfig(
                wave_type="非加工",
                size_class="le_20",
                N_max=N_max,
            ),
        },
        sweep=SweepConfig(
            num_pickers=4,
            pick_time_per_order=120,
            num_machines=2,
            process_time_per_jian=36,
            K_ws=8,
            daily_cutoff="14:00",
            daily_deadline="18:00",
            picker_shift_start="08:00",
            picker_shift_end="18:00",
            machine_shift_start="08:00",
            machine_shift_end="18:00",
        ),
        candidate_W=["1h"],
        mip=MipConfig(
            solver="gurobi",
            time_limit=60,
            mip_gap=0.05,
            fallback="greedy",
        ),
        validation=ValidationConfig(
            recomputed_hit_rate=True,
            boundary_check=True,
            inventory_consistency_sample=5,
            stockout_filter_sample=10,
        ),
    )


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_case"


def test_simulate_window_basic():
    """对 tiny_case 跑 simulate_window,得最优总访问 3。"""
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)

    result = simulate_window(
        window_orders=orders,
        inv=snaps[0],
        cfg=cfg,
        sub_problem_key="fei_jia_gong_le_20",
    )

    assert isinstance(result, WindowResult)
    assert result.feasible is True
    assert result.total_visits == 3
    assert len(result.solution.wave_assignments) == 2


def test_simulate_window_filters_stockout_orders():
    """预扫描:全 SKU 全货架缺货订单 → 剔除,不进 MIP。"""
    stockout_order = Order(
        order_id="STOCKOUT",
        timestamp=datetime.datetime(2026, 7, 1, 14, 0),
        wave_type="非加工",
        lines=(OrderLine(sku_id="K_MISSING", qty=1),),
        件数=1,
        size_class="le_20",
        sub_problem_key=("非加工", "le_20"),
    )
    orders = list(load_orders(FIXTURE / "orders.csv")) + [stockout_order]
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)

    result = simulate_window(
        window_orders=orders,
        inv=snaps[0],
        cfg=cfg,
        sub_problem_key="fei_jia_gong_le_20",
    )

    assert "STOCKOUT" in result.rejected_order_ids
    # 剔除后剩 3 单,MIP 正常跑
    assert result.total_visits == 3


# === Task 21: Filter 2/3 拣货员与加工机器利用率 ===


def test_picker_utilization_below_W_is_feasible():
    """3 单 × 120s/单 / 4 picker = 90s < 3600s(1h)→ 利用率 ~2.5%,可行(< 100%)。"""
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)

    result = simulate_window(
        window_orders=orders,
        inv=snaps[0],
        cfg=cfg,
        sub_problem_key="fei_jia_gong_le_20",
        window_start=datetime.datetime(2026, 7, 1, 14, 0),
        W_seconds=3600,
    )

    assert result.feasible
    assert 0.0 < result.picker_utilization < 1.0
    # 3 × 120 / 4 / 3600 = 0.025
    assert abs(result.picker_utilization - 0.025) < 1e-6


def test_picker_utilization_above_W_is_infeasible():
    """W 太小:3 单 × 120s / 4 = 90s > W=10s → 不可行。"""
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)

    result = simulate_window(
        window_orders=orders,
        inv=snaps[0],
        cfg=cfg,
        sub_problem_key="fei_jia_gong_le_20",
        window_start=datetime.datetime(2026, 7, 1, 14, 0),
        W_seconds=10,
    )

    assert not result.feasible
    assert any("picker_overload" in n for n in result.notes)


def test_machine_utilization_only_for_jia_gong():
    """非加工队列 machine_utilization 应为 None。"""
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)

    result = simulate_window(
        window_orders=orders,
        inv=snaps[0],
        cfg=cfg,
        sub_problem_key="fei_jia_gong_le_20",
        window_start=datetime.datetime(2026, 7, 1, 14, 0),
        W_seconds=3600,
    )
    assert result.machine_utilization is None  # 非加工不查机器
