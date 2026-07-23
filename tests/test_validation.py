from pathlib import Path
import dataclasses
import datetime

import pytest
from src.data_loader import (
    load_orders,
    load_inventory_snapshots,
    Order,
    OrderLine,
    InventorySnapshot,
)
from src.mip_solver import MIPInput, JointMIPSolver, MIPSolution, WaveAssignment
from src.validation import (
    recompute_hit_rate,
    ValidationIssue,
    check_boundary,
    recompute_total_visits,
    check_subwave_internal_split,
    check_cross_subwave_split,
    check_inventory_consistency,
    audit_stockout_filter,
    check_mip_gap,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_case"


def _build_solution():
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    inp = MIPInput(
        window_orders=tuple(orders),
        inv=snaps[0],
        N_max=2,
        M_big=100,
        time_limit=60,
    )
    return JointMIPSolver().solve(inp), inp


def test_recompute_hit_rate_matches_mip_report():
    sol, _ = _build_solution()
    issues = recompute_hit_rate(sol)
    assert issues == [], f"发现命中率不一致:{issues}"


def test_recompute_hit_rate_detects_mismatch():
    sol, _ = _build_solution()
    bad_sol = dataclasses.replace(sol, hit_rate=sol.hit_rate + 1.0)
    issues = recompute_hit_rate(bad_sol)
    assert len(issues) >= 1
    assert issues[0].check_name == "6.2_recompute_hit_rate"


# === Task 14: 6.5 边界合理性检查 ===


def test_boundary_check_passes_for_normal_solution():
    sol, _ = _build_solution()
    # 货架上 SKU 种类数:tiny_case S1 有 K1,K2;S2 有 K2,K3 → max = 2
    issues = check_boundary(sol, max_skus_per_shelf=2)
    assert issues == []


def test_boundary_check_detects_hit_rate_below_one():
    sol, _ = _build_solution()
    # 篡改:visited_shelves 多加 2 个空货架,使 hits/visits < 1
    # (tiny_case 子波 0 原始 hits=3 visits=2 → 1.5;加 2 空货架后 3/4=0.75 < 1)
    bad_wave = dataclasses.replace(
        sol.wave_assignments[0],
        visited_shelves=sol.wave_assignments[0].visited_shelves + ("S_FAKE1", "S_FAKE2"),
    )
    bad_sol = dataclasses.replace(
        sol, wave_assignments=(bad_wave,) + sol.wave_assignments[1:]
    )
    issues = check_boundary(bad_sol, max_skus_per_shelf=2)
    assert any(i.check_name == "6.5_boundary" and i.severity == "error" for i in issues)


# === Task 15: 6.7 总访问数独立复算 ===


def test_recompute_total_visits_matches_mip():
    sol, _ = _build_solution()
    issues = recompute_total_visits(sol)
    assert issues == []


def test_recompute_total_visits_detects_mismatch():
    sol, _ = _build_solution()
    bad_sol = dataclasses.replace(sol, total_visits=sol.total_visits + 1)
    issues = recompute_total_visits(bad_sol)
    assert len(issues) >= 1
    assert issues[0].check_name == "6.7_recompute_total_visits"


# === Task 16: 6.8a 子波内拆分检查 ===


def _inv_for_test(sol):
    """从 sol 构造一个最小 inv(只含涉及的 shelf, sku)。"""
    inv_dict = {}
    for (shelf, sku), consumed in sol.consumption.items():
        inv_dict[(shelf, sku)] = consumed + 10  # 给点余量
    shelf_skus: dict[str, set[str]] = {}
    sku_shelves: dict[str, set[str]] = {}
    for (shelf, sku), q in inv_dict.items():
        if q > 0:
            shelf_skus.setdefault(shelf, set()).add(sku)
            sku_shelves.setdefault(sku, set()).add(shelf)
    return InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 14, 0),
        inv=inv_dict,
        shelf_skus=shelf_skus,
        sku_shelves=sku_shelves,
    )


def test_subwave_internal_split_no_issue_when_single_shelf():
    sol, _ = _build_solution()
    inv = _inv_for_test(sol)
    issues = check_subwave_internal_split(sol, inv)
    # tiny_case 最优解应无无意义拆分(MIP 不会做)
    assert issues == []


def test_subwave_internal_split_detects_when_present():
    """构造一个故意拆分的解:子波 0 把 SKU K2 拆到 S1 和 S2 拣,但单货架库存就够。"""
    order = Order(
        order_id="X",
        timestamp=datetime.datetime(2026, 7, 1, 14, 0),
        wave_type="非加工",
        lines=(OrderLine(sku_id="K2", qty=2),),
        件数=2,
        size_class="le_20",
        sub_problem_key=("非加工", "le_20"),
    )
    wave = WaveAssignment(
        subwave_idx=0,
        orders=(order,),
        visited_shelves=("S1", "S2"),
        pick_qty={("K2", "S1"): 1, ("K2", "S2"): 1},  # 故意拆分
        per_order_pick_qty={("X", "K2", "S1"): 1, ("X", "K2", "S2"): 1},
    )
    sol = MIPSolution(
        status="mock",
        wave_assignments=(wave,),
        total_visits=2,
        hit_rate=1.0,
        consumption={("S1", "K2"): 1, ("S2", "K2"): 1},
        objective_value=2.0,
        solver_name="mock",
    )
    inv = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 14, 0),
        inv={("S1", "K2"): 5, ("S2", "K2"): 5},
        shelf_skus={"S1": {"K2"}, "S2": {"K2"}},
        sku_shelves={"K2": {"S1", "S2"}},
    )
    issues = check_subwave_internal_split(sol, inv)
    assert any(
        i.check_name == "6.8a_subwave_internal_split"
        and i.context["sku"] == "K2"
        for i in issues
    )


# === Task 17: 6.8b 跨子波拆分检查 ===


def test_cross_subwave_split_fires_for_tiny_case():
    sol, _ = _build_solution()
    issues = check_cross_subwave_split(sol, N_max=2)
    # tiny_case 最优 {A,C}+{B}:K2/K3 各跨 2 子波且 total_orders=2 ≤ N_max
    # 或 {B,C}+{A}:K1/K2 各跨 2 子波且 total_orders=2 ≤ N_max
    # 任一最优至少 2 个 SKU 触发 6.8b
    assert len(issues) >= 1, f"tiny_case 最优应触发 6.8b,实际 issues={issues}"
    assert all(i.check_name == "6.8b_cross_subwave_split" for i in issues)


def test_cross_subwave_split_detects_when_present():
    """构造:2 子波各 1 订单,同 SKU 拆到 2 子波,但 total_orders ≤ N_max。"""
    order1 = Order(
        order_id="X1",
        timestamp=datetime.datetime(2026, 7, 1, 14, 0),
        wave_type="非加工",
        lines=(OrderLine(sku_id="K2", qty=1),),
        件数=1,
        size_class="le_20",
        sub_problem_key=("非加工", "le_20"),
    )
    order2 = Order(
        order_id="X2",
        timestamp=datetime.datetime(2026, 7, 1, 14, 1),
        wave_type="非加工",
        lines=(OrderLine(sku_id="K2", qty=1),),
        件数=1,
        size_class="le_20",
        sub_problem_key=("非加工", "le_20"),
    )
    sol = MIPSolution(
        status="mock",
        wave_assignments=(
            WaveAssignment(
                subwave_idx=0,
                orders=(order1,),
                visited_shelves=("S1",),
                pick_qty={("K2", "S1"): 1},
                per_order_pick_qty={("X1", "K2", "S1"): 1},
            ),
            WaveAssignment(
                subwave_idx=1,
                orders=(order2,),
                visited_shelves=("S1",),
                pick_qty={("K2", "S1"): 1},
                per_order_pick_qty={("X2", "K2", "S1"): 1},
            ),
        ),
        total_visits=2,
        hit_rate=1.0,
        consumption={("S1", "K2"): 2},
        objective_value=2.0,
        solver_name="mock",
    )
    issues = check_cross_subwave_split(sol, N_max=2)
    # K2 在 2 个子波,total_orders_using_k = 2 ≤ N_max=2 → 应报
    assert any(
        i.check_name == "6.8b_cross_subwave_split"
        and i.context["sku"] == "K2"
        for i in issues
    )


# === Task 18: 6.3 库存一致性抽样检查 ===


def test_inventory_consistency_ok_when_next_snapshot_higher():
    """补货会让 next > computed;不报错。"""
    inv_before = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 14, 0),
        inv={("S1", "K1"): 5, ("S2", "K2"): 5},
        shelf_skus={"S1": {"K1"}, "S2": {"K2"}},
        sku_shelves={"K1": {"S1"}, "K2": {"S2"}},
    )
    inv_after = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 15, 0),
        inv={("S1", "K1"): 7, ("S2", "K2"): 4},  # S1 补货 2,S2 减 1
        shelf_skus={"S1": {"K1"}, "S2": {"K2"}},
        sku_shelves={"K1": {"S1"}, "K2": {"S2"}},
    )
    # 消耗:S1 K1 消耗 2,S2 K2 消耗 3
    consumption = {("S1", "K1"): 2, ("S2", "K2"): 3}
    issues = check_inventory_consistency(inv_before, inv_after, consumption)
    # S1:5 - 2 = 3 < 7(补货,OK);S2:5 - 3 = 2 < 4(补货 2,OK)
    assert issues == []


def test_inventory_consistency_detects_when_next_lower_than_computed():
    """若 next < inv_after_computed,一定有 bug。"""
    inv_before = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 14, 0),
        inv={("S1", "K1"): 10},
        shelf_skus={"S1": {"K1"}}, sku_shelves={"K1": {"S1"}},
    )
    inv_after = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 15, 0),
        inv={("S1", "K1"): 2},  # before 10 - 消耗 3 = 7,但 next = 2 < 7
        shelf_skus={"S1": {"K1"}}, sku_shelves={"K1": {"S1"}},
    )
    consumption = {("S1", "K1"): 3}
    issues = check_inventory_consistency(inv_before, inv_after, consumption)
    assert any(i.check_name == "6.3_inventory_consistency" for i in issues)


# === Task 19: 6.4 缺货剔除核对 + 6.9 MIP gap 监控 ===


def _build_order(order_id: str, sku_id: str, qty: int) -> Order:
    return Order(
        order_id=order_id,
        timestamp=datetime.datetime(2026, 7, 1, 14, 0),
        wave_type="非加工",
        lines=(OrderLine(sku_id=sku_id, qty=qty),),
        件数=qty,
        size_class="le_20",
        sub_problem_key=("非加工", "le_20"),
    )


def test_stockout_filter_audit_flags_correctly_rejected():
    """订单全 SKU 跨所有货架缺货 → 应被剔除(无 issue)。"""
    inv = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 14, 0),
        inv={("S1", "K1"): 0, ("S2", "K1"): 0},  # K1 全 0
        shelf_skus={},
        sku_shelves={"K1": {"S1", "S2"}},
    )
    order = _build_order("X", "K1", 1)
    issues = audit_stockout_filter(rejected_orders=[order], inv=inv)
    # 正确剔除(全 0 库存)→ 无 issue
    assert issues == []


def test_stockout_filter_audit_flags_incorrectly_rejected():
    """订单某 SKU 在某货架有库存,却被剔除 → 报 issue。"""
    inv = InventorySnapshot(
        snapshot_time=datetime.datetime(2026, 7, 1, 14, 0),
        inv={("S1", "K1"): 5},  # K1 在 S1 有库存
        shelf_skus={"S1": {"K1"}},
        sku_shelves={"K1": {"S1"}},
    )
    order = _build_order("X", "K1", 1)
    issues = audit_stockout_filter(rejected_orders=[order], inv=inv)
    assert any(i.check_name == "6.4_stockout_filter" for i in issues)


def test_mip_gap_ok_when_below_threshold():
    sol = MIPSolution(
        status="optimal",
        wave_assignments=(),
        total_visits=0,
        hit_rate=0.0,
        consumption={},
        objective_value=0.0,
        mip_gap=0.03,
        solver_name="gurobi",
    )
    issues = check_mip_gap(sol, threshold=0.05)
    assert issues == []


def test_mip_gap_warns_when_above_threshold():
    sol = MIPSolution(
        status="time_limit",
        wave_assignments=(),
        total_visits=0,
        hit_rate=0.0,
        consumption={},
        objective_value=0.0,
        mip_gap=0.08,
        solver_name="gurobi",
    )
    issues = check_mip_gap(sol, threshold=0.05)
    assert any(i.check_name == "6.9_mip_gap" for i in issues)
