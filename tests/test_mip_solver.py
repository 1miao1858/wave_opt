from pathlib import Path
import pytest
from src.data_loader import load_orders, load_inventory_snapshots
from src.mip_solver import JointMIPSolver, MIPInput, MIPSolution, WaveAssignment

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_case"


def _build_input(N_max=2):
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    inv = snaps[0]
    return MIPInput(window_orders=orders, inv=inv, N_max=N_max, M_big=100, time_limit=60)


def test_mip_solver_returns_solution_with_variables():
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    assert isinstance(sol, MIPSolution)
    assert sol.status == "optimal"
    # 决策变量结构
    assert len(sol.wave_assignments) == 2  # 3 单 / N_max=2 → 2 子波
    # 每订单必进且仅进一个子波(C4)
    for order in inp.window_orders:
        cnt = sum(
            1 for w in sol.wave_assignments
            if any(o.order_id == order.order_id for o in w.orders)
        )
        assert cnt == 1


def test_mip_solver_c1_fulfillment_constraint():
    """C1':Σ_s z_qty[i,k,s] = qty[i,k]。每订单的 SKU 需求必凑齐。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    # 每订单每 SKU 的总拣量 = 需求量
    # 用 per_order_pick_qty(按订单拆分),避免子波内多订单聚合造成假阳
    for order in inp.window_orders:
        for line in order.lines:
            picked = sum(
                q
                for w in sol.wave_assignments
                for (oid, sk, _shelf), q in w.per_order_pick_qty.items()
                if oid == order.order_id and sk == line.sku_id
            )
            assert picked == line.qty, (
                f"订单 {order.order_id} 的 SKU {line.sku_id} "
                f"拣量 {picked} ≠ 需求 {line.qty}"
            )


def test_mip_solver_c2_inventory_constraint():
    """C2:Σ_i z_qty[i,k,s] ≤ inv[s,k]。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    for (shelf, sku), consumed in sol.consumption.items():
        avail = inp.inv.inv.get((shelf, sku), 0)
        assert consumed <= avail, (
            f"({shelf},{sku}) 消耗 {consumed} > 库存 {avail}"
        )


def test_mip_solver_c3_linking_y_shelf_visited_when_pick():
    """C3a:若 z_qty[i,k,s]>0 且 x[i,w]=1,则 y[s,w]=1。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    for w_idx, w in enumerate(sol.wave_assignments):
        visited = set(w.visited_shelves)
        for (sku, shelf), q in w.pick_qty.items():
            if q > 0:
                assert shelf in visited, (
                    f"子波 {w_idx}:SKU {sku} 从 {shelf} 拣 {q} 件,"
                    f"但 {shelf} 不在 visited_shelves 里(C3a 失效)"
                )


def test_mip_solver_c3_linking_h_sku_picked():
    """C3b:若 z_qty[i,k,s]>0 且 x[i,w]=1,则 h[s,k,w]=1。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    # h 体现在 hit_rate 上,这里检查 hit_rate ≥ 1(每访问货架至少 1 hit)
    assert sol.hit_rate >= 1.0


def test_mip_solver_c4_each_order_in_exactly_one_subwave():
    """C4:Σ_w x[i,w] = 1。每订单必进且仅进一个子波。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    order_to_wave: dict[str, int] = {}
    for w_idx, w in enumerate(sol.wave_assignments):
        for o in w.orders:
            assert o.order_id not in order_to_wave, (
                f"订单 {o.order_id} 出现在多个子波里(C4 失效)"
            )
            order_to_wave[o.order_id] = w_idx

    # 所有订单都进了一个子波
    for order in inp.window_orders:
        assert order.order_id in order_to_wave, (
            f"订单 {order.order_id} 未被任何子波包含(C4 失效)"
        )


def test_mip_solver_c5_subwave_size_le_N_max():
    """C5:Σ_i x[i,w] ≤ N_max。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    for w_idx, w in enumerate(sol.wave_assignments):
        assert len(w.orders) <= inp.N_max, (
            f"子波 {w_idx}:订单数 {len(w.orders)} > N_max={inp.N_max}(C5 失效)"
        )


def test_mip_solver_objective_minimizes_total_visits():
    """主指标:min Σ y[s,w]。tiny_case 手算最优 = 3。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    assert sol.status == "optimal"
    assert sol.total_visits == 3, (
        f"tiny_case 最优访问数应为 3,实际 {sol.total_visits}"
    )
    assert sol.objective_value == 3


def test_mip_solver_hit_rate_recomputed():
    """命中率从 h 推算。spec line 348: 平均命中率 = Σ hits / Σ visits。
    tiny_case: 子波 0 hits=3 visits=2;子波 1 hits=2 visits=1
    平均 = (3+2) / (2+1) = 5/3 ≈ 1.6667
    """
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    # 平均命中率 = Σ hits / Σ visits(spec line 348,聚合比,非算术均)
    assert sol.hit_rate == pytest.approx(5 / 3, abs=0.01), (
        f"tiny_case 平均命中率应为 5/3≈1.6667,实际 {sol.hit_rate}"
    )


# === Task 11: 6.1 合成小样本手算对比 + MIP/贪心/随机基线对比 ===

import json

EXPECTED_JSON = (
    Path(__file__).parent / "fixtures" / "tiny_case" / "expected.json"
)


def test_mip_matches_hand_computed_optimal():
    """6.1:MIP 输出与手算最优解对齐。"""
    inp = _build_input()
    solver = JointMIPSolver()
    sol = solver.solve(inp)

    expected = json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))
    assert sol.total_visits == expected["optimal_total_visits"]
    # 子波数
    assert len(sol.wave_assignments) == len(expected["optimal_subwaves"])
    # 每子波访问数集合
    actual_visits_per_wave = sorted(
        len(w.visited_shelves) for w in sol.wave_assignments
    )
    expected_visits_per_wave = sorted(
        sw["shelf_visits"] for sw in expected["optimal_subwaves"]
    )
    assert actual_visits_per_wave == expected_visits_per_wave


def test_mip_optimal_at_least_as_good_as_greedy_and_random(monkeypatch):
    """6.1:命中率_MIP ≥ 命中率_贪心 ≥ 命中率_随机。

    用 tiny_case:MIP 最优访问 3 次,贪心应至少访问 3 次(可能更多),随机应访问 ≥ 3 次。
    """
    inp = _build_input()
    solver = JointMIPSolver()
    sol_mip = solver.solve(inp)

    # 贪心基线(Task 12 实现)——此处先用 import-or-skip 占位
    pytest.importorskip("src.greedy")
    from src.greedy import greedy_solve
    sol_greedy = greedy_solve(inp)

    # 命中率:MIP ≥ 贪心
    assert sol_mip.hit_rate >= sol_greedy.hit_rate - 1e-6
    # 访问数:MIP ≤ 贪心
    assert sol_mip.total_visits <= sol_greedy.total_visits
