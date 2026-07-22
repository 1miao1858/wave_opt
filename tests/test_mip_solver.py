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
    for order in inp.window_orders:
        for line in order.lines:
            picked = sum(
                w.pick_qty.get((line.sku_id, s), 0)
                for w in sol.wave_assignments
                for s in [s2 for (sk, s2) in w.pick_qty.keys() if sk == line.sku_id]
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
