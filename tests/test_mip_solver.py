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
