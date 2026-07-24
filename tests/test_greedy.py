from pathlib import Path
import pytest
from src.data_loader import load_orders, load_inventory_snapshots
from src.mip_solver import MIPInput
from src.greedy import greedy_solve

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_case"


def _build_input(N_max=2):
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    return MIPInput(
        window_orders=tuple(orders),
        inv=snaps[0],
        N_max=N_max,
        time_limit=60,
    )


def test_greedy_returns_valid_solution():
    inp = _build_input()
    sol = greedy_solve(inp)

    assert sol.status == "fallback_greedy" or sol.status == "greedy"
    assert sol.total_visits >= 3  # 不劣于最优
    # C4:每订单必进且仅进一个子波
    seen = set()
    for w in sol.wave_assignments:
        for o in w.orders:
            assert o.order_id not in seen
            seen.add(o.order_id)
    assert len(seen) == len(inp.window_orders)
    # C5:每子波 ≤ N_max
    for w in sol.wave_assignments:
        assert len(w.orders) <= inp.N_max


def test_greedy_c1_fulfillment():
    """每订单每 SKU 必凑齐(用 per_order_pick_qty,避免子波内聚合假阳)。"""
    inp = _build_input()
    sol = greedy_solve(inp)
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


def test_greedy_c2_inventory():
    inp = _build_input()
    sol = greedy_solve(inp)
    for (shelf, sku), consumed in sol.consumption.items():
        avail = inp.inv.inv.get((shelf, sku), 0)
        assert consumed <= avail
