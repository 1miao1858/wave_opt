from pathlib import Path
import pytest
from src.data_loader import load_orders, load_inventory_snapshots
from src.mip_solver import MIPInput, JointMIPSolver

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


# === 方案 A 测试 ===

def test_two_phase_a_c1_fulfillment():
    """每订单每 SKU 必凑齐(用 per_order_pick_qty,避免子波内聚合假阳)。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_a", mip_gap=0.05).solve(inp)
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


def test_two_phase_a_c2_inventory():
    """每 (shelf, sku) 总消耗 ≤ 库存。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_a", mip_gap=0.05).solve(inp)
    for (shelf, sku), consumed in sol.consumption.items():
        avail = inp.inv.inv.get((shelf, sku), 0)
        assert consumed <= avail, (
            f"({shelf}, {sku}) 消耗 {consumed} > 库存 {avail}"
        )


def test_two_phase_a_c4_composition():
    """每订单必进且仅进一个子波。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_a", mip_gap=0.05).solve(inp)
    seen = set()
    for w in sol.wave_assignments:
        for o in w.orders:
            assert o.order_id not in seen, f"订单 {o.order_id} 出现在多个子波"
            seen.add(o.order_id)
    assert len(seen) == len(inp.window_orders), f"覆盖订单 {len(seen)} ≠ 总 {len(inp.window_orders)}"


def test_two_phase_a_c5_capacity():
    """每子波订单数 ≤ N_max。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_a", mip_gap=0.05).solve(inp)
    for w in sol.wave_assignments:
        assert len(w.orders) <= inp.N_max, f"子波 {w.subwave_idx} 订单数 {len(w.orders)} > N_max {inp.N_max}"


def test_two_phase_a_obj_matches_joint_on_tiny_case():
    """tiny_case 上方案 A 的 obj 应等于 joint 最优(3)。"""
    inp = _build_input()
    joint_sol = JointMIPSolver(solver_name="highs", mip_gap=0.05).solve(inp)
    a_sol = JointMIPSolver(solver_name="two_phase_a", mip_gap=0.05).solve(inp)
    assert a_sol.objective_value == joint_sol.objective_value, (
        f"A obj {a_sol.objective_value} ≠ joint obj {joint_sol.objective_value}"
    )


# === 方案 B 测试 ===

def test_two_phase_b_c1_fulfillment():
    """每订单每 SKU 必凑齐。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_b", mip_gap=0.05).solve(inp)
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


def test_two_phase_b_c2_inventory():
    """每 (shelf, sku) 总消耗 ≤ 库存。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_b", mip_gap=0.05).solve(inp)
    for (shelf, sku), consumed in sol.consumption.items():
        avail = inp.inv.inv.get((shelf, sku), 0)
        assert consumed <= avail, f"({shelf}, {sku}) 消耗 {consumed} > 库存 {avail}"


def test_two_phase_b_c4_composition():
    """每订单必进且仅进一个子波。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_b", mip_gap=0.05).solve(inp)
    seen = set()
    for w in sol.wave_assignments:
        for o in w.orders:
            assert o.order_id not in seen, f"订单 {o.order_id} 出现在多个子波"
            seen.add(o.order_id)
    assert len(seen) == len(inp.window_orders)


def test_two_phase_b_c5_capacity():
    """每子波订单数 ≤ N_max。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="two_phase_b", mip_gap=0.05).solve(inp)
    for w in sol.wave_assignments:
        assert len(w.orders) <= inp.N_max, f"子波 {w.subwave_idx} 订单数 {len(w.orders)} > N_max"


def test_two_phase_b_obj_matches_joint_on_tiny_case():
    """tiny_case 上方案 B 的 obj 应等于 joint 最优(3)。"""
    inp = _build_input()
    joint_sol = JointMIPSolver(solver_name="highs", mip_gap=0.05).solve(inp)
    b_sol = JointMIPSolver(solver_name="two_phase_b", mip_gap=0.05).solve(inp)
    assert b_sol.objective_value == joint_sol.objective_value, (
        f"B obj {b_sol.objective_value} ≠ joint obj {joint_sol.objective_value}"
    )


# === greedy_a 测试(两阶段贪心,方案 A 结构)===

def test_greedy_a_c1_fulfillment():
    """每订单每 SKU 必凑齐。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="greedy_a", mip_gap=0.05).solve(inp)
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


def test_greedy_a_c2_inventory():
    """每 (shelf, sku) 总消耗 ≤ 库存。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="greedy_a", mip_gap=0.05).solve(inp)
    for (shelf, sku), consumed in sol.consumption.items():
        avail = inp.inv.inv.get((shelf, sku), 0)
        assert consumed <= avail, f"({shelf}, {sku}) 消耗 {consumed} > 库存 {avail}"


def test_greedy_a_c4_composition():
    """每订单必进且仅进一个子波。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="greedy_a", mip_gap=0.05).solve(inp)
    seen = set()
    for w in sol.wave_assignments:
        for o in w.orders:
            assert o.order_id not in seen, f"订单 {o.order_id} 出现在多个子波"
            seen.add(o.order_id)
    assert len(seen) == len(inp.window_orders)


def test_greedy_a_c5_capacity():
    """每子波订单数 ≤ N_max。"""
    inp = _build_input()
    sol = JointMIPSolver(solver_name="greedy_a", mip_gap=0.05).solve(inp)
    for w in sol.wave_assignments:
        assert len(w.orders) <= inp.N_max, f"子波 {w.subwave_idx} 订单数 {len(w.orders)} > N_max {inp.N_max}"
