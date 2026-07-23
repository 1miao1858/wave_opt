from pathlib import Path
import dataclasses

import pytest
from src.data_loader import load_orders, load_inventory_snapshots
from src.mip_solver import MIPInput, JointMIPSolver, MIPSolution
from src.validation import recompute_hit_rate, ValidationIssue

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
