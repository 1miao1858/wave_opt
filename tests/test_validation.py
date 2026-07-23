from pathlib import Path
import dataclasses

import pytest
from src.data_loader import load_orders, load_inventory_snapshots
from src.mip_solver import MIPInput, JointMIPSolver, MIPSolution
from src.validation import recompute_hit_rate, ValidationIssue, check_boundary

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
