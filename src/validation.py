"""验证检查(6.1-6.10)。

每检查函数接收 MIPSolution + 相关上下文,返回 list[ValidationIssue]。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.mip_solver import MIPSolution


@dataclass(frozen=True)
class ValidationIssue:
    check_name: str  # "6.2_recompute_hit_rate" 等
    severity: str  # "error" / "warning"
    message: str
    context: dict  # 例如 {"subwave_idx": 0, "reported": 1.5, "recomputed": 2.0}


def _recompute_hit_rate_from_solution(sol: MIPSolution) -> tuple[float, list[dict]]:
    """从 z_qty(=pick_qty)重算每子波命中率。返回 (avg_hit_rate, per_subwave)。"""
    per_subwave: list[dict] = []
    total_hits = 0
    total_visits = 0
    for w in sol.wave_assignments:
        shelf_sku_hits: dict[str, set[str]] = {}
        for (sku, shelf), q in w.pick_qty.items():
            if q > 0:
                shelf_sku_hits.setdefault(shelf, set()).add(sku)
        hits = sum(len(sk_set) for sk_set in shelf_sku_hits.values())
        visits = len(w.visited_shelves)
        per_subwave.append({
            "subwave_idx": w.subwave_idx,
            "hits": hits,
            "visits": visits,
            "hit_rate": (hits / visits) if visits > 0 else 0.0,
        })
        total_hits += hits
        total_visits += visits
    avg = (total_hits / total_visits) if total_visits > 0 else 0.0
    return avg, per_subwave


def recompute_hit_rate(sol: MIPSolution) -> list[ValidationIssue]:
    """6.2:每子波独立复算命中率,与 MIP 报告对齐。"""
    issues: list[ValidationIssue] = []
    recomputed, _per_subwave = _recompute_hit_rate_from_solution(sol)
    if abs(recomputed - sol.hit_rate) > 1e-6:
        issues.append(
            ValidationIssue(
                check_name="6.2_recompute_hit_rate",
                severity="error",
                message=(
                    f"MIP 报告平均命中率 {sol.hit_rate:.4f} 与复算 {recomputed:.4f} 不符"
                ),
                context={"reported": sol.hit_rate, "recomputed": recomputed},
            )
        )
    return issues


def check_boundary(sol: MIPSolution, max_skus_per_shelf: int) -> list[ValidationIssue]:
    """6.5:每子波命中率应在 [1, max_skus_per_shelf] 区间。"""
    issues: list[ValidationIssue] = []
    _, per_subwave = _recompute_hit_rate_from_solution(sol)
    for sw in per_subwave:
        hr = sw["hit_rate"]
        if hr < 1.0 - 1e-6:
            issues.append(
                ValidationIssue(
                    check_name="6.5_boundary",
                    severity="error",
                    message=f"子波 {sw['subwave_idx']} 命中率 {hr:.4f} < 1(下界违规)",
                    context=sw,
                )
            )
        elif hr > max_skus_per_shelf + 1e-6:
            issues.append(
                ValidationIssue(
                    check_name="6.5_boundary",
                    severity="error",
                    message=(
                        f"子波 {sw['subwave_idx']} 命中率 {hr:.4f} > "
                        f"max_skus_per_shelf={max_skus_per_shelf}(上界违规)"
                    ),
                    context=sw,
                )
            )
    return issues
