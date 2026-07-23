"""验证检查(6.1-6.10)。

每检查函数接收 MIPSolution + 相关上下文,返回 list[ValidationIssue]。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import InventorySnapshot, Order
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


def recompute_total_visits(sol: MIPSolution) -> list[ValidationIssue]:
    """6.7:从 z_qty 推 y[s,w],独立复算总访问数。"""
    issues: list[ValidationIssue] = []
    total = 0
    for w in sol.wave_assignments:
        # 从 pick_qty 推 visited_shelves
        derived = {shelf for (sku, shelf), q in w.pick_qty.items() if q > 0}
        # 与 MIP 报告的 visited_shelves 比对
        if derived != set(w.visited_shelves):
            issues.append(
                ValidationIssue(
                    check_name="6.7_recompute_total_visits",
                    severity="error",
                    message=(
                        f"子波 {w.subwave_idx}:从 z_qty 推货架 {sorted(derived)} ≠ "
                        f"MIP 报告 {sorted(w.visited_shelves)}"
                    ),
                    context={
                        "subwave_idx": w.subwave_idx,
                        "derived": sorted(derived),
                        "reported": sorted(w.visited_shelves),
                    },
                )
            )
        total += len(derived)
    if total != sol.total_visits:
        issues.append(
            ValidationIssue(
                check_name="6.7_recompute_total_visits",
                severity="error",
                message=(
                    f"MIP 报告总访问数 {sol.total_visits} 与复算 {total} 不符"
                ),
                context={"reported": sol.total_visits, "recomputed": total},
            )
        )
    return issues


def check_subwave_internal_split(
    sol: MIPSolution, inv: InventorySnapshot
) -> list[ValidationIssue]:
    """6.8a:同 SKU 在同子波内被拆到多货架——核对是否有库存必要性。"""
    issues: list[ValidationIssue] = []
    for w in sol.wave_assignments:
        # 按 SKU 聚合该子波用了哪些货架
        sku_to_shelves: dict[str, set[str]] = {}
        sku_to_qty: dict[str, int] = {}
        for (sku, shelf), q in w.pick_qty.items():
            if q > 0:
                sku_to_shelves.setdefault(sku, set()).add(shelf)
                sku_to_qty[sku] = sku_to_qty.get(sku, 0) + q
        for sku, shelves in sku_to_shelves.items():
            if len(shelves) <= 1:
                continue
            # 拆了——核对单货架库存是否够
            total_needed = sku_to_qty[sku]
            max_single = max(inv.inv.get((s, sku), 0) for s in shelves)
            if max_single >= total_needed:
                issues.append(
                    ValidationIssue(
                        check_name="6.8a_subwave_internal_split",
                        severity="warning",
                        message=(
                            f"子波 {w.subwave_idx} 的 SKU {sku} 拆到 {sorted(shelves)} "
                            f"拣,但单货架库存够({max_single} ≥ {total_needed})——无意义拆分"
                        ),
                        context={
                            "subwave_idx": w.subwave_idx,
                            "sku": sku,
                            "shelves": sorted(shelves),
                            "total_needed": total_needed,
                            "max_single_shelf": max_single,
                        },
                    )
                )
    return issues


def check_cross_subwave_split(
    sol: MIPSolution, N_max: int
) -> list[ValidationIssue]:
    """6.8b:同 SKU 被拆到多个子波拣,但 total_orders ≤ N_max(本可同子波)。"""
    issues: list[ValidationIssue] = []
    # 每个 SKU 的 total_orders(从订单行算,不依赖子波分配)
    sku_to_orders: dict[str, set[str]] = {}
    sku_to_subwaves: dict[str, set[int]] = {}
    for w in sol.wave_assignments:
        for o in w.orders:
            for line in o.lines:
                sku_to_orders.setdefault(line.sku_id, set()).add(o.order_id)
        for (sku, shelf), q in w.pick_qty.items():
            if q > 0:
                sku_to_subwaves.setdefault(sku, set()).add(w.subwave_idx)

    for sku, subwaves in sku_to_subwaves.items():
        if len(subwaves) <= 1:
            continue
        total_orders = len(sku_to_orders.get(sku, set()))
        if total_orders <= N_max:
            issues.append(
                ValidationIssue(
                    check_name="6.8b_cross_subwave_split",
                    severity="warning",
                    message=(
                        f"SKU {sku} 拆到 {sorted(subwaves)} 个子波拣,"
                        f"但 total_orders={total_orders} ≤ N_max={N_max}"
                        f"(本可同子波)"
                    ),
                    context={
                        "sku": sku,
                        "subwaves": sorted(subwaves),
                        "total_orders": total_orders,
                        "N_max": N_max,
                    },
                )
            )
    return issues


def check_inventory_consistency(
    inv_before: InventorySnapshot,
    inv_after: InventorySnapshot,
    consumption: dict[tuple[str, str], int],
) -> list[ValidationIssue]:
    """6.3:inv_after_computed = inv_before - consumption;与真实 inv_after 对比。

    补货会让 inv_after > inv_after_computed(预期);若 inv_after < inv_after_computed,
    一定有 bug(消耗算错或库存快照错位)。
    """
    issues: list[ValidationIssue] = []
    for (shelf, sku), consumed in consumption.items():
        before = inv_before.inv.get((shelf, sku), 0)
        after_real = inv_after.inv.get((shelf, sku), 0)
        after_computed = before - consumed
        if after_real < after_computed:
            issues.append(
                ValidationIssue(
                    check_name="6.3_inventory_consistency",
                    severity="error",
                    message=(
                        f"({shelf},{sku}):before={before},消耗={consumed},"
                        f"computed_after={after_computed},但真实 next={after_real}"
                        f"(next < computed,可能消耗算错或库存快照错位)"
                    ),
                    context={
                        "shelf": shelf,
                        "sku": sku,
                        "before": before,
                        "consumed": consumed,
                        "after_computed": after_computed,
                        "after_real": after_real,
                    },
                )
            )
    return issues


def audit_stockout_filter(
    rejected_orders: list[Order], inv: InventorySnapshot
) -> list[ValidationIssue]:
    """6.4:核对被剔除订单是否真的全货架缺货(任一 SKU 不缺 → 报 issue)。"""
    issues: list[ValidationIssue] = []
    for order in rejected_orders:
        for line in order.lines:
            shelves = inv.sku_shelves.get(line.sku_id, set())
            total_avail = sum(inv.inv.get((s, line.sku_id), 0) for s in shelves)
            if total_avail >= line.qty:
                issues.append(
                    ValidationIssue(
                        check_name="6.4_stockout_filter",
                        severity="error",
                        message=(
                            f"订单 {order.order_id} 被剔除,但 SKU {line.sku_id} "
                            f"全货架库存 {total_avail} ≥ 需求 {line.qty}"
                            f"(预扫描逻辑可能误剔)"
                        ),
                        context={
                            "order_id": order.order_id,
                            "sku": line.sku_id,
                            "needed": line.qty,
                            "total_avail": total_avail,
                        },
                    )
                )
                break  # 一个 SKU 不缺就够了,不必继续查
    return issues


def check_mip_gap(sol: MIPSolution, threshold: float) -> list[ValidationIssue]:
    """6.9:MIP gap 应 ≤ threshold(POC 5%)。"""
    issues: list[ValidationIssue] = []
    if sol.mip_gap is not None and sol.mip_gap > threshold:
        issues.append(
            ValidationIssue(
                check_name="6.9_mip_gap",
                severity="warning",
                message=(
                    f"MIP gap {sol.mip_gap:.4f} > 阈值 {threshold:.4f}"
                    f"(求解器未收敛到 POC 标准)"
                ),
                context={"mip_gap": sol.mip_gap, "threshold": threshold},
            )
        )
    return issues
