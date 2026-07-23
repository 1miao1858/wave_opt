"""频率扫描主循环 + simulate_window + Filter 1-4。

simulate_window:对单窗口跑一次 Joint MIP,产 WindowResult。
SweepRunner:4 子问题 × 3 W × 多日窗口主循环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.config import Config
from src.data_loader import InventorySnapshot, Order
from src.mip_solver import MIPInput, JointMIPSolver, MIPSolution
from src.greedy import greedy_solve


@dataclass(frozen=True)
class WindowResult:
    window_start: object  # datetime
    sub_problem_key: str
    feasible: bool
    solution: MIPSolution | None
    total_visits: int
    picker_utilization: float  # 0-1
    machine_utilization: float | None
    rejected_order_ids: tuple[str, ...]
    reject_reasons: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _prefilter_stockout(
    orders: list[Order], inv: InventorySnapshot
) -> tuple[list[Order], list[tuple[str, str]]]:
    """剔除任一 SKU 全货架缺货的订单。返回 (kept, [(order_id, reason)])。"""
    kept: list[Order] = []
    rejected: list[tuple[str, str]] = []
    for o in orders:
        stockout = False
        for line in o.lines:
            shelves = inv.sku_shelves.get(line.sku_id, set())
            total = sum(inv.inv.get((s, line.sku_id), 0) for s in shelves)
            if total < line.qty:
                stockout = True
                rejected.append(
                    (o.order_id, f"SKU {line.sku_id} 全货架缺货({total}<{line.qty})")
                )
                break
        if not stockout:
            kept.append(o)
    return kept, rejected


def simulate_window(
    window_orders: list[Order],
    inv: InventorySnapshot,
    cfg: Config,
    sub_problem_key: str,
    window_start=None,
    W_seconds: int | None = None,
) -> WindowResult:
    """对单窗口跑一次 Joint MIP + 产能过滤(Filter 2/3)。"""
    # 预扫描缺货剔除
    kept_orders, rejected = _prefilter_stockout(window_orders, inv)
    rejected_ids = tuple(oid for oid, _ in rejected)
    reject_reasons = tuple(r for _, r in rejected)

    if not kept_orders:
        return WindowResult(
            window_start=window_start,
            sub_problem_key=sub_problem_key,
            feasible=False,
            solution=None,
            total_visits=0,
            picker_utilization=0.0,
            machine_utilization=None,
            rejected_order_ids=rejected_ids,
            reject_reasons=reject_reasons,
            notes=("empty_after_prefilter",),
        )

    sub_cfg = cfg.sub_problems[sub_problem_key]

    # === Filter 2:拣货员利用率 Σ 单单单耗 / num_pickers ≤ W ===
    total_pick_time = sum(
        cfg.sweep.pick_time_per_order for _ in kept_orders
    )  # 简化:每单固定单耗;若需按订单件数缩放,后续可加
    if W_seconds is None:
        W_seconds = 3600  # 默认 1h
    picker_util = total_pick_time / cfg.sweep.num_pickers / W_seconds

    notes: list[str] = []
    if picker_util > 1.0 + 1e-6:
        return WindowResult(
            window_start=window_start,
            sub_problem_key=sub_problem_key,
            feasible=False,
            solution=None,
            total_visits=0,
            picker_utilization=picker_util,
            machine_utilization=None,
            rejected_order_ids=rejected_ids,
            reject_reasons=reject_reasons,
            notes=(f"picker_overload_{picker_util:.4f}",),
        )

    # === Filter 3:加工机器利用率(仅加工队列)===
    machine_util: float | None = None
    if sub_cfg.wave_type == "加工":
        total_jian = sum(o.件数 for o in kept_orders)
        machine_util = (
            total_jian * cfg.sweep.process_time_per_jian
            / cfg.sweep.num_machines
            / W_seconds
        )
        if machine_util > 1.0 + 1e-6:
            return WindowResult(
                window_start=window_start,
                sub_problem_key=sub_problem_key,
                feasible=False,
                solution=None,
                total_visits=0,
                picker_utilization=picker_util,
                machine_utilization=machine_util,
                rejected_order_ids=rejected_ids,
                reject_reasons=reject_reasons,
                notes=(f"machine_overload_{machine_util:.4f}",),
            )

    # === Joint MIP ===
    inp = MIPInput(
        window_orders=tuple(kept_orders),
        inv=inv,
        N_max=sub_cfg.N_max,
        M_big=100,
        time_limit=cfg.mip.time_limit,
    )
    solver = JointMIPSolver(
        solver_name=cfg.mip.solver, mip_gap=cfg.mip.mip_gap
    )
    sol = solver.solve(inp)

    # 超时或不可行 → fallback 贪心
    if sol.status in ("time_limit", "infeasible"):
        if cfg.mip.fallback == "greedy":
            prev_status = sol.status
            sol = greedy_solve(inp)
            notes.append(f"fallback_greedy_due_to_{prev_status}")
        else:
            return WindowResult(
                window_start=window_start,
                sub_problem_key=sub_problem_key,
                feasible=False,
                solution=None,
                total_visits=0,
                picker_utilization=picker_util,
                machine_utilization=machine_util,
                rejected_order_ids=rejected_ids,
                reject_reasons=reject_reasons,
                notes=tuple(notes + [f"mip_status_{sol.status}"]),
            )

    return WindowResult(
        window_start=window_start,
        sub_problem_key=sub_problem_key,
        feasible=True,
        solution=sol,
        total_visits=sol.total_visits,
        picker_utilization=picker_util,
        machine_utilization=machine_util,
        rejected_order_ids=rejected_ids,
        reject_reasons=reject_reasons,
        notes=tuple(notes),
    )
