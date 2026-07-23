"""频率扫描主循环 + simulate_window + Filter 1-4。

simulate_window:对单窗口跑一次 Joint MIP,产 WindowResult。
SweepRunner:4 子问题 × 3 W × 多日窗口主循环。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


MAX_SUBWAVES = 10  # Filter 1 阈值


def _subwave_count(n_orders: int, N_max: int) -> int:
    return max(1, (n_orders + N_max - 1) // N_max)


def check_filter1(n_orders: int, N_max: int) -> tuple[bool, str]:
    """Filter 1:|W_sub| = ceil(n/N_max),> 10 不可行。"""
    n_sub = _subwave_count(n_orders, N_max)
    if n_sub > MAX_SUBWAVES:
        return False, f"subwave_count_too_large_{n_sub}_>{MAX_SUBWAVES}"
    return True, ""


def check_daily_deadline(
    total_daily_orders: int, cfg: Config, daily_available_seconds: int
) -> tuple[bool, str]:
    """Filter 4:全日订单量 × 单单单耗 ≤ daily_available_seconds × num_pickers。"""
    total_pick_seconds = total_daily_orders * cfg.sweep.pick_time_per_order
    capacity_seconds = daily_available_seconds * cfg.sweep.num_pickers
    if total_pick_seconds > capacity_seconds:
        return False, (
            f"daily_deadline_overflow: "
            f"need {total_pick_seconds}s > capacity {capacity_seconds}s"
        )
    return True, ""


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

    # === Filter 1:MIP sizing(子波数 ≤ 10)===
    ok1, reason1 = check_filter1(len(kept_orders), sub_cfg.N_max)
    if not ok1:
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
            notes=(reason1,),
        )

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


@dataclass(frozen=True)
class SweepResult:
    sub_problem_key: str
    best_W: str | None
    best_total_visits: int | None
    # 每 W 的聚合
    window_results_by_W: dict[str, list[WindowResult]]  # W -> [per-window results]

    @property
    def avg_visits_by_W(self) -> dict[str, float]:
        return {
            w: (sum(r.total_visits for r in rs) / len(rs) if rs else 0.0)
            for w, rs in self.window_results_by_W.items()
        }


def _parse_W(W_str: str) -> int:
    """'1h' -> 3600, '2h' -> 7200, '4h' -> 14400。"""
    if W_str.endswith("h"):
        return int(W_str[:-1]) * 3600
    raise ValueError(f"不支持的 W 格式:{W_str}(仅支持 'Nh')")


def _windows_for_day(W_seconds: int, day_start: datetime, day_end: datetime):
    """生成 [day_start, day_end) 内的 W 窗口起始时刻列表。"""
    windows = []
    t = day_start
    while t < day_end:
        windows.append(t)
        t = t + timedelta(seconds=W_seconds)
    return windows


class SweepRunner:
    """4 子问题 × 3 W × 多日窗口主循环。"""

    def __init__(
        self,
        cfg: Config,
        all_orders: list[Order],
        snapshots: list[InventorySnapshot],
    ):
        self.cfg = cfg
        self.all_orders = all_orders
        self.snapshots = sorted(snapshots, key=lambda s: s.snapshot_time)

    def _orders_in_window(
        self, window_start: datetime, W_seconds: int, sub_problem_key: str
    ) -> list[Order]:
        end = window_start + timedelta(seconds=W_seconds)
        sub_cfg = self.cfg.sub_problems[sub_problem_key]
        return [
            o
            for o in self.all_orders
            if o.sub_problem_key == (sub_cfg.wave_type, sub_cfg.size_class)
            and window_start <= o.timestamp < end
        ]

    def run_sub_problem(self, sub_problem_key: str) -> SweepResult:
        results_by_W: dict[str, list[WindowResult]] = {
            W: [] for W in self.cfg.candidate_W
        }

        if not self.all_orders:
            return SweepResult(
                sub_problem_key=sub_problem_key,
                best_W=None,
                best_total_visits=None,
                window_results_by_W=results_by_W,
            )

        min_day = min(o.timestamp.date() for o in self.all_orders)
        max_day = max(o.timestamp.date() for o in self.all_orders)
        day = min_day
        while day <= max_day:
            # 每日可用工时(简化:从 picker_shift_start 到 picker_shift_end)
            day_start = datetime.combine(
                day,
                datetime.strptime(self.cfg.sweep.picker_shift_start, "%H:%M").time(),
            )
            day_end = datetime.combine(
                day,
                datetime.strptime(self.cfg.sweep.picker_shift_end, "%H:%M").time(),
            )
            daily_available_seconds = (day_end - day_start).total_seconds()

            # Filter 4:每日时效
            sub_cfg = self.cfg.sub_problems[sub_problem_key]
            day_orders_count = sum(
                1
                for o in self.all_orders
                if o.sub_problem_key == (sub_cfg.wave_type, sub_cfg.size_class)
                and o.timestamp.date() == day
            )
            ok4, reason4 = check_daily_deadline(
                day_orders_count, self.cfg, int(daily_available_seconds)
            )
            if not ok4:
                # 全日不可行,跳过所有 W(记到结果里)
                for W in self.cfg.candidate_W:
                    fake_result = WindowResult(
                        window_start=day_start,
                        sub_problem_key=sub_problem_key,
                        feasible=False,
                        solution=None,
                        total_visits=0,
                        picker_utilization=0.0,
                        machine_utilization=None,
                        rejected_order_ids=(),
                        reject_reasons=(),
                        notes=(reason4,),
                    )
                    results_by_W[W].append(fake_result)
                day += timedelta(days=1)
                continue

            for W in self.cfg.candidate_W:
                W_seconds = _parse_W(W)
                for w_start in _windows_for_day(W_seconds, day_start, day_end):
                    window_orders = self._orders_in_window(
                        w_start, W_seconds, sub_problem_key
                    )
                    if not window_orders:
                        continue
                    inv = InventorySnapshot.get_snapshot_at(self.snapshots, w_start)
                    result = simulate_window(
                        window_orders=window_orders,
                        inv=inv,
                        cfg=self.cfg,
                        sub_problem_key=sub_problem_key,
                        window_start=w_start,
                        W_seconds=W_seconds,
                    )
                    results_by_W[W].append(result)

            day += timedelta(days=1)

        # best_W:可行集中总访问数之和最小(若并列看命中率方差小——简化:取首个)
        feasible_Ws = {
            W: rs
            for W, rs in results_by_W.items()
            if rs and all(r.feasible for r in rs)
        }
        if not feasible_Ws:
            best_W = None
            best_total = None
        else:
            totals = {
                W: sum(r.total_visits for r in rs) for W, rs in feasible_Ws.items()
            }
            best_W = min(totals, key=totals.get)
            best_total = totals[best_W]

        return SweepResult(
            sub_problem_key=sub_problem_key,
            best_W=best_W,
            best_total_visits=best_total,
            window_results_by_W=results_by_W,
        )
