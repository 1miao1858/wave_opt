"""贪心基线:超时 fallback + 6.1 对比。

策略:FIFO 顺序 + 边际增益贪心。
- 订单按 timestamp 升序
- 依次塞进当前子波,塞前先看是否会触发新货架访问;若必须触发则看新子波是否更好
- 对每 SKU 选剩余库存最多的货架(优先选不触发新访问的货架)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data_loader import InventorySnapshot, Order
from src.mip_solver import MIPInput, MIPSolution, WaveAssignment


def greedy_solve(inp: MIPInput) -> MIPSolution:
    orders = sorted(inp.window_orders, key=lambda o: o.timestamp)
    inv = dict(inp.inv.inv)  # 可变副本
    # sku -> list of shelves(按当前库存降序)
    sku_shelves = {
        k: sorted(list(s_set), key=lambda s: -inv.get((s, k), 0))
        for k, s_set in inp.inv.sku_shelves.items()
    }

    N_max = inp.N_max
    subwaves: list[dict] = []
    current: dict | None = None

    def open_new_subwave() -> dict:
        return {
            "orders": [],
            "visited": set(),
            "pick_qty": {},
            "per_order_pick_qty": {},
            "shelf_sku_hits": {},
        }

    for order in orders:
        # 对每 SKU 选货架;记录 picks 带 order_id
        picks: list[tuple[str, str, int]] = []  # (sku, shelf, qty)
        need_new_visit = False
        for line in order.lines:
            chosen_shelf = None
            # 优先选已在 visited 的货架
            for s in sku_shelves.get(line.sku_id, []):
                if current and s in current["visited"]:
                    if inv.get((s, line.sku_id), 0) >= line.qty:
                        chosen_shelf = s
                        break
            if chosen_shelf is None:
                # 退而选库存最多的货架(可能触发新访问)
                for s in sku_shelves.get(line.sku_id, []):
                    if inv.get((s, line.sku_id), 0) >= line.qty:
                        chosen_shelf = s
                        need_new_visit = True
                        break
            if chosen_shelf is None:
                # 跨货架凑齐(库存任一货架都不够)
                remaining = line.qty
                for s in sku_shelves.get(line.sku_id, []):
                    avail = inv.get((s, line.sku_id), 0)
                    if avail <= 0:
                        continue
                    take = min(avail, remaining)
                    picks.append((line.sku_id, s, take))
                    remaining -= take
                    if s not in (current["visited"] if current else set()):
                        need_new_visit = True
                    if remaining <= 0:
                        break
                if remaining > 0:
                    raise RuntimeError(
                        f"订单 {order.order_id} 的 SKU {line.sku_id} 全货架库存不足"
                    )
                continue
            picks.append((line.sku_id, chosen_shelf, line.qty))

        # 决策:塞当前子波还是开新子波
        if current is None or len(current["orders"]) >= N_max:
            current = open_new_subwave()
            subwaves.append(current)

        # 塞进当前子波
        current["orders"].append(order)
        for sku, s, q in picks:
            current["pick_qty"][(sku, s)] = (
                current["pick_qty"].get((sku, s), 0) + q
            )
            current["per_order_pick_qty"][(order.order_id, sku, s)] = q
            current["visited"].add(s)
            current["shelf_sku_hits"].setdefault(s, set()).add(sku)
            inv[(s, sku)] = inv.get((s, sku), 0) - q

    # 构造 MIPSolution
    wave_assignments: list[WaveAssignment] = []
    consumption: dict[tuple[str, str], int] = {}
    total_hits = 0
    total_visits = 0
    for w_idx, w in enumerate(subwaves):
        visited_tuple = tuple(sorted(w["visited"]))
        hits = sum(len(sk_set) for sk_set in w["shelf_sku_hits"].values())
        total_hits += hits
        total_visits += len(visited_tuple)
        for (sku, s), q in w["pick_qty"].items():
            consumption[(s, sku)] = consumption.get((s, sku), 0) + q
        wave_assignments.append(
            WaveAssignment(
                subwave_idx=w_idx,
                orders=tuple(w["orders"]),
                visited_shelves=visited_tuple,
                pick_qty=dict(w["pick_qty"]),
                per_order_pick_qty=dict(w["per_order_pick_qty"]),
            )
        )

    hit_rate = (total_hits / total_visits) if total_visits > 0 else 0.0
    return MIPSolution(
        status="fallback_greedy",
        wave_assignments=tuple(wave_assignments),
        total_visits=total_visits,
        hit_rate=hit_rate,
        consumption=consumption,
        objective_value=float(total_visits),
        solver_name="greedy",
    )
