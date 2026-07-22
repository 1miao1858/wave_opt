"""Joint MIP 求解器:窗口内同时决定子波组成 + 货架分配。

变量:x[i,w] / z_qty[i,k,s] / y[s,w] / h[s,k,w]
约束:C1' 履行 / C2 库存 / C3 链接(含 (1-x[i,w])) / C4 子波组成 / C5 单子波上限
目标:min Σ_{w,s} y[s,w]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

from src.data_loader import InventorySnapshot, Order


@dataclass(frozen=True)
class MIPInput:
    window_orders: tuple[Order, ...]
    inv: InventorySnapshot
    N_max: int
    M_big: int
    time_limit: int  # 秒


@dataclass(frozen=True)
class WaveAssignment:
    subwave_idx: int  # 0-based
    orders: tuple[Order, ...]
    visited_shelves: tuple[str, ...]  # y[s,w]=1 的货架
    # (sku, shelf) -> qty:子波内聚合拣量,用于 6.2/6.7 报告
    pick_qty: dict[tuple[str, str], int] = field(default_factory=dict)
    # (order_id, sku, shelf) -> qty:按订单拆分,用于 6.8a/6.8b 验证检查
    per_order_pick_qty: dict[tuple[str, str, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class MIPSolution:
    status: str  # "optimal" / "time_limit" / "infeasible" / "fallback_greedy"
    wave_assignments: tuple[WaveAssignment, ...]
    total_visits: int
    hit_rate: float  # Σ hits / Σ visits
    consumption: dict[tuple[str, str], int]  # (shelf, sku) -> 总消耗
    objective_value: float
    mip_gap: float | None = None
    solver_name: str = "gurobi"


class JointMIPSolver:
    """窗口内 Joint MIP:决定子波组成 + 货架分配。"""

    def __init__(self, solver_name: str = "gurobi", mip_gap: float = 0.05):
        self.solver_name = solver_name
        self.mip_gap = mip_gap

    def solve(self, inp: MIPInput) -> MIPSolution:
        if not HAS_GUROBI and self.solver_name == "gurobi":
            raise RuntimeError("gurobipy 未安装,请安装或改用 scip")

        # === 集合与参数 ===
        I = list(range(len(inp.window_orders)))
        orders_by_idx = dict(zip(I, inp.window_orders))
        # K(i):订单 i 涉及的 SKU 集(去重,用于变量与约束枚举)
        K_i = {i: sorted({l.sku_id for l in orders_by_idx[i].lines}) for i in I}
        # S(k):含 SKU k 的货架集
        S_k: dict[str, list[str]] = {}
        for k in {k for ks in K_i.values() for k in ks}:
            S_k[k] = sorted(inp.inv.sku_shelves.get(k, set()))
        # 全 SKU 集
        K = sorted(S_k.keys())
        # 全货架集(只取相关货架)
        S = sorted({s for shelves in S_k.values() for s in shelves})
        # 参数
        qty = {
            (i, k): sum(l.qty for l in orders_by_idx[i].lines if l.sku_id == k)
            for i in I for k in K_i[i]
        }
        inv_sk = dict(inp.inv.inv)  # (shelf, sku) -> qty

        # 子波数 |W_sub| = ceil(|I| / N_max)
        n_sub = max(1, (len(I) + inp.N_max - 1) // inp.N_max)
        W_sub = list(range(n_sub))

        # 大 M
        M_big = inp.M_big

        # === Gurobi 模型 ===
        m = gp.Model("joint_wave")
        m.Params.TimeLimit = inp.time_limit
        m.Params.MIPGap = self.mip_gap
        m.Params.OutputFlag = 0

        # 变量
        x = {
            (i, w): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{w}")
            for i in I for w in W_sub
        }
        z_qty = {
            (i, k, s): m.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"z_{i}_{k}_{s}"
            )
            for i in I for k in K_i[i] for s in S_k[k]
        }
        y = {
            (s, w): m.addVar(vtype=GRB.BINARY, name=f"y_{s}_{w}")
            for s in S for w in W_sub
        }
        h = {
            (s, k, w): m.addVar(vtype=GRB.BINARY, name=f"h_{s}_{k}_{w}")
            for s in S for k in _skus_on_shelf(S_k, s) for w in W_sub
        }
        m.update()

        # === C1'. 履行约束(允许跨货架拆分)===
        # Σ_{s ∈ S(k)} z_qty[i,k,s] = qty[i,k]   ∀ i ∈ I, k ∈ K(i)
        for i in I:
            for k in K_i[i]:
                m.addConstr(
                    gp.quicksum(z_qty[(i, k, s)] for s in S_k[k]) == qty[(i, k)],
                    name=f"C1_{i}_{k}",
                )

        # === C2. 库存不超(跨子波共享)===
        # Σ_{i: k ∈ K(i)} z_qty[i,k,s] ≤ inv[s,k]   ∀ s ∈ S, k ∈ K(s)
        for s in S:
            for k in _skus_on_shelf(S_k, s):
                m.addConstr(
                    gp.quicksum(
                        z_qty[(i, k, s)]
                        for i in I if k in K_i[i] and (i, k, s) in z_qty
                    ) <= inv_sk.get((s, k), 0),
                    name=f"C2_{s}_{k}",
                )

        # === C3. 访问与拣货关联(通过 (1-x[i,w]) 隐式按子波关联)===
        # C3a: z_qty[i,k,s] ≤ M_big × y[s,w] + M_big × (1 - x[i,w])    ∀ i,k,s,w
        # C3b: z_qty[i,k,s] ≤ M_big × h[s,k,w] + M_big × (1 - x[i,w])  ∀ i,k,s,w
        for i in I:
            for k in K_i[i]:
                for s in S_k[k]:
                    for w in W_sub:
                        m.addConstr(
                            z_qty[(i, k, s)]
                            <= M_big * y[(s, w)] + M_big * (1 - x[(i, w)]),
                            name=f"C3a_{i}_{k}_{s}_{w}",
                        )
                        if (s, k, w) in h:
                            m.addConstr(
                                z_qty[(i, k, s)]
                                <= M_big * h[(s, k, w)] + M_big * (1 - x[(i, w)]),
                                name=f"C3b_{i}_{k}_{s}_{w}",
                            )

        # === C4. 子波组成(每订单必进且仅进一个子波)===
        # Σ_{w ∈ W_sub} x[i,w] = 1   ∀ i ∈ I
        for i in I:
            m.addConstr(
                gp.quicksum(x[(i, w)] for w in W_sub) == 1,
                name=f"C4_{i}",
            )

        # === C5. 单子波订单数上限 ===
        # Σ_{i ∈ I} x[i,w] ≤ N_max   ∀ w ∈ W_sub
        for w in W_sub:
            m.addConstr(
                gp.quicksum(x[(i, w)] for i in I) <= inp.N_max,
                name=f"C5_{w}",
            )

        # === 目标函数:min Σ_{w,s} y[s,w] ===
        m.setObjective(gp.quicksum(y[(s, w)] for s in S for w in W_sub), GRB.MINIMIZE)

        # === 求解 ===
        m.optimize()

        # === 提取结果 ===
        if m.status == GRB.INFEASIBLE:
            return MIPSolution(
                status="infeasible",
                wave_assignments=(),
                total_visits=0,
                hit_rate=0.0,
                consumption={},
                objective_value=0.0,
                solver_name=self.solver_name,
            )

        # 子波组成:每订单进哪个子波
        order_to_wave: dict[int, int] = {}
        for i in I:
            for w in W_sub:
                if x[(i, w)].X > 0.5:
                    order_to_wave[i] = w
                    break

        # 按子波聚合
        wave_to_orders: dict[int, list] = {w: [] for w in W_sub}
        for i, w in order_to_wave.items():
            wave_to_orders[w].append(i)

        wave_assignments: list[WaveAssignment] = []
        consumption: dict[tuple[str, str], int] = {}
        total_hits = 0
        total_visits = 0

        for w in W_sub:
            order_idxs = wave_to_orders[w]
            orders_in_w = tuple(orders_by_idx[i] for i in order_idxs)
            visited: list[str] = []
            pick_qty: dict[tuple[str, str], int] = {}
            per_order_pick_qty: dict[tuple[str, str, str], int] = {}
            shelf_sku_hits: dict[str, set[str]] = {}  # shelf -> {sku picked}

            for i in order_idxs:
                order_id = orders_by_idx[i].order_id
                for k in K_i[i]:
                    for s in S_k[k]:
                        q = z_qty[(i, k, s)].X
                        if q is None or q < 0.5:
                            continue
                        q_int = int(round(q))
                        if q_int == 0:
                            continue
                        pick_qty[(k, s)] = pick_qty.get((k, s), 0) + q_int
                        per_order_pick_qty[(order_id, k, s)] = q_int
                        consumption[(s, k)] = consumption.get((s, k), 0) + q_int
                        if s not in visited:
                            visited.append(s)
                        shelf_sku_hits.setdefault(s, set()).add(k)

            # 用 y[(s,w)].X 校验
            y_visited = [s for s in S if y[(s, w)].X > 0.5]
            visited = visited if visited else y_visited  # 优先用 z_qty 推

            hits = sum(len(sk_set) for sk_set in shelf_sku_hits.values())
            total_hits += hits
            total_visits += len(visited)

            wave_assignments.append(
                WaveAssignment(
                    subwave_idx=w,
                    orders=orders_in_w,
                    visited_shelves=tuple(visited),
                    pick_qty=pick_qty,
                    per_order_pick_qty=per_order_pick_qty,
                )
            )

        hit_rate = (total_hits / total_visits) if total_visits > 0 else 0.0

        return MIPSolution(
            status="optimal" if m.status == GRB.OPTIMAL else "time_limit",
            wave_assignments=tuple(wave_assignments),
            total_visits=total_visits,
            hit_rate=hit_rate,
            consumption=consumption,
            objective_value=m.ObjVal if m.ObjVal is not None else 0.0,
            mip_gap=m.MIPGap if m.MIPGap is not None else None,
            solver_name=self.solver_name,
        )


def _skus_on_shelf(S_k: dict[str, list[str]], shelf: str) -> list[str]:
    """返回货架 shelf 上的所有 SKU(K(s))。"""
    return [k for k, shelves in S_k.items() if shelf in shelves]
