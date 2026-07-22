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
    # (sku, shelf) -> qty
    pick_qty: dict[tuple[str, str], int] = field(default_factory=dict)


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

        # 集合
        I = list(range(len(inp.window_orders)))
        orders_by_idx = dict(zip(I, inp.window_orders))
        # K(i):订单 i 涉及的 SKU 集
        K_i = {i: [l.sku_id for l in orders_by_idx[i].lines] for i in I}
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

        # === Gurobi 建模 ===
        # 变量、约束、目标的实际构建在 Task 7-10 完成;此处先返回一个未实现错误,
        # 让 Task 6 的测试失败 → 在 Task 7-10 中逐步实现并使其通过。
        raise NotImplementedError(
            "JointMIPSolver.solve 将在 Task 7-10 完成;Task 6 只建数据结构"
        )
