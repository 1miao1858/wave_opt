"""Joint MIP 求解器:窗口内同时决定子波组成 + 货架分配。

变量:x[i,w] / z_qty[i,k,s] / y[s,w]
约束:C1' 履行 / C2 库存 / C3 链接(含 (1-x[i,w])) / C4 子波组成 / C5 单子波上限
目标:min Σ_{w,s} y[s,w]

后端:Gurobi(restricted license 2000 变量上限)/ SCIP(无 license,慢)/ HiGHS(无 license,开源,常快 2-5x)。

建模收紧(2026-07-24):
- 删 h[s,k,w] 辅助变量:原 spec 3.4 留作"事后推算命中率"用,实际代码用 z_qty 推算,h 是死代码
- 删 C3b 约束:仅与 h 关联,删 h 后失去意义
- M_big 从全局 max(qty) 改为 per-(i,k) = qty[i,k]:LP 松弛更紧,求解快 10x+
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

try:
    import pyscipopt as ps
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False

try:
    import highspy
    import numpy as np
    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

from src.data_loader import InventorySnapshot, Order


@dataclass(frozen=True)
class MIPInput:
    window_orders: tuple[Order, ...]
    inv: InventorySnapshot
    N_max: int
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
        if self.solver_name == "gurobi":
            if not HAS_GUROBI:
                raise RuntimeError("gurobipy 未安装,请安装或改用 scip/highs")
            return self._solve_gurobi(inp)
        elif self.solver_name == "scip":
            if not HAS_SCIP:
                raise RuntimeError("pyscipopt 未安装,请 pip install pyscipopt")
            return self._solve_scip(inp)
        elif self.solver_name == "highs":
            if not HAS_HIGHS:
                raise RuntimeError("highspy 未安装,请 pip install highspy")
            return self._solve_highs(inp)
        else:
            raise ValueError(f"未知 solver_name={self.solver_name}(支持: gurobi / scip / highs)")

    def _prep_sets(self, inp: MIPInput):
        """共同集合预处理。"""
        I = list(range(len(inp.window_orders)))
        orders_by_idx = dict(zip(I, inp.window_orders))
        K_i = {i: sorted({l.sku_id for l in orders_by_idx[i].lines}) for i in I}
        S_k: dict[str, list[str]] = {}
        for k in {k for ks in K_i.values() for k in ks}:
            S_k[k] = sorted(inp.inv.sku_shelves.get(k, set()))
        S = sorted({s for shelves in S_k.values() for s in shelves})
        qty = {
            (i, k): sum(l.qty for l in orders_by_idx[i].lines if l.sku_id == k)
            for i in I for k in K_i[i]
        }
        inv_sk = dict(inp.inv.inv)
        n_sub = max(1, (len(I) + inp.N_max - 1) // inp.N_max)
        W_sub = list(range(n_sub))
        K_per_shelf = {s: [k for k, shelves in S_k.items() if s in shelves] for s in S}
        return I, orders_by_idx, K_i, S_k, S, qty, inv_sk, W_sub, K_per_shelf

    def _solve_scip(self, inp: MIPInput) -> MIPSolution:
        I, orders_by_idx, K_i, S_k, S, qty, inv_sk, W_sub, K_per_shelf = self._prep_sets(inp)

        m = ps.Model("joint_wave")
        m.hideOutput()
        m.setRealParam('limits/time', inp.time_limit)
        m.setRealParam('limits/gap', self.mip_gap)

        # 变量
        x = {(i, w): m.addVar(name=f"x_{i}_{w}", vtype='BINARY')
             for i in I for w in W_sub}
        z_qty = {(i, k, s): m.addVar(name=f"z_{i}_{k}_{s}", vtype='INTEGER', lb=0)
                 for i in I for k in K_i[i] for s in S_k[k]}
        y = {(s, w): m.addVar(name=f"y_{s}_{w}", vtype='BINARY')
             for s in S for w in W_sub}

        # C1' 履行
        for i in I:
            for k in K_i[i]:
                m.addCons(
                    ps.quicksum(z_qty[(i, k, s)] for s in S_k[k]) == qty[(i, k)],
                    name=f"C1_{i}_{k}",
                )

        # C2 库存
        for s in S:
            for k in K_per_shelf[s]:
                m.addCons(
                    ps.quicksum(
                        z_qty[(i, k, s)]
                        for i in I if k in K_i[i] and (i, k, s) in z_qty
                    ) <= inv_sk.get((s, k), 0),
                    name=f"C2_{s}_{k}",
                )

        # C3 链接(M_big = qty[i,k],per-(i,k) 紧化;删 C3b/h)
        for i in I:
            for k in K_i[i]:
                for s in S_k[k]:
                    for w in W_sub:
                        m.addCons(
                            z_qty[(i, k, s)]
                            <= qty[(i, k)] * y[(s, w)] + qty[(i, k)] * (1 - x[(i, w)]),
                            name=f"C3a_{i}_{k}_{s}_{w}",
                        )

        # C4 子波组成
        for i in I:
            m.addCons(
                ps.quicksum(x[(i, w)] for w in W_sub) == 1,
                name=f"C4_{i}",
            )

        # C5 单子波上限
        for w in W_sub:
            m.addCons(
                ps.quicksum(x[(i, w)] for i in I) <= inp.N_max,
                name=f"C5_{w}",
            )

        # 目标:min Σ_{w,s} y[s,w]
        m.setObjective(ps.quicksum(y[(s, w)] for s in S for w in W_sub),
                       sense='minimize')

        m.optimize()

        status = m.getStatus()
        if status == "infeasible":
            return MIPSolution(
                status="infeasible",
                wave_assignments=(),
                total_visits=0,
                hit_rate=0.0,
                consumption={},
                objective_value=0.0,
                solver_name=self.solver_name,
            )

        order_to_wave: dict[int, int] = {}
        for i in I:
            for w in W_sub:
                if m.getVal(x[(i, w)]) > 0.5:
                    order_to_wave[i] = w
                    break

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
            shelf_sku_hits: dict[str, set[str]] = {}

            for i in order_idxs:
                order_id = orders_by_idx[i].order_id
                for k in K_i[i]:
                    for s in S_k[k]:
                        q = m.getVal(z_qty[(i, k, s)])
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

            y_visited = [s for s in S if m.getVal(y[(s, w)]) > 0.5]
            visited = visited if visited else y_visited

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

        obj_val = m.getObjVal() if m.getObjVal() is not None else 0.0
        try:
            gap = m.getGap()
        except Exception:
            gap = None

        return MIPSolution(
            status="optimal" if status == "optimal" else "time_limit",
            wave_assignments=tuple(wave_assignments),
            total_visits=total_visits,
            hit_rate=hit_rate,
            consumption=consumption,
            objective_value=obj_val,
            mip_gap=gap,
            solver_name=self.solver_name,
        )

    def _solve_gurobi(self, inp: MIPInput) -> MIPSolution:
        if not HAS_GUROBI and self.solver_name == "gurobi":
            raise RuntimeError("gurobipy 未安装,请安装或改用 scip")

        I, orders_by_idx, K_i, S_k, S, qty, inv_sk, W_sub, K_per_shelf = self._prep_sets(inp)

        # === Gurobi 模型 ===
        m = gp.Model("joint_wave")
        m.Params.TimeLimit = inp.time_limit
        m.Params.MIPGap = self.mip_gap
        m.Params.OutputFlag = 0

        # 变量(删 h:辅助变量,只用于事后命中率推算,实际用 z_qty 推算,死代码)
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
        m.update()

        # === C1'. 履行约束(允许跨货架拆分)===
        for i in I:
            for k in K_i[i]:
                m.addConstr(
                    gp.quicksum(z_qty[(i, k, s)] for s in S_k[k]) == qty[(i, k)],
                    name=f"C1_{i}_{k}",
                )

        # === C2. 库存不超(跨子波共享)===
        for s in S:
            for k in K_per_shelf[s]:
                m.addConstr(
                    gp.quicksum(
                        z_qty[(i, k, s)]
                        for i in I if k in K_i[i] and (i, k, s) in z_qty
                    ) <= inv_sk.get((s, k), 0),
                    name=f"C2_{s}_{k}",
                )

        # === C3. 访问与拣货关联(M_big = qty[i,k] per-(i,k) 紧化;删 C3b/h)===
        for i in I:
            for k in K_i[i]:
                for s in S_k[k]:
                    for w in W_sub:
                        m.addConstr(
                            z_qty[(i, k, s)]
                            <= qty[(i, k)] * y[(s, w)] + qty[(i, k)] * (1 - x[(i, w)]),
                            name=f"C3a_{i}_{k}_{s}_{w}",
                        )

        # === C4. 子波组成(每订单必进且仅进一个子波)===
        for i in I:
            m.addConstr(
                gp.quicksum(x[(i, w)] for w in W_sub) == 1,
                name=f"C4_{i}",
            )

        # === C5. 单子波订单数上限 ===
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

    def _solve_highs(self, inp: MIPInput) -> MIPSolution:
        """HiGHS backend — 开源 MILP 求解器,API 用 col/row 索引 + numpy 数组。"""
        I, orders_by_idx, K_i, S_k, S, qty, inv_sk, W_sub, K_per_shelf = self._prep_sets(inp)

        h = highspy.Highs()
        h.silent()
        h.setOptionValue("time_limit", float(inp.time_limit))
        h.setOptionValue("mip_rel_gap", float(self.mip_gap))
        h.setOptionValue("output_flag", "false")

        # === 添加列(变量)===
        # 用数组批量 addCol 更高效,但 HiGHS Python binding 一次只加一个 col
        x_idx: dict[tuple[int, int], int] = {}
        z_idx: dict[tuple[int, int, str], int] = {}
        y_idx: dict[tuple[str, int], int] = {}
        empty_idx = np.array([], dtype=np.int32)
        empty_val = np.array([], dtype=np.float64)

        # x[i,w]:二元,cost=0
        for i in I:
            for w in W_sub:
                x_idx[(i, w)] = h.getNumCol()
                h.addCol(0.0, 0.0, 1.0, 0, empty_idx, empty_val)
                h.setInteger(h.getNumCol() - 1)

        # z_qty[i,k,s]:整数,上界 = qty[i,k](自然上界,跟 C3 紧化一致),cost=0
        for i in I:
            for k in K_i[i]:
                for s in S_k[k]:
                    z_idx[(i, k, s)] = h.getNumCol()
                    ub = float(qty[(i, k)])
                    h.addCol(0.0, 0.0, ub, 0, empty_idx, empty_val)
                    h.setInteger(h.getNumCol() - 1)

        # y[s,w]:二元,cost=1(目标函数系数)
        for s in S:
            for w in W_sub:
                y_idx[(s, w)] = h.getNumCol()
                h.addCol(1.0, 0.0, 1.0, 0, empty_idx, empty_val)
                h.setInteger(h.getNumCol() - 1)

        # === 添加行(约束)===
        # C1' 履行:Σ_{s∈S(k)} z_qty[i,k,s] == qty[i,k]
        for i in I:
            for k in K_i[i]:
                shelves = S_k[k]
                idxs = np.array([z_idx[(i, k, s)] for s in shelves], dtype=np.int32)
                vals = np.ones(len(shelves), dtype=np.float64)
                q = float(qty[(i, k)])
                h.addRow(q, q, len(shelves), idxs, vals)

        # C2 库存:Σ_{i: k∈K(i)} z_qty[i,k,s] ≤ inv[s,k]
        for s in S:
            for k in K_per_shelf[s]:
                cols = []
                for i in I:
                    if k in K_i[i] and (i, k, s) in z_idx:
                        cols.append(z_idx[(i, k, s)])
                if not cols:
                    continue
                idxs = np.array(cols, dtype=np.int32)
                vals = np.ones(len(cols), dtype=np.float64)
                ub = float(inv_sk.get((s, k), 0))
                h.addRow(-highspy.kHighsInf, ub, len(cols), idxs, vals)

        # C3 链接:z_qty[i,k,s] ≤ qty[i,k] * y[s,w] + qty[i,k] * (1 - x[i,w])
        #   →  qty[i,k] * y[s,w] + qty[i,k] * (1 - x[i,w]) - z_qty[i,k,s] ≥ 0
        #   →  -qty[i,k]*y[s,w] + qty[i,k]*x[i,w] + z_qty[i,k,s] ≤ qty[i,k]
        for i in I:
            for k in K_i[i]:
                q = float(qty[(i, k)])
                for s in S_k[k]:
                    for w in W_sub:
                        idxs = np.array([
                            y_idx[(s, w)],
                            x_idx[(i, w)],
                            z_idx[(i, k, s)],
                        ], dtype=np.int32)
                        vals = np.array([-q, q, 1.0], dtype=np.float64)
                        h.addRow(-highspy.kHighsInf, q, 3, idxs, vals)

        # C4 子波组成:Σ_w x[i,w] == 1
        for i in I:
            idxs = np.array([x_idx[(i, w)] for w in W_sub], dtype=np.int32)
            vals = np.ones(len(W_sub), dtype=np.float64)
            h.addRow(1.0, 1.0, len(W_sub), idxs, vals)

        # C5 单子波上限:Σ_i x[i,w] ≤ N_max
        for w in W_sub:
            idxs = np.array([x_idx[(i, w)] for i in I], dtype=np.int32)
            vals = np.ones(len(I), dtype=np.float64)
            h.addRow(-highspy.kHighsInf, float(inp.N_max), len(I), idxs, vals)

        # === 目标:min Σ y[s,w](已在 addCol cost=1 for y, 其余 0)===
        h.setMinimize()

        # === 求解 ===
        h.run()
        status_name = h.getModelStatus().name

        if status_name == "kInfeasible":
            return MIPSolution(
                status="infeasible",
                wave_assignments=(),
                total_visits=0,
                hit_rate=0.0,
                consumption={},
                objective_value=0.0,
                solver_name=self.solver_name,
            )

        # === 提取解 ===
        col_value = h.getSolution().col_value

        def _get(var_idx: int) -> float:
            return float(col_value[var_idx])

        # 子波组成
        order_to_wave: dict[int, int] = {}
        for i in I:
            for w in W_sub:
                if _get(x_idx[(i, w)]) > 0.5:
                    order_to_wave[i] = w
                    break

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
            shelf_sku_hits: dict[str, set[str]] = {}

            for i in order_idxs:
                order_id = orders_by_idx[i].order_id
                for k in K_i[i]:
                    for s in S_k[k]:
                        q = _get(z_idx[(i, k, s)])
                        if q < 0.5:
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

            y_visited = [s for s in S if _get(y_idx[(s, w)]) > 0.5]
            visited = visited if visited else y_visited

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

        info = h.getInfo()
        obj_val = float(info.objective_function_value)
        gap = float(info.mip_gap) if info.mip_gap != float('inf') else None

        # 状态映射
        if status_name == "kOptimal":
            out_status = "optimal"
        elif status_name in ("kTimeLimit", "kIterationLimit", "kSolutionLimit",
                              "kObjectiveTarget", "kObjectiveBound"):
            out_status = "time_limit"
        elif status_name == "kUnbounded":
            out_status = "infeasible"
        else:
            out_status = "time_limit"  # 兜底

        return MIPSolution(
            status=out_status,
            wave_assignments=tuple(wave_assignments),
            total_visits=total_visits,
            hit_rate=hit_rate,
            consumption=consumption,
            objective_value=obj_val,
            mip_gap=gap,
            solver_name=self.solver_name,
        )
