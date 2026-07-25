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
        elif self.solver_name == "two_phase_a":
            if not HAS_HIGHS:
                raise RuntimeError("two_phase_a 依赖 highspy,请 pip install highspy")
            return self._solve_two_phase_a(inp)
        elif self.solver_name == "two_phase_b":
            if not HAS_HIGHS:
                raise RuntimeError("two_phase_b 依赖 highspy,请 pip install highspy")
            return self._solve_two_phase_b(inp)
        elif self.solver_name == "greedy_a":
            return self._solve_greedy_a(inp)
        else:
            raise ValueError(f"未知 solver_name={self.solver_name}(支持: gurobi / scip / highs / two_phase_a / two_phase_b / greedy_a)")

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

    def _solve_two_phase_a(self, inp: MIPInput) -> MIPSolution:
        """方案 A:阶段 1 解 z_qty(全局货架占用最小),阶段 2 解 x[i,w](波次组合)。"""
        I, orders_by_idx, K_i, S_k, S, qty, inv_sk, W_sub, K_per_shelf = self._prep_sets(inp)
        empty_idx = np.array([], dtype=np.int32)
        empty_val = np.array([], dtype=np.float64)

        # === 阶段 1:全局货架分配,最小化占用货架数 ===
        h1 = highspy.Highs()
        h1.silent()
        phase1_tl = max(10, inp.time_limit // 3)
        h1.setOptionValue("time_limit", float(phase1_tl))
        h1.setOptionValue("mip_rel_gap", float(self.mip_gap))
        h1.setOptionValue("output_flag", "false")

        z_idx: dict[tuple[int, str, str], int] = {}
        y1_idx: dict[str, int] = {}

        for i in I:
            for k in K_i[i]:
                for s in S_k[k]:
                    z_idx[(i, k, s)] = h1.getNumCol()
                    h1.addCol(0.0, 0.0, float(qty[(i, k)]), 0, empty_idx, empty_val)
                    h1.setInteger(h1.getNumCol() - 1)

        for s in S:
            y1_idx[s] = h1.getNumCol()
            h1.addCol(1.0, 0.0, 1.0, 0, empty_idx, empty_val)
            h1.setInteger(h1.getNumCol() - 1)

        # C1' 履行
        for i in I:
            for k in K_i[i]:
                shelves = S_k[k]
                idxs = np.array([z_idx[(i, k, s)] for s in shelves], dtype=np.int32)
                vals = np.ones(len(shelves), dtype=np.float64)
                q = float(qty[(i, k)])
                h1.addRow(q, q, len(shelves), idxs, vals)

        # C2 全局库存
        for s in S:
            for k in K_per_shelf[s]:
                cols = [z_idx[(i, k, s)] for i in I if k in K_i[i] and (i, k, s) in z_idx]
                if not cols:
                    continue
                idxs = np.array(cols, dtype=np.int32)
                vals = np.ones(len(cols), dtype=np.float64)
                ub = float(inv_sk.get((s, k), 0))
                h1.addRow(-highspy.kHighsInf, ub, len(cols), idxs, vals)

        # C3' z_qty[i,k,s] ≤ qty[i,k] · y[s] → -qty[i,k]·y[s] + z_qty ≤ 0
        for i in I:
            for k in K_i[i]:
                q = float(qty[(i, k)])
                for s in S_k[k]:
                    idxs = np.array([y1_idx[s], z_idx[(i, k, s)]], dtype=np.int32)
                    vals = np.array([-q, 1.0], dtype=np.float64)
                    h1.addRow(-highspy.kHighsInf, 0.0, 2, idxs, vals)

        h1.setMinimize()
        h1.run()
        st1 = h1.getModelStatus().name
        if st1 == "kInfeasible":
            return MIPSolution(
                status="infeasible",
                wave_assignments=(),
                total_visits=0, hit_rate=0.0, consumption={},
                objective_value=0.0, solver_name=self.solver_name,
            )

        col1 = h1.getSolution().col_value
        z_qty_fixed: dict[tuple[int, str, str], int] = {}
        for key, idx in z_idx.items():
            v = float(col1[idx])
            z_qty_fixed[key] = int(round(v)) if v >= 0.5 else 0

        # === 阶段 2:波次组合,给定 z_qty ===
        R_i: dict[int, set[str]] = {i: set() for i in I}
        for (i, k, s), q_int in z_qty_fixed.items():
            if q_int > 0:
                R_i[i].add(s)

        phase2_tl = max(10, inp.time_limit - phase1_tl)
        h2 = highspy.Highs()
        h2.silent()
        h2.setOptionValue("time_limit", float(phase2_tl))
        h2.setOptionValue("mip_rel_gap", float(self.mip_gap))
        h2.setOptionValue("output_flag", "false")

        x2_idx: dict[tuple[int, int], int] = {}
        y2_idx: dict[tuple[str, int], int] = {}

        for i in I:
            for w in W_sub:
                x2_idx[(i, w)] = h2.getNumCol()
                h2.addCol(0.0, 0.0, 1.0, 0, empty_idx, empty_val)
                h2.setInteger(h2.getNumCol() - 1)

        for s in S:
            for w in W_sub:
                y2_idx[(s, w)] = h2.getNumCol()
                h2.addCol(1.0, 0.0, 1.0, 0, empty_idx, empty_val)
                h2.setInteger(h2.getNumCol() - 1)

        # C4:Σ_w x[i,w] = 1
        for i in I:
            idxs = np.array([x2_idx[(i, w)] for w in W_sub], dtype=np.int32)
            vals = np.ones(len(W_sub), dtype=np.float64)
            h2.addRow(1.0, 1.0, len(W_sub), idxs, vals)

        # C5:Σ_i x[i,w] ≤ N_max
        for w in W_sub:
            idxs = np.array([x2_idx[(i, w)] for i in I], dtype=np.int32)
            vals = np.ones(len(I), dtype=np.float64)
            h2.addRow(-highspy.kHighsInf, float(inp.N_max), len(I), idxs, vals)

        # C6:Σ_{i : s ∈ R_i} x[i,w] ≤ |I| · y[s,w] → -|I|·y[s,w] + Σ x ≤ 0
        for s in S:
            orders_using_s = [i for i in I if s in R_i[i]]
            if not orders_using_s:
                continue
            for w in W_sub:
                idxs = np.array(
                    [y2_idx[(s, w)]] + [x2_idx[(i, w)] for i in orders_using_s],
                    dtype=np.int32,
                )
                vals = np.array(
                    [-float(len(I))] + [1.0] * len(orders_using_s),
                    dtype=np.float64,
                )
                h2.addRow(-highspy.kHighsInf, 0.0, len(orders_using_s) + 1, idxs, vals)

        h2.setMinimize()
        h2.run()
        st2 = h2.getModelStatus().name

        col2 = h2.getSolution().col_value

        order_to_wave: dict[int, int] = {}
        for i in I:
            for w in W_sub:
                if float(col2[x2_idx[(i, w)]]) > 0.5:
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
                        q_int = z_qty_fixed.get((i, k, s), 0)
                        if q_int == 0:
                            continue
                        pick_qty[(k, s)] = pick_qty.get((k, s), 0) + q_int
                        per_order_pick_qty[(order_id, k, s)] = q_int
                        consumption[(s, k)] = consumption.get((s, k), 0) + q_int
                        if s not in visited:
                            visited.append(s)
                        shelf_sku_hits.setdefault(s, set()).add(k)

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

        info2 = h2.getInfo()
        obj_val = float(info2.objective_function_value)
        gap2 = float(info2.mip_gap) if info2.mip_gap != float('inf') else None

        if st1 == "kOptimal" and st2 == "kOptimal":
            out_status = "optimal"
        elif st2 == "kInfeasible":
            out_status = "infeasible"
        else:
            out_status = "time_limit"

        return MIPSolution(
            status=out_status,
            wave_assignments=tuple(wave_assignments),
            total_visits=total_visits,
            hit_rate=hit_rate,
            consumption=consumption,
            objective_value=obj_val,
            mip_gap=gap2,
            solver_name=self.solver_name,
        )

    def _solve_two_phase_b(self, inp: MIPInput) -> MIPSolution:
        """方案 B:阶段 1 解 x[i,w](SKU-子波占用最小),阶段 2 每子波独立解 z_qty。"""
        I, orders_by_idx, K_i, S_k, _S, qty, inv_sk, W_sub, _K_per_shelf = self._prep_sets(inp)
        empty_idx = np.array([], dtype=np.int32)
        empty_val = np.array([], dtype=np.float64)

        # === 阶段 1:波次组合,最小化 SKU-子波占用数 ===
        h1 = highspy.Highs()
        h1.silent()
        phase1_tl = max(10, inp.time_limit // 3)
        h1.setOptionValue("time_limit", float(phase1_tl))
        h1.setOptionValue("mip_rel_gap", float(self.mip_gap))
        h1.setOptionValue("output_flag", "false")

        x1_idx: dict[tuple[int, int], int] = {}
        for i in I:
            for w in W_sub:
                x1_idx[(i, w)] = h1.getNumCol()
                h1.addCol(0.0, 0.0, 1.0, 0, empty_idx, empty_val)
                h1.setInteger(h1.getNumCol() - 1)

        # z[k,w]:二元,子波 w 是否有订单用 SKU k
        # K 全集 = ∪_i K_i
        K_all = sorted({k for ks in K_i.values() for k in ks})
        z1_idx: dict[tuple[str, int], int] = {}
        for k in K_all:
            for w in W_sub:
                z1_idx[(k, w)] = h1.getNumCol()
                h1.addCol(1.0, 0.0, 1.0, 0, empty_idx, empty_val)  # cost=1
                h1.setInteger(h1.getNumCol() - 1)

        # C4:Σ_w x[i,w] = 1
        for i in I:
            idxs = np.array([x1_idx[(i, w)] for w in W_sub], dtype=np.int32)
            vals = np.ones(len(W_sub), dtype=np.float64)
            h1.addRow(1.0, 1.0, len(W_sub), idxs, vals)

        # C5:Σ_i x[i,w] ≤ N_max
        for w in W_sub:
            idxs = np.array([x1_idx[(i, w)] for i in I], dtype=np.int32)
            vals = np.ones(len(I), dtype=np.float64)
            h1.addRow(-highspy.kHighsInf, float(inp.N_max), len(I), idxs, vals)

        # C7:x[i,w] ≤ z[k,w] ∀ i, k ∈ K_i, w → z[k,w] - x[i,w] ≥ 0 → x[i,w] - z[k,w] ≤ 0
        for i in I:
            for k in K_i[i]:
                for w in W_sub:
                    idxs = np.array([x1_idx[(i, w)], z1_idx[(k, w)]], dtype=np.int32)
                    vals = np.array([1.0, -1.0], dtype=np.float64)
                    h1.addRow(-highspy.kHighsInf, 0.0, 2, idxs, vals)

        h1.setMinimize()
        h1.run()
        st1 = h1.getModelStatus().name
        if st1 == "kInfeasible":
            return MIPSolution(
                status="infeasible",
                wave_assignments=(),
                total_visits=0, hit_rate=0.0, consumption={},
                objective_value=0.0, solver_name=self.solver_name,
            )

        col1 = h1.getSolution().col_value
        # 提取 x
        order_to_wave: dict[int, int] = {}
        for i in I:
            for w in W_sub:
                if float(col1[x1_idx[(i, w)]]) > 0.5:
                    order_to_wave[i] = w
                    break

        wave_to_orders: dict[int, list] = {w: [] for w in W_sub}
        for i, w in order_to_wave.items():
            wave_to_orders[w].append(i)

        # === 阶段 2:每子波独立 MIP,顺序解 + 库存扣减 ===
        inv_remaining: dict[tuple[str, str], int] = dict(inv_sk)  # 可变副本
        phase2_tl_per = max(10, (inp.time_limit - phase1_tl) // max(1, len(W_sub)))

        wave_assignments: list[WaveAssignment] = []
        consumption: dict[tuple[str, str], int] = {}
        total_hits = 0
        total_visits = 0
        notes: list[str] = []
        all_optimal = (st1 == "kOptimal")

        for w in W_sub:
            order_idxs = wave_to_orders[w]
            if not order_idxs:
                wave_assignments.append(
                    WaveAssignment(subwave_idx=w, orders=(), visited_shelves=())
                )
                continue

            # 子波内 SKU-货架 限制
            # 收集本子波涉及的 (k, s) 对
            sub_K: set[str] = set()
            for i in order_idxs:
                sub_K.update(K_i[i])
            sub_S_k: dict[str, list[str]] = {k: S_k.get(k, []) for k in sub_K}
            sub_S = sorted({s for shelves in sub_S_k.values() for s in shelves})
            sub_qty = {(i, k): qty[(i, k)] for i in order_idxs for k in K_i[i]}

            # 子波内 z_idx / y_idx
            hw = highspy.Highs()
            hw.silent()
            hw.setOptionValue("time_limit", float(phase2_tl_per))
            hw.setOptionValue("mip_rel_gap", float(self.mip_gap))
            hw.setOptionValue("output_flag", "false")

            zw_idx: dict[tuple[int, str, str], int] = {}
            yw_idx: dict[str, int] = {}

            for i in order_idxs:
                for k in K_i[i]:
                    for s in sub_S_k[k]:
                        zw_idx[(i, k, s)] = hw.getNumCol()
                        hw.addCol(0.0, 0.0, float(sub_qty[(i, k)]), 0, empty_idx, empty_val)
                        hw.setInteger(hw.getNumCol() - 1)

            for s in sub_S:
                yw_idx[s] = hw.getNumCol()
                hw.addCol(1.0, 0.0, 1.0, 0, empty_idx, empty_val)
                hw.setInteger(hw.getNumCol() - 1)

            # C1' 履行
            for i in order_idxs:
                for k in K_i[i]:
                    shelves = sub_S_k[k]
                    if not shelves:
                        continue
                    idxs = np.array([zw_idx[(i, k, s)] for s in shelves], dtype=np.int32)
                    vals = np.ones(len(shelves), dtype=np.float64)
                    q = float(sub_qty[(i, k)])
                    hw.addRow(q, q, len(shelves), idxs, vals)

            # C2 子波内剩余库存
            for s in sub_S:
                for k in sub_K:
                    if s not in sub_S_k[k]:
                        continue
                    cols = [zw_idx[(i, k, s)] for i in order_idxs if k in K_i[i] and (i, k, s) in zw_idx]
                    if not cols:
                        continue
                    idxs = np.array(cols, dtype=np.int32)
                    vals = np.ones(len(cols), dtype=np.float64)
                    ub = float(inv_remaining.get((s, k), 0))
                    hw.addRow(-highspy.kHighsInf, ub, len(cols), idxs, vals)

            # C3a z_qty[i,k,s] ≤ qty[i,k] · y[s] → -qty[i,k]·y[s] + z_qty ≤ 0
            for i in order_idxs:
                for k in K_i[i]:
                    q = float(sub_qty[(i, k)])
                    for s in sub_S_k[k]:
                        idxs = np.array([yw_idx[s], zw_idx[(i, k, s)]], dtype=np.int32)
                        vals = np.array([-q, 1.0], dtype=np.float64)
                        hw.addRow(-highspy.kHighsInf, 0.0, 2, idxs, vals)

            hw.setMinimize()
            hw.run()
            st_w = hw.getModelStatus().name

            visited: list[str] = []
            pick_qty: dict[tuple[str, str], int] = {}
            per_order_pick_qty: dict[tuple[str, str, str], int] = {}
            shelf_sku_hits: dict[str, set[str]] = {}

            if st_w == "kInfeasible":
                # 库存不够,子波内 simple greedy fallback
                notes.append(f"fallback_greedy_subwave_{w}_inv_insufficient")
                all_optimal = False
                # 简单贪心:每订单每 SKU 找第一个有库存的货架拣,不够则报错(忽略部分拣)
                for i in order_idxs:
                    order_id = orders_by_idx[i].order_id
                    for k in K_i[i]:
                        need = sub_qty[(i, k)]
                        for s in sub_S_k[k]:
                            if need <= 0:
                                break
                            avail = inv_remaining.get((s, k), 0)
                            if avail <= 0:
                                continue
                            take = min(avail, need)
                            if take <= 0:
                                continue
                            pick_qty[(k, s)] = pick_qty.get((k, s), 0) + take
                            per_order_pick_qty[(order_id, k, s)] = take
                            consumption[(s, k)] = consumption.get((s, k), 0) + take
                            inv_remaining[(s, k)] = avail - take
                            need -= take
                            if s not in visited:
                                visited.append(s)
                            shelf_sku_hits.setdefault(s, set()).add(k)
            else:
                col_w = hw.getSolution().col_value
                if st_w != "kOptimal":
                    all_optimal = False
                for i in order_idxs:
                    order_id = orders_by_idx[i].order_id
                    for k in K_i[i]:
                        for s in sub_S_k[k]:
                            v = float(col_w[zw_idx[(i, k, s)]])
                            q_int = int(round(v)) if v >= 0.5 else 0
                            if q_int == 0:
                                continue
                            pick_qty[(k, s)] = pick_qty.get((k, s), 0) + q_int
                            per_order_pick_qty[(order_id, k, s)] = q_int
                            consumption[(s, k)] = consumption.get((s, k), 0) + q_int
                            inv_remaining[(s, k)] = inv_remaining.get((s, k), 0) - q_int
                            if s not in visited:
                                visited.append(s)
                            shelf_sku_hits.setdefault(s, set()).add(k)

            hits = sum(len(sk_set) for sk_set in shelf_sku_hits.values())
            total_hits += hits
            total_visits += len(visited)

            wave_assignments.append(
                WaveAssignment(
                    subwave_idx=w,
                    orders=tuple(orders_by_idx[i] for i in order_idxs),
                    visited_shelves=tuple(visited),
                    pick_qty=pick_qty,
                    per_order_pick_qty=per_order_pick_qty,
                )
            )

        hit_rate = (total_hits / total_visits) if total_visits > 0 else 0.0
        # obj 用阶段 2 各子波 Σ_s y[s] 之和(= total_visits)
        obj_val = float(total_visits)

        # 状态
        if all_optimal and not notes:
            out_status = "optimal"
        elif any("fallback" in n for n in notes):
            out_status = "fallback_greedy"
        else:
            out_status = "time_limit"

        return MIPSolution(
            status=out_status,
            wave_assignments=tuple(wave_assignments),
            total_visits=total_visits,
            hit_rate=hit_rate,
            consumption=consumption,
            objective_value=obj_val,
            mip_gap=None,  # 跨子波 gap 不直接可比,留 None
            solver_name=self.solver_name,
        )

    def _solve_greedy_a(self, inp: MIPInput) -> MIPSolution:
        """两阶段贪心(方案 A 结构,无 MIP):
        阶段 1:对每订单每 SKU,优先选「已用过的货架」凑齐,其次库存多的,
                最小化 used_shelves 全集大小。
        阶段 2:按 R_i 大小降序放,每子波挑「加入后新增访问数最少」的订单,
                最小化 Σ_w |∪_{i∈w} R_i|。

        无 LP 松弛,gap=None;status 用 "optimal" 表示贪心完成(非数学最优)。
        """
        I, orders_by_idx, K_i, S_k, _S, qty, inv_sk, W_sub, _K_per_shelf = self._prep_sets(inp)

        # === 阶段 1:greedy z_qty 分配 ===
        remaining_inv = dict(inv_sk)
        used_shelves: set[str] = set()
        per_order_R: dict[int, set[str]] = {i: set() for i in I}
        z_qty: dict[tuple[int, str, str], int] = {}

        for i in I:
            for k in K_i[i]:
                shelves = S_k[k]
                # 优先选已在 used_shelves 的(s not in used 排前 → False=0 优先),
                # 其次库存多的(-avail 排前 → 多的先)
                sorted_shelves = sorted(
                    shelves,
                    key=lambda s, k=k: (
                        s not in used_shelves,
                        -remaining_inv.get((s, k), 0),
                    ),
                )
                remaining = qty[(i, k)]
                for s in sorted_shelves:
                    if remaining <= 0:
                        break
                    avail = remaining_inv.get((s, k), 0)
                    if avail <= 0:
                        continue
                    take = min(avail, remaining)
                    z_qty[(i, k, s)] = take
                    remaining_inv[(s, k)] = avail - take
                    remaining -= take
                    used_shelves.add(s)
                    per_order_R[i].add(s)

        # === 阶段 2:greedy 组波 ===
        wave_orders: list[list[int]] = [[] for _ in W_sub]
        wave_shelves: list[set[str]] = [set() for _ in W_sub]
        unassigned = sorted(I, key=lambda i: -len(per_order_R[i]))

        for w in W_sub:
            while len(wave_orders[w]) < inp.N_max and unassigned:
                best_i = None
                best_cost = float("inf")
                ws = wave_shelves[w]
                for i in unassigned:
                    cost = len(per_order_R[i] - ws)
                    if cost < best_cost:
                        best_cost = cost
                        best_i = i
                if best_i is None:
                    break
                wave_orders[w].append(best_i)
                ws.update(per_order_R[best_i])
                unassigned.remove(best_i)

        # 防御:unassigned 还有(子波数算错时)塞到最后一波
        if unassigned:
            last_w = W_sub[-1]
            for i in unassigned:
                wave_orders[last_w].append(i)
                wave_shelves[last_w].update(per_order_R[i])

        # === 构造 MIPSolution ===
        wave_assignments: list[WaveAssignment] = []
        consumption: dict[tuple[str, str], int] = {}
        total_hits = 0
        total_visits = 0

        for w in W_sub:
            order_idxs = wave_orders[w]
            orders_in_w = tuple(orders_by_idx[i] for i in order_idxs)
            visited: list[str] = []
            pick_qty: dict[tuple[str, str], int] = {}
            per_order_pick_qty: dict[tuple[str, str, str], int] = {}
            shelf_sku_hits: dict[str, set[str]] = {}

            for i in order_idxs:
                order_id = orders_by_idx[i].order_id
                for k in K_i[i]:
                    for s in S_k[k]:
                        q = z_qty.get((i, k, s), 0)
                        if q == 0:
                            continue
                        pick_qty[(k, s)] = pick_qty.get((k, s), 0) + q
                        per_order_pick_qty[(order_id, k, s)] = q
                        consumption[(s, k)] = consumption.get((s, k), 0) + q
                        if s not in visited:
                            visited.append(s)
                        shelf_sku_hits.setdefault(s, set()).add(k)

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
            status="optimal",  # greedy 完成即 "optimal"(非数学最优,但无 LP bound 可报)
            wave_assignments=tuple(wave_assignments),
            total_visits=total_visits,
            hit_rate=hit_rate,
            consumption=consumption,
            objective_value=float(total_visits),
            mip_gap=None,
            solver_name=self.solver_name,
        )