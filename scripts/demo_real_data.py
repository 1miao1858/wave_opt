"""用真实 7-19 数据跑 MIP demo:Gurobi vs SCIP 对比。

临时假设(业务确认前):
- A1 加工/非加工:全部当「非加工」(5 天数据全是普通出库)
- B1 货架粒度:上级容器编码(strip 来源容器编码 末尾 -XX)
- B2 时间戳:创建日期
- B3 件数:分配件数
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.data_loader import InventorySnapshot, Order, OrderLine
from src.mip_solver import JointMIPSolver, MIPInput

DATA = Path("/Users/admin/蔡司项目数据")


def load_orders_from_xlsx(path: Path) -> list[Order]:
    df = pd.read_excel(path)
    df = df[df["任务类型名称"] == "普通出库"].copy()
    df["创建日期"] = pd.to_datetime(df["创建日期"])
    df["分配件数"] = df["分配件数"].fillna(0).astype(int)
    orders: list[Order] = []
    for order_id, g in df.groupby("订单号", sort=False):
        first = g.iloc[0]
        lines = tuple(
            OrderLine(sku_id=str(r.商品编码), qty=int(r.分配件数))
            for r in g.itertuples()
        )
        orders.append(
            Order(
                order_id=str(order_id),
                timestamp=first["创建日期"].to_pydatetime(),
                wave_type="非加工",
                lines=lines,
            )
        )
    return orders


def load_inventory_from_xlsx(path: Path, snapshot_time: datetime) -> InventorySnapshot:
    df = pd.read_excel(path, usecols=["上级容器编码", "商品编码", "可用数量"])
    df = df.rename(columns={"上级容器编码": "shelf", "商品编码": "sku", "可用数量": "qty"})
    df["qty"] = df["qty"].fillna(0).astype(int)
    df = df[df["qty"] > 0]

    inv: dict[tuple[str, str], int] = {}
    shelf_skus: dict[str, set[str]] = {}
    sku_shelves: dict[str, set[str]] = {}
    for r in df.itertuples():
        key = (r.shelf, r.sku)
        inv[key] = inv.get(key, 0) + int(r.qty)
        shelf_skus.setdefault(r.shelf, set()).add(r.sku)
        sku_shelves.setdefault(r.sku, set()).add(r.shelf)
    return InventorySnapshot(
        snapshot_time=snapshot_time,
        inv=inv,
        shelf_skus=shelf_skus,
        sku_shelves=sku_shelves,
    )


def filter_fulfillable(orders: list[Order], inv: InventorySnapshot) -> list[Order]:
    out = []
    for o in orders:
        if all(
            any((s, l.sku_id) in inv.inv for s in inv.sku_shelves.get(l.sku_id, set()))
            for l in o.lines
        ):
            out.append(o)
    return out


def count_vars_constraints(inp: MIPInput) -> tuple[int, int, int]:
    I = list(range(len(inp.window_orders)))
    K_i = {i: sorted({l.sku_id for l in inp.window_orders[i].lines}) for i in I}
    S_k: dict[str, list[str]] = {}
    for k in {k for ks in K_i.values() for k in ks}:
        S_k[k] = sorted(inp.inv.sku_shelves.get(k, set()))
    S = sorted({s for shelves in S_k.values() for s in shelves})
    n_sub = max(1, (len(I) + inp.N_max - 1) // inp.N_max)
    n_x = len(I) * n_sub
    n_z = sum(len(S_k[k]) for i in I for k in K_i[i])
    n_y = len(S) * n_sub
    K_per_shelf = {s: [k for k, shelves in S_k.items() if s in shelves] for s in S}
    n_h = sum(len(K_per_shelf[s]) for s in S) * n_sub
    return n_x + n_z + n_y + n_h, n_x + n_y + n_h, n_z


def run_one(orders, inv, N_max, time_limit, solver_name="scip"):
    inp = MIPInput(
        window_orders=tuple(orders),
        inv=inv,
        N_max=N_max,
        M_big=max((l.qty for o in orders for l in o.lines), default=1),
        time_limit=time_limit,
    )
    n_var, n_bin, n_int = count_vars_constraints(inp)
    solver = JointMIPSolver(solver_name=solver_name, mip_gap=0.05)
    t0 = time.perf_counter()
    sol = solver.solve(inp)
    elapsed = time.perf_counter() - t0
    return {
        "n_orders": len(orders),
        "N_max": N_max,
        "n_subwaves": max(1, (len(orders) + N_max - 1) // N_max),
        "n_vars": n_var,
        "n_binary": n_bin,
        "n_integer": n_int,
        "elapsed_s": elapsed,
        "status": sol.status,
        "total_visits": sol.total_visits,
        "hit_rate": sol.hit_rate,
        "objective": sol.objective_value,
        "mip_gap": sol.mip_gap,
        "solver": solver_name,
    }


def print_row(r, show_gap=False):
    gap = f"{r['mip_gap']:.4f}" if r.get('mip_gap') is not None else "-"
    line = (f"{r['solver']:>7} {r['n_orders']:>7} {r['N_max']:>5} {r['n_subwaves']:>5} "
            f"{r['n_vars']:>8} {r['elapsed_s']:>8.2f} {r['status']:>10} "
            f"{r['total_visits']:>7} {r['hit_rate']:>6.3f}")
    if show_gap:
        line += f" {gap:>7}"
    print(line)


def main():
    print("=" * 100)
    print("Step 1: 加载 7-19 数据")
    print("=" * 100)
    orders_all = load_orders_from_xlsx(DATA / "作业数据" / "作业_2026-07-19.xlsx")
    inv = load_inventory_from_xlsx(
        DATA / "库存数据" / "库存_2026-07-19_日报.xlsx",
        snapshot_time=datetime(2026, 7, 19, 0, 0, 0),
    )
    print(f"  全量订单: {len(orders_all)}")
    print(f"  库存货架: {len(inv.shelf_skus)}")
    print(f"  库存 SKU: {len(inv.sku_shelves)}")

    orders_ok = filter_fulfillable(orders_all, inv)
    print(f"  可履约订单(SKU 全在库存): {len(orders_ok)} ({len(orders_ok)/len(orders_all)*100:.1f}%)")
    orders_ok = sorted(orders_ok, key=lambda o: o.order_id)

    # === Gurobi 小规模(baseline) ===
    print("\n" + "=" * 100)
    print("Step 2a: Gurobi 跑小规模(restricted license ≤2000 变量)")
    print("=" * 100)
    print(f"{'solver':>7} {'n_ord':>7} {'N_max':>5} {'n_sub':>5} {'n_vars':>8} {'time_s':>8} {'status':>10} {'visits':>7} {'hit':>6}")
    print("-" * 90)
    for n_orders, N_max in [(10, 5), (20, 5), (20, 10), (30, 10)]:
        sample = orders_ok[:n_orders]
        try:
            r = run_one(sample, inv, N_max=N_max, time_limit=60, solver_name="gurobi")
            print_row(r)
        except Exception as e:
            print(f"{'gurobi':>7} {n_orders:>7} {N_max:>5} {(n_orders + N_max - 1)//N_max:>5}  FAIL: {type(e).__name__}")

    # === SCIP 全规模 ===
    print("\n" + "=" * 100)
    print("Step 2b: SCIP 跑全规模(无 license 限制)")
    print("=" * 100)
    print(f"{'solver':>7} {'n_ord':>7} {'N_max':>5} {'n_sub':>5} {'n_vars':>8} {'time_s':>8} {'status':>10} {'visits':>7} {'hit':>6} {'gap':>7}")
    print("-" * 100)
    scip_cases = [
        (10, 5),
        (20, 5),
        (20, 10),
        (30, 10),
        (50, 10),
        (50, 5),
        (100, 10),
        (100, 50),
        (200, 50),  # spec 3.7 场景
        (500, 50),
    ]
    for n_orders, N_max in scip_cases:
        sample = orders_ok[:n_orders]
        try:
            r = run_one(sample, inv, N_max=N_max, time_limit=120, solver_name="scip")
            print_row(r, show_gap=True)
        except Exception as e:
            print(f"{'scip':>7} {n_orders:>7} {N_max:>5} {(n_orders + N_max - 1)//N_max:>5}  FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
