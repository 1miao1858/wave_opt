"""组波优化 POC — 真实数据 MIP demo(可移植版)。

用法:
    # 在 POC 项目根目录下
    python scripts/demo_real_data.py \
        --orders /path/to/作业_2026-07-19.xlsx \
        --inventory /path/to/库存_2026-07-19_日报.xlsx \
        --output benchmark.csv

    # 只跑 Gurobi + spec 3.7 场景
    python scripts/demo_real_data.py --solver gurobi --scenarios 200x50

    # 跑全部默认场景,两 solver 对比
    python scripts/demo_real_data.py --solver both

默认场景(可用 --scenarios 覆盖):
    10x5, 20x5, 20x10, 30x10, 50x10, 50x5, 100x10, 100x50, 200x50(spec 3.7), 500x50

输出:CSV 表格 [solver, n_orders, N_max, n_sub, n_vars, time_s, status, visits, hit, gap, obj]

临时假设(业务确认前,见 docs/business_questions_open.md):
- 全部当「非加工」(5 天数据全是普通出库)
- 货架粒度:上级容器编码(strip 来源容器编码 末尾 -XX)
- 时间戳:创建日期
- 件数:分配件数
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.data_loader import InventorySnapshot, Order, OrderLine
from src.mip_solver import HAS_GUROBI, HAS_SCIP, JointMIPSolver, MIPInput

DEFAULT_SCENARIOS = [
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


def parse_args():
    p = argparse.ArgumentParser(description="组波优化 POC — 真实数据 MIP demo")
    p.add_argument(
        "--orders",
        required=True,
        help="作业数据 xlsx 路径(如 作业_2026-07-19.xlsx)",
    )
    p.add_argument(
        "--inventory",
        required=True,
        help="库存数据 xlsx 路径(如 库存_2026-07-19_日报.xlsx)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径;不指定则只打印到 stdout",
    )
    p.add_argument(
        "--solver",
        choices=["gurobi", "scip", "both"],
        default="both",
        help="求解器选择(默认 both)",
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help="场景列表,格式 '10x5,20x10,200x50';默认跑全部 10 个",
    )
    p.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="单次求解时间上限(秒,默认 120)",
    )
    p.add_argument(
        "--mip-gap",
        type=float,
        default=0.05,
        help="MIP gap 阈值(默认 0.05)",
    )
    p.add_argument(
        "--snapshot-time",
        default="2026-07-19 00:00:00",
        help="库存快照时间戳(默认 2026-07-19 00:00:00)",
    )
    return p.parse_args()


def parse_scenarios(s: str | None) -> list[tuple[int, int]]:
    if not s:
        return DEFAULT_SCENARIOS
    out = []
    for item in s.split(","):
        n, N = item.split("x")
        out.append((int(n), int(N)))
    return out


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
    """估计变量数/二元数/整数数(不构造模型)。"""
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


def run_one(orders, inv, N_max, time_limit, solver_name, mip_gap):
    inp = MIPInput(
        window_orders=tuple(orders),
        inv=inv,
        N_max=N_max,
        time_limit=time_limit,
    )
    n_var, n_bin, n_int = count_vars_constraints(inp)
    solver = JointMIPSolver(solver_name=solver_name, mip_gap=mip_gap)
    t0 = time.perf_counter()
    sol = solver.solve(inp)
    elapsed = time.perf_counter() - t0
    return {
        "solver": solver_name,
        "n_orders": len(orders),
        "N_max": N_max,
        "n_sub": max(1, (len(orders) + N_max - 1) // N_max),
        "n_vars": n_var,
        "n_binary": n_bin,
        "n_integer": n_int,
        "time_s": round(elapsed, 3),
        "status": sol.status,
        "visits": sol.total_visits,
        "hit_rate": round(sol.hit_rate, 4),
        "gap": round(sol.mip_gap, 4) if sol.mip_gap is not None else "",
        "obj": round(sol.objective_value, 2),
    }


def main():
    args = parse_args()
    orders_path = Path(args.orders)
    inv_path = Path(args.inventory)
    if not orders_path.exists():
        print(f"ERROR: 作业数据文件不存在: {orders_path}", file=sys.stderr)
        sys.exit(1)
    if not inv_path.exists():
        print(f"ERROR: 库存数据文件不存在: {inv_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 100)
    print("Step 1: 加载数据")
    print("=" * 100)
    orders_all = load_orders_from_xlsx(orders_path)
    inv = load_inventory_from_xlsx(
        inv_path,
        snapshot_time=datetime.strptime(args.snapshot_time, "%Y-%m-%d %H:%M:%S"),
    )
    print(f"  作业文件: {orders_path.name}")
    print(f"  库存文件: {inv_path.name}")
    print(f"  全量订单: {len(orders_all)}")
    print(f"  库存货架: {len(inv.shelf_skus)}")
    print(f"  库存 SKU: {len(inv.sku_shelves)}")

    orders_ok = filter_fulfillable(orders_all, inv)
    print(f"  可履约订单(SKU 全在库存): {len(orders_ok)} ({len(orders_ok)/max(1,len(orders_all))*100:.1f}%)")
    orders_ok = sorted(orders_ok, key=lambda o: o.order_id)

    scenarios = parse_scenarios(args.scenarios)
    solvers = ["gurobi", "scip"] if args.solver == "both" else [args.solver]

    print()
    print("=" * 100)
    print(f"Step 2: 跑 {len(scenarios)} 个场景 × {len(solvers)} 个 solver(time_limit={args.time_limit}s, gap={args.mip_gap})")
    print("=" * 100)
    header = (f"{'solver':>7} {'n_ord':>6} {'N_max':>5} {'n_sub':>5} {'n_vars':>8} "
              f"{'time_s':>8} {'status':>10} {'visits':>7} {'hit':>6} {'gap':>7} {'obj':>7}")
    print(header)
    print("-" * 100)

    results = []
    for solver_name in solvers:
        if solver_name == "gurobi" and not HAS_GUROBI:
            print(f"  [skip] gurobi 未安装")
            continue
        if solver_name == "scip" and not HAS_SCIP:
            print(f"  [skip] scip 未安装")
            continue
        for n_orders, N_max in scenarios:
            sample = orders_ok[:n_orders]
            if len(sample) < n_orders:
                print(f"  [skip] {solver_name} {n_orders}x{N_max}:样本不足(只有 {len(sample)} 单)")
                continue
            try:
                r = run_one(sample, inv, N_max, args.time_limit, solver_name, args.mip_gap)
                results.append(r)
                gap_str = f"{r['gap']:.4f}" if r['gap'] != "" else "-"
                print(f"{r['solver']:>7} {r['n_orders']:>6} {r['N_max']:>5} {r['n_sub']:>5} "
                      f"{r['n_vars']:>8} {r['time_s']:>8.2f} {r['status']:>10} {r['visits']:>7} "
                      f"{r['hit_rate']:>6.3f} {gap_str:>7} {r['obj']:>7.0f}")
            except Exception as e:
                print(f"  [FAIL] {solver_name} {n_orders}x{N_max}: {type(e).__name__}: {e}")
                results.append({
                    "solver": solver_name, "n_orders": n_orders, "N_max": N_max,
                    "n_sub": (n_orders + N_max - 1)//N_max, "n_vars": "", "n_binary": "",
                    "n_integer": "", "time_s": "", "status": f"FAIL:{type(e).__name__}",
                    "visits": "", "hit_rate": "", "gap": "", "obj": "",
                })

    if args.output:
        out_path = Path(args.output)
        fieldnames = ["solver", "n_orders", "N_max", "n_sub", "n_vars", "n_binary",
                      "n_integer", "time_s", "status", "visits", "hit_rate", "gap", "obj"]
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\n[输出] {out_path.absolute()}")

    print()
    print("=" * 100)
    print("Step 3: 汇总")
    print("=" * 100)
    print(f"  求解器: {solvers}")
    print(f"  场景数: {len(scenarios)}")
    print(f"  时间上限: {args.time_limit}s")
    print(f"  Gap 阈值: {args.mip_gap}")
    print(f"  总跑通: {sum(1 for r in results if r['status'] in ('optimal','time_limit'))}/{len(results)}")
    print(f"  最优解(optimal): {sum(1 for r in results if r['status']=='optimal')}")
    print(f"  license 卡 / 失败: {sum(1 for r in results if 'FAIL' in str(r['status']) or 'LICENSE' in str(r['status']))}")


if __name__ == "__main__":
    main()
