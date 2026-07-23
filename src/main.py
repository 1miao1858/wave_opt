"""POC 入口:读 config + 数据 → 跑 sweep → 输出 Excel。"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.data_loader import load_inventory_snapshots, load_orders
from src.output import write_excel
from src.sweep import SweepRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="组波优化 POC 主入口")
    p.add_argument("--config", required=True, help="config.yaml 路径")
    p.add_argument(
        "--data-dir",
        required=True,
        help="数据目录(含 orders.csv, inventory_snapshots.csv)",
    )
    p.add_argument("--output", required=True, help="输出 xlsx 路径")
    p.add_argument(
        "--subproblem",
        default=None,
        help="只跑某个子问题(调试用);默认跑全部",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config)
    orders = load_orders(Path(args.data_dir) / "orders.csv")
    snapshots = load_inventory_snapshots(
        Path(args.data_dir) / "inventory_snapshots.csv"
    )

    runner = SweepRunner(cfg=cfg, all_orders=orders, snapshots=snapshots)

    sub_problems = (
        [args.subproblem] if args.subproblem else list(cfg.sub_problems.keys())
    )
    results = {sp: runner.run_sub_problem(sp) for sp in sub_problems}

    # 输出文件名:若 output 是目录则加时间戳
    out_path = Path(args.output)
    if out_path.is_dir():
        out_path = (
            out_path
            / f"poc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

    write_excel(results, out_path)
    print(f"OK POC 跑完,输出:{out_path}")

    # 控制台汇总
    for sp, result in results.items():
        if result.best_W:
            print(
                f"  {sp}: best_W={result.best_W}, "
                f"total_visits={result.best_total_visits}"
            )
        else:
            print(
                f"  {sp}: 无可行 W(所有候选均超产能/时效)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
