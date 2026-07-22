"""数据加载 + 派生字段。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

_VALID_WAVE_TYPES = {"加工", "非加工"}
_SIZE_BOUNDARY = 20  # ≤20 / >20 分界


def _size_class(jian_shu: int) -> str:
    return "le_20" if jian_shu <= _SIZE_BOUNDARY else "gt_20"


@dataclass(frozen=True)
class OrderLine:
    sku_id: str
    qty: int


@dataclass(frozen=True)
class Order:
    order_id: str
    timestamp: datetime
    wave_type: str  # "加工" / "非加工"
    lines: tuple[OrderLine, ...]
    件数: int = field(default=0)  # sum(qty)
    size_class: str = field(default="")  # "le_20" / "gt_20"
    sub_problem_key: tuple[str, str] = field(default=("", ""))  # (wave_type, size_class)


def load_orders(path: Path | str) -> list[Order]:
    """加载 orders.csv(长格式),派生 件数、size_class、sub_problem_key。"""
    path = Path(path)
    df = pd.read_csv(path, dtype={"order_id": str, "sku_id": str, "wave_type": str})

    required = {"order_id", "timestamp", "wave_type", "sku_id", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"orders.csv 缺列:{missing}")

    invalid = set(df["wave_type"].unique()) - _VALID_WAVE_TYPES
    if invalid:
        raise ValueError(f"wave_type 非法值:{invalid}(允许:加工/非加工)")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["qty"] = df["qty"].astype(int)

    orders: list[Order] = []
    for order_id, group in df.groupby("order_id", sort=False):
        first = group.iloc[0]
        lines = tuple(
            OrderLine(sku_id=str(r.sku_id), qty=int(r.qty))
            for r in group.itertuples()
        )
        jian_shu = sum(l.qty for l in lines)
        sc = _size_class(jian_shu)
        orders.append(
            Order(
                order_id=str(order_id),
                timestamp=first["timestamp"].to_pydatetime(),
                wave_type=str(first["wave_type"]),
                lines=lines,
                件数=jian_shu,
                size_class=sc,
                sub_problem_key=(str(first["wave_type"]), sc),
            )
        )
    return orders
