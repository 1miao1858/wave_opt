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


@dataclass(frozen=True)
class InventorySnapshot:
    snapshot_time: datetime
    # inv[(shelf_id, sku_id)] = qty(可能为 0,诊断用)
    inv: dict[tuple[str, str], int] = field(default_factory=dict)
    # 派生:shelf -> {sku}(qty > 0)
    shelf_skus: dict[str, set[str]] = field(default_factory=dict)
    # 派生:sku -> {shelf}(qty > 0),即 S(k)
    sku_shelves: dict[str, set[str]] = field(default_factory=dict)

    @staticmethod
    def get_snapshot_at(
        snapshots: list["InventorySnapshot"], t: datetime
    ) -> "InventorySnapshot":
        """取 snapshot_time ≤ t 的最近一个;若无,抛错。"""
        candidates = [s for s in snapshots if s.snapshot_time <= t]
        if not candidates:
            raise ValueError(f"无早于 {t} 的库存快照")
        return max(candidates, key=lambda s: s.snapshot_time)


def load_inventory_snapshots(path: Path | str) -> list[InventorySnapshot]:
    """加载 inventory_snapshots.csv,按时刻分组,派生 shelf_skus / sku_shelves。"""
    path = Path(path)
    df = pd.read_csv(path, dtype={"shelf_id": str, "sku_id": str})

    required = {"snapshot_time", "shelf_id", "sku_id", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"inventory_snapshots.csv 缺列:{missing}")

    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    df["qty"] = df["qty"].astype(int)
    df = df.sort_values("snapshot_time")

    snapshots: list[InventorySnapshot] = []
    for snap_time, group in df.groupby("snapshot_time", sort=True):
        inv: dict[tuple[str, str], int] = {}
        shelf_skus: dict[str, set[str]] = {}
        sku_shelves: dict[str, set[str]] = {}
        for r in group.itertuples():
            key = (r.shelf_id, r.sku_id)
            inv[key] = int(r.qty)
            if r.qty > 0:
                shelf_skus.setdefault(r.shelf_id, set()).add(r.sku_id)
                sku_shelves.setdefault(r.sku_id, set()).add(r.shelf_id)
        snapshots.append(
            InventorySnapshot(
                snapshot_time=snap_time.to_pydatetime(),
                inv=inv,
                shelf_skus=shelf_skus,
                sku_shelves=sku_shelves,
            )
        )
    return snapshots
