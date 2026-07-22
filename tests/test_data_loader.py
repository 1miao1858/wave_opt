from pathlib import Path
import pytest
from src.data_loader import Order, load_orders


def test_load_orders_basic(tmp_path):
    csv_content = """order_id,timestamp,wave_type,sku_id,qty
O001,2026-07-01 14:23:11,加工,9.B582.0300.0000,2
O001,2026-07-01 14:23:11,加工,9.B582.0275.0000,2
O002,2026-07-01 14:25:33,非加工,9.GVZ2.0150.0000,1
O003,2026-07-01 14:30:00,加工,9.X001.0001.0000,25
"""
    p = tmp_path / "orders.csv"
    p.write_text(csv_content, encoding="utf-8")

    orders = load_orders(p)

    assert len(orders) == 3
    o001 = next(o for o in orders if o.order_id == "O001")
    assert o001.wave_type == "加工"
    assert o001.件数 == 4  # 2+2
    assert o001.size_class == "le_20"  # 4 ≤ 20
    assert o001.sub_problem_key == ("加工", "le_20")
    assert len(o001.lines) == 2
    assert o001.lines[0].sku_id == "9.B582.0300.0000"
    assert o001.lines[0].qty == 2

    o003 = next(o for o in orders if o.order_id == "O003")
    assert o003.件数 == 25
    assert o003.size_class == "gt_20"
    assert o003.sub_problem_key == ("加工", "gt_20")


def test_load_orders_invalid_wave_type(tmp_path):
    csv_content = """order_id,timestamp,wave_type,sku_id,qty
O001,2026-07-01 14:23:11,其他,9.B582.0300.0000,2
"""
    p = tmp_path / "orders.csv"
    p.write_text(csv_content, encoding="utf-8")

    with pytest.raises(ValueError, match="wave_type"):
        load_orders(p)


def test_load_orders_missing_column(tmp_path):
    csv_content = """order_id,timestamp,sku_id,qty
O001,2026-07-01 14:23:11,9.B582.0300.0000,2
"""
    p = tmp_path / "orders.csv"
    p.write_text(csv_content, encoding="utf-8")

    with pytest.raises(ValueError, match="wave_type"):
        load_orders(p)


from src.data_loader import InventorySnapshot, load_inventory_snapshots


def test_load_inventory_snapshots_basic(tmp_path):
    csv_content = """snapshot_time,shelf_id,sku_id,qty
2026-07-01 14:00:00,S001,9.B582.0300.0000,15
2026-07-01 14:00:00,S001,9.B582.0275.0000,8
2026-07-01 14:00:00,S002,9.B582.0300.0000,3
2026-07-01 15:00:00,S001,9.B582.0300.0000,11
"""
    p = tmp_path / "inv.csv"
    p.write_text(csv_content, encoding="utf-8")

    snapshots = load_inventory_snapshots(p)

    assert len(snapshots) == 2  # 14:00 和 15:00
    snap_14 = next(s for s in snapshots if s.snapshot_time.hour == 14)
    assert snap_14.snapshot_time.hour == 14
    # inv[(shelf, sku)] = qty
    assert snap_14.inv[("S001", "9.B582.0300.0000")] == 15
    assert snap_14.inv[("S002", "9.B582.0300.0000")] == 3
    # 货架-SKU 映射派生(qty > 0 视为在架)
    assert "9.B582.0300.0000" in snap_14.shelf_skus["S001"]
    assert "9.B582.0275.0000" in snap_14.shelf_skus["S001"]
    # shelf -> set of SKUs
    assert snap_14.shelf_skus["S002"] == {"9.B582.0300.0000"}
    # sku -> set of shelves(S(k))
    assert snap_14.sku_shelves["9.B582.0300.0000"] == {"S001", "S002"}


def test_load_inventory_zero_qty_not_in_mapping(tmp_path):
    """qty=0 不进 shelf-SKU 映射(视为不在架)。"""
    csv_content = """snapshot_time,shelf_id,sku_id,qty
2026-07-01 14:00:00,S001,9.B582.0300.0000,0
2026-07-01 14:00:00,S001,9.B582.0275.0000,5
"""
    p = tmp_path / "inv.csv"
    p.write_text(csv_content, encoding="utf-8")

    snapshots = load_inventory_snapshots(p)
    snap = snapshots[0]
    assert "9.B582.0300.0000" not in snap.shelf_skus["S001"]
    assert "9.B582.0275.0000" in snap.shelf_skus["S001"]
    # inv 仍记录 0(便于诊断),但映射派生时排除
    assert snap.inv[("S001", "9.B582.0300.0000")] == 0


def test_load_inventory_get_snapshot_at(tmp_path):
    """按整点取最近一个 ≤ 指定时刻的快照。"""
    csv_content = """snapshot_time,shelf_id,sku_id,qty
2026-07-01 13:00:00,S001,K1,5
2026-07-01 14:00:00,S001,K1,3
2026-07-01 15:00:00,S001,K1,1
"""
    p = tmp_path / "inv.csv"
    p.write_text(csv_content, encoding="utf-8")

    snapshots = load_inventory_snapshots(p)
    from datetime import datetime
    snap_at_14_30 = InventorySnapshot.get_snapshot_at(
        snapshots, datetime(2026, 7, 1, 14, 30)
    )
    assert snap_at_14_30.snapshot_time.hour == 14  # 取最近的 ≤ 14:30
