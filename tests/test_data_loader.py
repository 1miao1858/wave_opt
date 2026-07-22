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
