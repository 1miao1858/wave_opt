from pathlib import Path

import pytest
from openpyxl import load_workbook

from src.config import (
    Config,
    SubProblemConfig,
    SweepConfig,
    MipConfig,
    ValidationConfig,
)
from src.data_loader import load_orders, load_inventory_snapshots
from src.sweep import SweepRunner, SweepResult, WindowResult
from src.output import write_excel


def _make_config(N_max=2):
    return Config(
        sub_problems={
            "fei_jia_gong_le_20": SubProblemConfig(
                wave_type="非加工",
                size_class="le_20",
                N_max=N_max,
            ),
        },
        sweep=SweepConfig(
            num_pickers=4,
            pick_time_per_order=120,
            num_machines=2,
            process_time_per_jian=36,
            K_ws=8,
            daily_cutoff="14:00",
            daily_deadline="18:00",
            picker_shift_start="08:00",
            picker_shift_end="18:00",
            machine_shift_start="08:00",
            machine_shift_end="18:00",
        ),
        candidate_W=["1h"],
        mip=MipConfig(
            solver="gurobi",
            time_limit=60,
            mip_gap=0.05,
            fallback="greedy",
        ),
        validation=ValidationConfig(
            recomputed_hit_rate=True,
            boundary_check=True,
            inventory_consistency_sample=5,
            stockout_filter_sample=10,
        ),
    )


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_case"


def _build_sweep_results():
    orders = load_orders(FIXTURE / "orders.csv")
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)
    runner = SweepRunner(cfg=cfg, all_orders=orders, snapshots=snaps)
    return {sp: runner.run_sub_problem(sp) for sp in cfg.sub_problems}


def test_excel_sheet1_has_summary(tmp_path):
    results = _build_sweep_results()
    out_file = tmp_path / "out.xlsx"
    write_excel(results, out_file)

    wb = load_workbook(out_file)
    assert "Sheet1_汇总" in wb.sheetnames
    ws = wb["Sheet1_汇总"]
    # 表头应有:子问题 | W | 总货架访问数 | 平均命中率 | 命中率方差 | 平均子波规模 | 拣货利用率 | 加工利用率 | 截单可行 | 备注
    headers = [c.value for c in ws[1]]
    assert "子问题" in headers
    assert "总货架访问数" in headers
    assert "平均命中率" in headers
    # 数据行:1 子问题 × 1 W = 1 行
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(data_rows) >= 1
    row = data_rows[0]
    assert row[0] == "fei_jia_gong_le_20"
    assert isinstance(row[2], (int, float))  # 总货架访问数
    assert isinstance(row[3], (int, float))  # 平均命中率


def test_excel_sheet2_has_visits_curve(tmp_path):
    results = _build_sweep_results()
    out_file = tmp_path / "out.xlsx"
    write_excel(results, out_file)

    wb = load_workbook(out_file)
    assert "Sheet2_总访问数曲线" in wb.sheetnames
    ws = wb["Sheet2_总访问数曲线"]
    # 表头:子问题 | W | 全日总货架访问数
    headers = [c.value for c in ws[1]]
    assert "子问题" in headers
    assert "W" in headers
    assert "全日总货架访问数" in headers
    # 至少一行数据(tiny_case 至少 1 个子问题 × 1 个 W)
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(data_rows) >= 1


def test_excel_sheet3_has_hit_rate_and_utilization(tmp_path):
    results = _build_sweep_results()
    out_file = tmp_path / "out.xlsx"
    write_excel(results, out_file)

    wb = load_workbook(out_file)
    assert "Sheet3_命中率利用率" in wb.sheetnames
    ws = wb["Sheet3_命中率利用率"]
    headers = [c.value for c in ws[1]]
    # 应含:子问题 | W | 平均命中率 | 拣货利用率 | 加工利用率
    assert "子问题" in headers
    assert "平均命中率" in headers
    assert "拣货利用率" in headers
    assert "加工利用率" in headers
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(data_rows) >= 1


def test_excel_sheet4_has_wave_details(tmp_path):
    results = _build_sweep_results()
    out_file = tmp_path / "out.xlsx"
    write_excel(results, out_file)

    wb = load_workbook(out_file)
    assert "Sheet4_波次明细" in wb.sheetnames
    ws = wb["Sheet4_波次明细"]
    headers = [c.value for c in ws[1]]
    expected_cols = [
        "波次ID",
        "子问题",
        "窗口ID",
        "子波序号",
        "触发时刻",
        "窗口时段",
        "订单数",
        "订单列表",
        "访问货架数",
        "hits",
        "命中率",
        "库存消耗明细",
    ]
    for col in expected_cols:
        assert col in headers, f"缺列:{col}"
    # tiny_case 1 窗口 × 2 子波 → 至少 2 行
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(data_rows) >= 2


def test_excel_sheet5_has_anomalies(tmp_path):
    results = _build_sweep_results()
    out_file = tmp_path / "out.xlsx"
    write_excel(results, out_file)

    wb = load_workbook(out_file)
    assert "Sheet5_异常诊断" in wb.sheetnames
    ws = wb["Sheet5_异常诊断"]
    headers = [c.value for c in ws[1]]
    # 应含:类别 | 子问题 | 窗口 | 详情
    assert "类别" in headers
    assert "详情" in headers
    # 数据行可能为空(tiny_case 无异常),但表头应存在
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    # 不强制要求行数(tiny_case 应无异常)


def test_excel_sheet5_records_stockout_rejection(tmp_path):
    """构造一个有缺货剔除的场景:Sheet 5 应记录。"""
    import datetime

    from src.data_loader import Order, OrderLine

    orders = list(load_orders(FIXTURE / "orders.csv")) + [
        Order(
            order_id="STOCKOUT",
            timestamp=datetime.datetime(2026, 7, 1, 14, 0),
            wave_type="非加工",
            lines=(OrderLine(sku_id="K_MISSING", qty=1),),
            件数=1,
            size_class="le_20",
            sub_problem_key=("非加工", "le_20"),
        ),
    ]
    snaps = load_inventory_snapshots(FIXTURE / "inventory_snapshots.csv")
    cfg = _make_config(N_max=2)
    runner = SweepRunner(cfg=cfg, all_orders=orders, snapshots=snaps)
    results = {sp: runner.run_sub_problem(sp) for sp in cfg.sub_problems}

    out_file = tmp_path / "out.xlsx"
    write_excel(results, out_file)

    wb = load_workbook(out_file)
    ws = wb["Sheet5_异常诊断"]
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    # 应有至少一行:缺货剔除
    stockout_rows = [r for r in data_rows if r[0] == "缺货剔除"]
    assert len(stockout_rows) >= 1
    assert any("STOCKOUT" in str(r[3]) for r in stockout_rows)
