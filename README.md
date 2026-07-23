# 组波优化 POC

> 货到人系统组波优化 POC:离线频率扫描 + Joint MIP 求解 + Excel 输出。
> 设计 spec: `docs/superpowers/specs/2026-07-22-组波优化-poc-design.md`
> 实施计划: `docs/superpowers/plans/2026-07-22-组波优化-poc.md`

## 安装

```bash
cd 组波优化-POC
pip install -e ".[dev]"
```

需要 Gurobi 学术/商业 license。若拿不到:

1. 改 `config.yaml` 的 `mip.solver` 为 `scip` 或 `cbc`
2. 或在 `src/mip_solver.py` 顶部替换 gurobipy 为 pyscipopt

## 准备数据

把数据放到 `data/` 目录:

- `orders.csv`:订单长格式(order_id, timestamp, wave_type, sku_id, qty)
- `inventory_snapshots.csv`:每小时整点库存快照(snapshot_time, shelf_id, sku_id, qty)
- `sku_master.csv`(可选):SKU 主数据,仅诊断用

参数填入 `config.yaml`(子问题 N_max、sweep 产能、MIP 配置)。待补参数见 spec 第 10 节。

## 运行

```bash
# 完整扫描
python -m src.main --config config.yaml --data-dir data/ --output output/

# 只跑某子问题(调试)
python -m src.main --config config.yaml --data-dir data/ --output output/ \
    --subproblem fei_jia_gong_le_20
```

输出:`output/poc_results_<timestamp>.xlsx`,含 5 个 Sheet:

| Sheet | 内容 |
|---|---|
| Sheet1_汇总 | 每子问题 × 每 W 的总访问数 + 平均命中率 + 利用率 + 截单可行;最优 W 高亮 |
| Sheet2_总访问数曲线 | 每子问题一条曲线,X=W, Y=全日总访问数 |
| Sheet3_命中率利用率 | 平均命中率 + 拣货/加工利用率 |
| Sheet4_波次明细 | 每子波一行:订单列表、访问货架、hits、命中率、库存消耗 |
| Sheet5_异常诊断 | 缺货剔除、不可行窗口、MIP 超时 fallback、gap 超阈、子波拆分过多 |

## 测试

```bash
pytest                       # 全部
pytest tests/test_mip_solver.py -v   # MIP + 6.1 手算对比
pytest tests/test_validation.py -v   # 6.2-6.9 验证检查
pytest tests/test_smoke.py -v        # 端到端 smoke
```

## 核心算法

**Joint MIP**(每窗口一次):同时决定

- 子波组成:`x[i,w]`(订单 i 分到子波 w)
- 货架分配:`z_qty[i,k,s]`(订单 i 的 SKU k 从货架 s 拣的件数)、`y[s,w]`(子波 w 访问货架 s)

约束:

- C1' 履行(可跨货架拆分):`Σ_s z_qty[i,k,s] = qty[i,k]`
- C2 库存(跨子波共享):`Σ_i z_qty[i,k,s] ≤ inv[s,k]`
- C3 链接(含 `(1-x[i,w])` 隐式按子波关联):`z_qty[i,k,s] ≤ M_big × y[s,w] + M_big × (1 - x[i,w])`
- C4 子波组成:`Σ_w x[i,w] = 1`
- C5 单子波上限:`Σ_i x[i,w] ≤ N_max`

目标:`min Σ_{w,s} y[s,w]`(线性,最小化总货架访问数;跨子波相加 = 全日总访问数)

命中率降为报告指标(从 `h[s,k,w]` 推算),不进目标——避免分数规划复杂度和"无意义拆分"陷阱。

## 验证检查(自动)

| 检查 | 频率 |
|---|---|
| 6.1 MIP 正确性自验(合成小样本手算对比) | 开发期一次 |
| 6.2 命中率独立复算 | 每子波自动 |
| 6.5 边界合理性检查 | 每子波自动 |
| 6.7 总访问数独立复算 | 每窗口自动 |
| 6.8a 子波内拆分检查 | 每窗口自动 |
| 6.8b 跨子波拆分检查 | 每窗口自动 |
| 6.3 库存一致性 | 抽样 5 窗口 |
| 6.4 缺货剔除核对 | 抽样 10 订单 |
| 6.9 MIP gap 监控 | 每窗口自动 |

## 不在 POC 范围

- 实时生产运行(滚动批量、WMS API 推送、实时库存)
- 跨窗口延迟订单队列(POC 简化:窗口订单要么进子波要么缺货剔除)
- 库存分布/货架重排优化(独立后续项目)

详见 spec 第 1.4 和第 9 节。
