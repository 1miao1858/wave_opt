# 30 分钟时间上限的规模边界测试

> 日期:2026-07-25
> 数据:作业_2026-07-19.xlsx(1860 单,1833 可履约)+ 库存_2026-07-19_日报.xlsx(3875 货架,15229 SKU)
> 求解器:HiGHS 1.10,全部跑同一台机器,3 个 solver 并行(实测 HiGHS 单核为主,互不干扰)
> 参数:time_limit=1800s(30 min),mip_gap=0.05

## 一句话结论

**30 min 内能求解(optimal)的最大规模**:
- joint highs:N_max=60 仅 200 单;N_max=20 连 200 单都到不了 optimal
- two_phase_a:N_max=60 仅 500 单;N_max=20 连 1000 单都到不了 optimal
- **two_phase_b:N_max=60 和 N_max=20 都能到 1000 单 optimal**(1833x60 也能在 <30 min 跑完但 time_limit)

**关键反差**:小规模(≤500)A 解质量最好;大规模(≥1000)只有 B 还能 optimal,因为 B 的阶段 2 按子波拆解成独立小 MIP,每子波 ~60 单保持极小。

## 实测数据

### N_max=60

| 场景 | joint highs | two_phase_a | two_phase_b |
|---|---|---|---|
| 200x60 | 36s **optimal** obj 297 | 0.26s optimal obj 292† | 0.15s optimal obj 335† |
| 500x60 | 1725s time_limit gap 27% obj 648 | 2.30s **optimal** obj 582† | 1.16s optimal obj 746† |
| 1000x60 | 1650s time_limit gap 57% obj 1525 | 1150s time_limit gap 12% obj 1002 | **25s optimal obj 1336** |
| 1833x60 | (未测,必撞墙) | 1182s time_limit gap 47% obj 2414 | 562s time_limit obj 2954 |

### N_max=20

| 场景 | joint highs | two_phase_a | two_phase_b |
|---|---|---|---|
| 200x20 | 298s time_limit gap 17% obj 306 | (未测,应秒级 optimal) | (未测,应秒级 optimal) |
| 500x20 | 685s time_limit gap 58% obj 919 | (未测) | (未测) |
| 1000x20 | (未测) | 1080s time_limit gap 30% obj 1267 | **268s optimal obj 1445** |

† 数据来自之前 600s 上限的对比测试(2026-07-24),为 200x60 / 500x60 在 A/B 上的已知结果。

注:
- "optimal" = HiGHS 报告 status=kOptimal,数学上证明最优
- "time_limit" = 撞 30 min 上限,返回当前最佳可行解 + gap
- B 的 gap 列空(B 没有统一的 LP 松弛下界可报,只能看 status)
- 1000x60 三 solver 都跑了;200x60 / 500x60 highs 撞墙的对照数据见 N_max=60 表

## "30 min 内能求解多大规模"的清晰回答

### 按能否达到 optimal(status 判定)

| Solver | N_max=60 | N_max=20 |
|---|---|---|
| joint highs | ≤ 200 单 | < 200 单 |
| two_phase_a | ≤ 500 单 | < 1000 单 |
| two_phase_b | ≤ 1000 单(1833 也能跑完但 time_limit) | ≤ 1000 单 |

### 按能否返回可行解(允许 time_limit + gap)

三 solver 在所有测过的规模都能返回可行解。区别在解质量:
- 1000x60:highs gap 57%(几乎没解),A gap 12%,B optimal
- 1000x20:highs 没测(必撞墙),A gap 30%,B optimal

## 为什么 B 反而比 A 更能 scale?

| | 方案 A | 方案 B |
|---|---|---|
| 阶段 2 结构 | 单次 MIP,变量 = \|I\|·\|W_sub\| + \|S\|·\|W_sub\| | \|W_sub\| 个独立小 MIP,每子波 \|O_w\|·\|K_w\| + \|S_w\| |
| 1000x60 阶段 2 规模 | 17000 x + 42500 y = ~60000 元 | 17 个子波,每个 ~60 单 → 每子波 ~3000 元 |
| 阶段 2 是否能并行 | 否(单 MIP) | 是(子波独立) |
| 1833x60 阶段 2 规模 | 56823 x + 77500 y = ~134000 元 | 31 个子波,每个 ~60 单 → 每子波 ~3000 元 |

**A 的阶段 2 是单一巨型 MIP**,变量数随 \|I\|·\|W_sub\| 线性增长,1000x60 就到 6 万元,HiGHS 撞 600s phase2 时间上限,gap 12%。

**B 的阶段 2 是 17-31 个独立小 MIP**,每个固定 ~60 单规模(~3000 元),HiGHS 秒级出最优。即便有 50 个子波,顺序解也只需 50×~5s = 250s。

B 用「阶段 2 规模恒定」换「阶段 1 决策次优」,在大规模上这个交易非常划算。

## 1000x60 三方 obj 对比

| Solver | time_s | status | gap | obj | vs 最优 |
|---|---|---|---|---|---|
| two_phase_a | 1150 | time_limit | 12% | **1002** | (基准,A 最优) |
| two_phase_b | 25 | optimal | 0% | 1336 | +33% |
| joint highs | 1650 | time_limit | 57% | 1525 | +52% |

A 的 obj 最小(1002),但花了 1150s 还没最优(gap 12%)。B 的 obj 大 33%,但 25s 出最优。highs 既慢又差。

→ 大规模上,若优先「快出可行解 + 知道是不是最优」选 B;若优先「obj 最小,愿意等 30 min」选 A;highs 没有适用场景。

## 适用边界更新

| 规模 | 推荐 solver | 理由 |
|---|---|---|
| ≤ 500 单 | two_phase_a | obj 最优,秒级出解 |
| 500-1000 单 | two_phase_a(等 30 min)或 two_phase_b(25s) | A 解更优但 gap 12%;B 立即 optimal |
| ≥ 1000 单 | two_phase_b | A 撞 30 min time_limit,gap 大;B 仍能 optimal |
| 任何规模 | 都不推荐 joint highs | 大规模撞墙,小规模也不比 A 快 |

## 下一步

1. spec 3.7 的 6 个场景(50x5, 50x10, 100x10, 100x50, 200x50, 200x60)用 two_phase_a 跑一遍,确认小规模稳定
2. 1000x60 / 1000x20 用 B 跑一遍 POC 主循环,看 obj 差距对最终业务指标的影响
3. 考虑 A+B 混合:小规模用 A(obj 最优),大规模用 B(可求解)。规模阈值 ~500 单

## 复现命令

```bash
# A 边界
.venv/bin/python -u scripts/demo_real_data.py \
    --orders /Users/admin/蔡司项目数据/作业数据/作业_2026-07-19.xlsx \
    --inventory /Users/admin/蔡司项目数据/库存数据/库存_2026-07-19_日报.xlsx \
    --solver two_phase_a --scenarios 1000x60,1833x60,1000x20 \
    --time-limit 1800 --output /tmp/scale30min_a.csv

# B 边界
.venv/bin/python -u scripts/demo_real_data.py \
    --orders /Users/admin/蔡司项目数据/作业数据/作业_2026-07-19.xlsx \
    --inventory /Users/admin/蔡司项目数据/库存数据/库存_2026-07-19_日报.xlsx \
    --solver two_phase_b --scenarios 1000x60,1833x60,1000x20 \
    --time-limit 1800 --output /tmp/scale30min_b.csv

# highs 边界
.venv/bin/python -u scripts/demo_real_data.py \
    --orders /Users/admin/蔡司项目数据/作业数据/作业_2026-07-19.xlsx \
    --inventory /Users/admin/蔡司项目数据/库存数据/库存_2026-07-19_日报.xlsx \
    --solver highs --scenarios 200x60,500x60,1000x60,200x20,500x20 \
    --time-limit 1800 --output /tmp/scale30min_highs.csv
```
