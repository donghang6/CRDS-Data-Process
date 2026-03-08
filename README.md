# CRDS-Data-Process

CRDS（腔衰荡光谱）数据处理工具包 —— 从原始衰荡时间到光谱参数提取的完整五步流水线。

## 处理流程

```
原始衰荡数据 → 离群值过滤 → 标准具去除 → 单光谱拟合 → 多光谱联合拟合 → N₂展宽线性回归
     Step 1         Step 2        Step 3          Step 4              Step 5
```

| 步骤 | 说明 | 输出目录 |
|------|------|----------|
| **Step 1** 衰荡时间处理 | sigma-clip 离群值过滤，计算平均 τ 和吸收系数 α | `output/results/ringdown/` |
| **Step 2** 标准具去除 | HITRAN 辅助定位吸收区，FFT 滤波去除标准具效应 | `output/results/etalon/` |
| **Step 3** 单光谱拟合 | MATS 单压力独立拟合，提取线强、展宽、位移等参数 | `output/results/mats/` |
| **Step 4** 多光谱联合拟合 | 纯 O₂ 多压力联合拟合，MAD 筛选离群线强；误差过大参数自动固定后重拟合 | `output/results/mats_multi/` |
| **Step 5** N₂ 展宽提取 | 利用 O₂+N₂ 单光谱结果 + 纯 O₂ 联合拟合，加权线性回归提取 N₂ 展宽/位移 | `output/results/final/O2_N2/` |

最终所有拟合参数汇总至 `output/results/final/spectral_parameters.csv`（每次运行自动备份旧表）。

## 项目结构

```
CRDS-Data-Process/
├── pyproject.toml                          # 项目构建配置与依赖
├── main.py                                 # 流水线入口（支持命令行指定目标）
├── src/crds_process/                       # 核心代码包
│   ├── io/                                 #   数据 I/O
│   │   └── readers.py                      #     原始数据读取与文件名解析
│   ├── ringdown/                           #   衰荡时间处理
│   │   ├── filtering.py                    #     sigma-clip / IQR 离群值过滤
│   │   └── processing.py                   #     统计处理与 τ 计算
│   ├── baseline/                           #   基线处理
│   │   └── etalon.py                       #     标准具效应检测与 FFT 去除
│   ├── spectral/                           #   光谱拟合
│   │   ├── mats_wrapper.py                 #     MATS 单/多光谱拟合封装
│   │   └── linear_regression.py            #     N₂ 展宽加权线性回归
│   ├── preprocessing.py                    #   原始数据发现与预处理
│   ├── gas_config.py                       #   气体混合比/稀释气体配置
│   ├── log.py                              #   日志配置
│   └── pipeline.py                         #   五步端到端流水线
├── data/
│   ├── raw/                                # 原始数据 (gas_type/transition/pressure/)
│   │   ├── O2/                             #   纯 O₂ 数据
│   │   └── O2_N2/                          #   O₂+N₂ 混合气数据
│   ├── processed/                          # 预留的中间数据目录
│   └── hitran/                             # HITRAN 参考数据（运行下载脚本后生成）
├── scripts/
│   └── download_o2_lines.py                # HITRAN O₂ 谱线数据下载脚本（依赖 hitran-api/hapi）
└── output/
    ├── results/                            # 各步骤处理结果
    │   ├── ringdown/                       #   Step 1 输出
    │   ├── etalon/                         #   Step 2 输出
    │   ├── mats/                           #   Step 3 输出
    │   ├── mats_multi/                     #   Step 4 输出
    │   └── final/                          #   最终汇总
    │       ├── spectral_parameters.csv     #     参数主表（所有跃迁）
    │       ├── O2/{transition}/            #     纯 O₂ 联合拟合结果
    │       └── O2_N2/{transition}/         #     N₂ 线性回归结果
    └── logs/                               # 运行日志（带时间戳）
```

## 快速开始

```bash
# 安装（含开发工具 / notebook / MATS 适配）
pip install -e ".[dev,notebook,mats]"

# 若仅下载 HITRAN 参考线（可选）需额外安装 HAPI
pip install hitran-api

# 下载 HITRAN O₂ 参考数据（推荐）
python scripts/download_o2_lines.py

# 运行完整流水线（处理全部数据）
python main.py

# 仅处理指定目标
python main.py O2/9386.2076                  # O₂ 下 9386.2076 跃迁（所有压力）
python main.py O2/9386.2076/100Torr          # 精确到单个压力
python main.py O2/9386.2076 O2_N2/9386.2076  # 同时处理多个目标

# 仅拟合指定跃迁（吸收线）
python main.py O2/9403.163069 --fit-transitions 9403.163069
python main.py O2/9403.163069 --fit-lines 9403.163069,9401.731225

# 仅提取 N2 展宽，跳过纯 O2 联合拟合（依赖已有纯 O2 Step 4 结果）
python main.py --n2-only O2_N2/9403.163069
python main.py --from-ringdown --n2-only O2_N2/9403.163069
python main.py --from-etalon --n2-only O2_N2/9403.163069
python main.py --n2-only O2_N2/9403.163069 --optimize
python main.py --n2-only O2_N2/9403.163069 --optimize --min-pressures 4

# 生成建议重测点报告（只读取已有 final 结果）
python main.py --remeasure-report
python main.py --remeasure-report O2/9403.163069
python main.py --remeasure-report O2/9403.163069 O2_N2/9403.163069
python main.py --remeasure-report --remeasure-rel 0.05 --remeasure-sigma 3
python main.py --remeasure-report --remeasure-rel-o2 0.05 --remeasure-rel-o2n2 0.10

# 跳过 Step 1，复用已生成 ringdown 结果，从 Step 2 开始
python main.py --from-ringdown
python main.py --from-ringdown O2/9386.2076

# 跳过 Step 1/2，复用已生成 etalon 结果，从 Step 3 开始
python main.py --from-etalon
python main.py --from-etalon O2/9386.2076

# Step 4: 指定多光谱联合拟合压力（跳过自动筛选）
python main.py --pressures O2/9386.2076=100Torr,200Torr,300Torr

# Step 4: 自动搜索最优压力组合（按 QF 最大）
python main.py O2/9386.2076 --optimize
python main.py O2/9386.2076 --optimize --min-pressures 4
```

也可在代码中直接调用：

```python
from crds_process.pipeline import CRDSPipeline

# 全量运行
CRDSPipeline().run()

# 指定目标
CRDSPipeline(targets=["O2/9386.2076"]).run()
```

### Step 4 压力选择优先级

Step 4（纯 O₂ 联合拟合）的压力点选择按以下顺序执行：

1. 若传入 `--pressures`，使用你指定的压力列表。
2. 否则若启用 `--optimize`，枚举组合并选择 QF 最大的组合。
3. 否则使用默认 MAD 线强离群筛选。

补充：在 `--from-ringdown` / `--from-etalon` 续跑模式下，若同时提供
`--pressures`，Step 2/Step 3 也会仅处理该跃迁指定的压力。

### Step 5 压力选择优先级

Step 5（N₂ 线性回归）的压力点选择按以下顺序执行：

1. 若对 `O2_N2/{transition}` 传入 `--pressures`，仅使用这些压力做回归。
2. 否则若启用 `--optimize`，枚举所有压力组合（最少 3 个压力，或 `--min-pressures` 与 3 取较大值），按 `gamma0_N2` 的 `R²` 最大选择最终组合。
3. 否则使用全部可用压力直接回归。

启用自动搜索时，会额外输出 `pressure_optimization_n2.csv`，记录各组合的 `R²`、`gamma0_N2` 和最终选中的压力组合。

补充：Step 3 现在会对明显掉入坏局部极小值的单谱结果执行一次约束重拟合；Step 5 会自动跳过
`fit_valid=False` 或缺失关键误差（如 `sw_err`、`gamma0_air_err`）的压力点。

### 指定拟合跃迁

若一个光谱窗口中包含多条吸收线，而你只想拟合其中一条/几条，可使用：

- `--fit-transitions`
- `--fit-lines`
- `--fit-nu`（别名）

参数值为跃迁波数（cm⁻¹），支持逗号分隔多个值。启用后 Step 3/4 的 HITRAN 线表会仅保留这些跃迁。

### 仅提取 N2 展宽

若纯 O2 的 Step 4 联合拟合结果已经存在，而你只想更新 O2+N2 的单光谱拟合和
Step 5 线性回归，可使用 `--n2-only`。

该模式只处理 `O2_N2` 数据，跳过纯 O2 联合拟合，但 Step 5 仍会读取
`output/results/final/O2/{transition}/multi_fit_result.csv` 作为固定 O2 参考。

### 建议重测点报告

若你只想检查哪些压力点质量较差、建议重新测量，可使用 `--remeasure-report`。

该命令不会重新执行 Step 1~5，只会读取已有的 `output/results/final/` 结果，
并输出七张表以及一组按压力拆分的 `PDF` 清单：

- `output/results/final/remeasure_candidates.csv`：压力点明细
- `output/results/final/remeasure_pressures.csv`：按压力汇总后的建议重测列表
- `output/results/final/remeasure_pressures_O2.csv`：仅纯 O2 的按压力重测清单
- `output/results/final/remeasure_pressures_O2_N2.csv`：仅 O2_N2 的按压力重测清单
- `output/results/final/remeasure_transitions.csv`：按跃迁波数汇总后的建议重测列表
- `output/results/final/remeasure_transitions_O2.csv`：仅纯 O2 的建议重测跃迁
- `output/results/final/remeasure_transitions_O2_N2.csv`：仅 O2_N2 的建议重测跃迁
- `output/results/final/remeasure_pressure_lists/{gas_type}/*.pdf`：一个压力一个 PDF 清单；若相邻待重测跃迁间隔小于 `1 cm^-1`，则合并为一次测量，并在整组左右各外扩 `2 cm^-1`

检查规则包括：

1. 纯 O2：仅检查单谱 `sw`，并与同一跃迁下其他压力点的 `sw` 自比较，找出偏差过大的压力点
   仅在 `data/reference/o2_remeasure_pressure_plan.csv` 定义的压力集合内判断；
   额外加测但不在表内的纯 O2 压力，不列入重测建议
2. O2_N2：单谱 `sw` 相对对应纯 O2 联合拟合 `sw` 的偏差
3. 若单谱结果 `fit_valid=False` 或缺失关键误差，也会直接列入建议重测点
4. `spectral_parameters.csv` 中若核心参数为空，也会在跃迁级报告里单独标记为漏测参数
   （不添加压力），当前检查 `sw`、`gamma0_O2`、`gamma0_N2`

可通过以下参数调节阈值：

- `--remeasure-rel`：统一覆盖 O2 和 O2_N2 的相对偏差阈值
- `--remeasure-rel-o2`：纯 O2 的相对偏差阈值，默认 `0.05`（即 5%）
- `--remeasure-rel-o2n2`：O2_N2 的相对偏差阈值，默认 `0.10`（即 10%）
- `--remeasure-sigma`：偏差超过联合不确定度的阈值，默认 `3`

## 输出说明

### 参数主表 (`spectral_parameters.csv`)

以跃迁波数为第一列，每行一个跃迁，包含：

| 列组 | 参数 | 说明 |
|------|------|------|
| O₂ 参数 | `sw`, `gamma0_O2`, `n_gamma0_O2`, `SD_gamma_O2`, `delta0_O2`, `SD_delta_O2` | 纯 O₂ 联合拟合结果；每项带对应 `_err` 列 |
| N₂ 参数 | `gamma0_N2` | 仅保留 N₂ 展宽线性回归结果；附带 `_err`、`_R2`、`_npts` 列。若当前未生成 N₂ 线性回归结果，则为 `NaN` |
| HITRAN 参考 | `sw_HITRAN`, `gamma_self_HITRAN`, `gamma_air_HITRAN`, `gamma0_N2_HITRAN`, `n_air_HITRAN`, `delta_air_HITRAN`, `elower_HITRAN` | 其中 `gamma0_N2_HITRAN` 由干空气近似 `gamma_air = 0.21*gamma_self + 0.79*gamma_N2` 反推得到 |
| 辅助 | `n_spectra_O2`, `QF_O2`, `residual_std_O2` | 拟合质量指标 |

每次运行若表格已存在，自动备份为 `spectral_parameters_YYYYMMDD_HHMMSS.csv`。

### 拟合图表

每个拟合结果同时生成 PNG (150 dpi) 和 PDF (矢量) 格式，包含四个面板：
1. **原始光谱 + 拟合曲线** — α (ppm/cm) vs. 波数
2. **基线扣除光谱** — 去除基线后的纯吸收信号
3. **残差** — 数据与模型的偏差 (含 σ 标注)
4. **衰荡时间** — τ (μs) vs. 波数

## 数据目录约定

原始数据须按以下结构组织：

```
data/raw/
├── O2/                          # 气体类型
│   └── 9386.2076/               # 跃迁波数 (cm⁻¹)
│       ├── 70Torr/              # 压力
│       ├── 100Torr/
│       └── ...
└── O2_N2/                       # 混合气
    └── 9386.2076/
        ├── O2 300Torr N2 50Torr/
        └── ...
```

若需要主表中的 HITRAN 参考列，请准备：

```
data/hitran/O2_9000_10000_sw_ge_1e-29.csv
```

### 命名规范检测

流水线在 Step 1 之前会自动检测 `data/raw/` 下的命名是否符合规范：

| 层级 | 规范 | 示例 |
|------|------|------|
| 气体类型 | `O2` 或 `O2_N2` | `data/raw/O2/` |
| 跃迁波数 | 纯数字或浮点数 | `9386.2076` |
| 压力 (O₂) | `{数字}Torr` | `100Torr` |
| 压力 (混合) | `O2 {数字}Torr N2 {数字}Torr` | `O2 300Torr N2 50Torr` |
| 数据文件 | `{序号} {波数} {YYYYMMDDHHmmss}.txt` | `  1 9386.08204 20260121100904.txt` |

不符合规范的目录/文件会输出 ⚠ 警告，但不会阻止流水线继续运行。

## 依赖

- Python ≥ 3.10
- NumPy, SciPy, Pandas, Matplotlib, lmfit
- [MATS](https://github.com/usnistgov/MATS)（多光谱联合拟合引擎）
- `hitran-api`（仅 `scripts/download_o2_lines.py` 需要）
