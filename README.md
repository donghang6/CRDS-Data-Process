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
│   └── hitran/                             # HITRAN 光谱数据库文件
├── scripts/
│   └── download_o2_lines.py                # HITRAN O₂ 谱线数据下载脚本
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
# 安装（开发模式）
pip install -e ".[dev,notebook]"

# 运行完整流水线（处理全部数据）
python main.py

# 仅处理指定目标
python main.py O2/9386.2076                  # O₂ 下 9386.2076 跃迁（所有压力）
python main.py O2/9386.2076/100Torr          # 精确到单个压力
python main.py O2/9386.2076 O2_N2/9386.2076  # 同时处理多个目标
```

也可在代码中直接调用：

```python
from crds_process.pipeline import CRDSPipeline

# 全量运行
CRDSPipeline().run()

# 指定目标
CRDSPipeline(targets=["O2/9386.2076"]).run()
```

## 输出说明

### 参数主表 (`spectral_parameters.csv`)

以跃迁波数为第一列，每行一个跃迁，包含：

| 列组 | 参数 | 说明 |
|------|------|------|
| O₂ 参数 | `sw`, `gamma0_O2`, `SD_gamma_O2`, `delta0_O2`, `SD_delta_O2` | 线强、展宽、位移及其误差 |
| N₂ 参数 | `gamma0_N2`, `SD_gamma_N2`, `delta0_N2`, `SD_delta_N2` | 线性回归提取的 N₂ 展宽/位移及 R² |
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
