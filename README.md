# CRDS-Data-Process

CRDS（腔衰荡光谱）数据处理工具包 —— 从原始衰荡时间到 MATS 光谱参数拟合的完整流水线。

## 处理流程

```
原始衰荡数据 → 离群值过滤 → 吸收系数计算 → 基线扣除 → MATS光谱拟合 → 线强/展宽系数等参数
```

| 阶段 | 模块 | 说明 |
|------|------|------|
| 1. 数据读取 | `crds_process.io` | 解析文件名(序号/波数/时间戳)，读取衰荡事件 |
| 2. 衰荡处理 | `crds_process.ringdown` | sigma-clip / IQR 离群值过滤，统计平均τ |
| 3. 基线处理 | `crds_process.baseline` | 标准具效应检测与去除 |
| 4. 光谱拟合 | `crds_process.spectral` | MATS 封装，提取线强、展宽系数等参数 |

## 项目结构

```
CRDS-Data-Process/
├── pyproject.toml                          # 项目构建配置与依赖
├── main.py                                 # 流水线入口
├── src/crds_process/                       # 核心代码包
│   ├── io/                                 #   数据 I/O
│   │   └── readers.py                      #     原始数据读取
│   ├── ringdown/                           #   衰荡时间处理
│   │   ├── filtering.py                    #     离群值过滤
│   │   └── processing.py                   #     统计处理
│   ├── baseline/                           #   基线处理
│   │   └── etalon.py                       #     标准具效应去除
│   ├── spectral/                           #   光谱拟合
│   │   └── mats_wrapper.py                 #     MATS 封装
│   ├── preprocessing.py                    #   原始数据预处理
│   └── pipeline.py                         #   端到端流水线
├── data/
│   ├── raw/                                # 原始数据（不纳入版本控制）
│   └── hitran/                             # HITRAN 光谱数据库
├── scripts/
│   └── download_o2_lines.py                # HITRAN 数据下载脚本
└── output/
    └── results/                            # 拟合结果
```

## 快速开始

```bash
# 安装（开发模式）
pip install -e ".[dev,notebook]"

# 运行完整流水线
python main.py
```


## 依赖

- Python ≥ 3.10
- NumPy, SciPy, Pandas, Matplotlib, lmfit
- [MATS](https://github.com/usnistgov/MATS)（可选，用于光谱拟合）
