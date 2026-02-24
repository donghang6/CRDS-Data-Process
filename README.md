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
| 3. 吸收系数 | `crds_process.absorption` | α = (1/c)(1/τ − 1/τ₀) 转换及误差传播 |
| 4. 基线处理 | `crds_process.baseline` | 多项式/样条基线拟合与扣除 |
| 5. 光谱拟合 | `crds_process.spectral` | MATS 封装，提取线强、展宽系数等参数 |
| 6. 可视化 | `crds_process.visualization` | 衰荡分布、吸收光谱、拟合结果图 |

## 项目结构

```
CRDS-Data-Process/
├── pyproject.toml                          # 项目构建配置与依赖
├── config/
│   └── default.yaml                        # 默认实验参数配置
├── src/crds_process/                       # 核心代码包
│   ├── io/                                 #   数据 I/O
│   │   ├── readers.py                      #     原始数据读取
│   │   └── exporters.py                    #     结果导出
│   ├── ringdown/                           #   衰荡时间处理
│   │   ├── filtering.py                    #     离群值过滤
│   │   └── processing.py                   #     统计处理
│   ├── absorption/                         #   吸收系数
│   │   └── coefficients.py                 #     τ→α 转换
│   ├── baseline/                           #   基线处理
│   │   └── fitting.py                      #     基线拟合与扣除
│   ├── spectral/                           #   光谱拟合
│   │   └── mats_wrapper.py                 #     MATS 封装
│   ├── visualization/                      #   可视化
│   │   └── plots.py                        #     绘图函数
│   ├── config.py                           #   配置管理
│   ├── pipeline.py                         #   端到端流水线
│   └── cli.py                              #   命令行入口
├── tests/                                  # 单元测试
├── notebooks/                              # Jupyter 探索笔记本
│   ├── 01_ringdown_exploration.ipynb
│   ├── 02_absorption_spectrum.ipynb
│   └── 03_spectral_fitting.ipynb
├── data/
│   ├── raw/                                # 原始数据（不纳入版本控制）
│   └── processed/                          # 中间结果
└── output/
    ├── figures/                            # 图表输出
    └── results/                            # 拟合结果
```

## 快速开始

```bash
# 安装（开发模式）
pip install -e ".[dev,notebook]"

# 运行完整流水线
crds-process run --data-dir data/raw --config config/default.yaml

# 跳过 MATS 拟合（仅处理到基线扣除）
crds-process run --data-dir data/raw --skip-mats

# 运行测试
pytest
```

## 配置

编辑 `config/default.yaml` 设置实验参数：
- **cavity**: 腔长、镜面反射率
- **ringdown**: 离群值��滤方法与阈值
- **absorption**: 空腔衰荡时间 τ₀
- **baseline**: 基线拟合方法与基线区域
- **gas**: 气体种类、温度、压力
- **mats**: MATS 拟合参数（数据库、拟合参数列表等）

## 依赖

- Python ≥ 3.10
- NumPy, SciPy, Pandas, Matplotlib
- PyYAML, Pydantic, lmfit
- [MATS](https://github.com/usnistgov/MATS)（可选，用于光谱拟合）
