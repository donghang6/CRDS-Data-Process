# O2-O2 CIA 温度依赖系数处理流程与不确定度分析

本文档整理本项目中 O2-O2 碰撞诱导吸收二元系数 `B` 的温度依赖系数 `dB/dT` 计算过程，以及对应的不确定度传播方法。当前实验数据的主处理脚本为：

```bash
scripts/analyze_temperature_dependence_from_summary.py
```

输入表为：

```text
/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/summary.txt
```

该表已经包含每个波数点处不同温度、压力条件下的二元碰撞吸收系数 `B` 及其标准不确定度 `u_B`。因此温度依赖系数分析从 `B(nu)` 和 `u_B(nu)` 开始，不再重新计算吸收系数、数密度或积分强度。

## 1. 输入数据

输入表中每一行对应一个波数 `nu`，主要列如下：

| 列名 | 含义 | 单位 |
|---|---|---|
| `B 273` | 273 K, 500 Torr 的 O2-O2 二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `U B 273` | `B 273` 的标准不确定度 | `cm^-1 amagat^-2` |
| `B 303 500` | 303 K, 500 Torr 的二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `U B 303 500` | 303 K, 500 Torr 的标准不确定度 | `cm^-1 amagat^-2` |
| `B 303 600` | 303 K, 600 Torr 的二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `U B 303 600` | 303 K, 600 Torr 的标准不确定度 | `cm^-1 amagat^-2` |
| `B 303 700` | 303 K, 700 Torr 的二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `U B 303 700` | 303 K, 700 Torr 的标准不确定度 | `cm^-1 amagat^-2` |
| `B 333 500` | 333 K, 500 Torr 的二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `U B 333 500` | 333 K, 500 Torr 的标准不确定度 | `cm^-1 amagat^-2` |

其中 `B` 的不确定度来自前序误差传播：

```text
B(nu) = alpha(nu) / rho^2
```

若吸收系数 `alpha` 与数密度 `rho` 的不确定度相互独立，则：

```text
u_B = |B| sqrt[ (u_alpha / alpha)^2 + (2 u_rho / rho)^2 ]
```

这里 `alpha` 的不确定度来自 O2 和 Ar 拟合衰荡时间的不确定度传播；`rho` 的不确定度来自压力和温度的不确定度传播。

## 2. 303 K 三个压力点的合成

303 K 有 500, 600, 700 Torr 三组数据。为了让温度依赖拟合中的温度点保持为 273, 303, 333 K 三个独立温度，当前推荐做法是先将 303 K 的三组压力数据合成为一个 303 K 等效数据点。

对某一波数 `nu`，设 303 K 三组测量为：

```text
B_i(nu), u_i(nu),   i = 1, 2, 3
```

权重取反方差权重：

```text
w_i = 1 / u_i^2
```

303 K 合成值为：

```text
B_303(nu) = sum_i w_i B_i(nu) / sum_i w_i
```

合成值的内部标准不确定度为：

```text
u_303,internal(nu) = sqrt[ 1 / sum_i w_i ]
```

为了检查同温度不同压力组之间的离散程度，计算 303 K 压力组的卡方：

```text
chi2_303(nu) = sum_i [B_i(nu) - B_303(nu)]^2 / u_i(nu)^2
```

自由度为：

```text
dof_303 = N_303 - 1
```

约化卡方为：

```text
chi2_red,303 = chi2_303 / dof_303
```

Birge ratio 定义为：

```text
R_303 = max(1, sqrt(chi2_red,303))
```

最终用于温度拟合的 303 K 合成不确定度为：

```text
u_303(nu) = R_303 u_303,internal(nu)
```

如果三组压力之间的一致性好，`chi2_red,303 <= 1`，则 `R_303 = 1`，不放大不确定度；如果三组压力之间的离散程度超过各自 `u_B` 所能解释的范围，则 `R_303 > 1`，合成后的 303 K 不确定度会被放大。

## 3. 温度依赖系数的逐波数拟合

对每一个波数 `nu`，使用三个温度点：

```text
T = 273 K, 303 K, 333 K
```

对应的 `B` 值为：

```text
B_273(nu), B_303(nu), B_333(nu)
```

对应的不确定度为：

```text
u_273(nu), u_303(nu), u_333(nu)
```

假设在当前温度范围内 `B` 随温度可用一阶线性关系表示：

```text
B(nu, T) = a(nu) + b(nu) T
```

其中：

```text
b(nu) = dB(nu)/dT
```

即温度依赖系数。

## 4. 加权线性最小二乘

对固定波数 `nu`，构建设计矩阵：

```text
X = [ [1, T_1],
      [1, T_2],
      [1, T_3] ]
```

观测向量为：

```text
y = [B_273, B_303, B_333]^T
```

权重矩阵为：

```text
W = diag(1/u_273^2, 1/u_303^2, 1/u_333^2)
```

加权线性最小二乘解为：

```text
beta = [a, b]^T = (X^T W X)^(-1) X^T W y
```

因此每个波数点的温度依赖系数为：

```text
dB/dT = b
```

单位为：

```text
cm^-1 amagat^-2 K^-1
```

绘图时通常将其除以 `1e-9`，用：

```text
10^-9 cm^-1 amagat^-2 K^-1
```

作为纵轴单位。

## 5. 温度依赖系数的不确定度

加权最小二乘的内部协方差矩阵为：

```text
Cov(beta) = (X^T W X)^(-1)
```

截距和斜率的内部标准不确定度分别为：

```text
u_a,internal = sqrt[Cov(beta)_{00}]
u_b,internal = sqrt[Cov(beta)_{11}]
```

其中：

```text
u_b,internal = u(dB/dT)_internal
```

为了避免温度点之间的离散程度超过输入不确定度时低估拟合不确定度，进一步计算温度拟合残差：

```text
r_j = B_j - [a + b T_j]
```

拟合卡方为：

```text
chi2_fit = sum_j (r_j / u_j)^2
```

温度拟合中有 3 个温度点、2 个拟合参数，因此自由度为：

```text
dof_fit = 3 - 2 = 1
```

约化卡方为：

```text
chi2_red,fit = chi2_fit / dof_fit
```

温度拟合的 Birge ratio 为：

```text
R_fit = max(1, sqrt(chi2_red,fit))
```

最终温度依赖系数的标准不确定度为：

```text
u(dB/dT) = R_fit u_b,internal
```

同理，截距的不确定度为：

```text
u(a) = R_fit u_a,internal
```

如果实验点与线性模型在输入不确定度范围内一致，则 `R_fit = 1`；若实验点相对于线性模型的偏离明显大于输入不确定度，则 `R_fit > 1`，最终 `u(dB/dT)` 会被放大。

## 6. 输入不确定度放大两倍的保守分析

为检查输入 `u_B` 可能偏小对最终 `dB/dT` 不确定度的影响，脚本支持在计算前统一放大所有输入不确定度：

```text
u_B' = k_u u_B
```

当前保守分析使用：

```text
k_u = 2
```

对应命令参数为：

```text
--input-uncertainty-scale 2
```

需要注意的是，统一放大输入不确定度后，内部拟合协方差会随之增大；但 Birge ratio 可能减小。因此最终 `u(dB/dT)` 不一定在所有波数点处严格等于原来的两倍，而是由内部协方差和 Birge ratio 共同决定。

## 7. 推荐运行命令

### 标准不确定度版本

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/analyze_temperature_dependence_from_summary.py \
  --input '/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/summary.txt' \
  --output-dir output/results/analysis/B_temperature_dependence_from_summary_303_combined \
  --fit-303-mode combined \
  --combined-303-uncertainty scaled
```

### 输入 `u_B` 放大两倍的保守版本

```bash
conda run -n CRDS-Data-Process env PYTHONPATH=src python scripts/analyze_temperature_dependence_from_summary.py \
  --input '/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/summary.txt' \
  --output-dir output/results/analysis/B_temperature_dependence_from_summary_303_combined_uB_x2 \
  --fit-303-mode combined \
  --combined-303-uncertainty scaled \
  --input-uncertainty-scale 2
```

### Elsevier 双栏图

当前用于绘图的脚本读取 `B_temperature_dependence_from_summary_303_combined_uB_x2` 目录：

```bash
conda run -n CRDS-Data-Process env MPLCONFIGDIR=/private/tmp/mplconfig python scripts/plot_temperature_dependence_elsevier_double_column.py
```

## 8. 主要输出文件

标准版本输出目录：

```text
output/results/analysis/B_temperature_dependence_from_summary_303_combined
```

保守版本输出目录：

```text
output/results/analysis/B_temperature_dependence_from_summary_303_combined_uB_x2
```

主要文件：

| 文件 | 说明 |
|---|---|
| `temperature_dependence_weighted_fit.csv` | 全波数点的温度依赖拟合结果，包括 `dB/dT`、`u(dB/dT)`、截距、Birge ratio、残差等 |
| `temperature_dependence_weighted_selected_wavenumbers.csv` | 代表性波数点的数据，用于左图 `B-T` 拟合展示 |
| `temperature_dependence_weighted_summary.csv` | 整个波段的统计摘要 |
| `temperature_dependence_weighted_column_notes_zh.csv` | 输出列的中文注释 |
| `temperature_dependence_origin_left_panel_points.csv` | 适合 Origin 绘制左图散点和误差棒的数据 |
| `temperature_dependence_origin_left_panel_fit_lines.csv` | 适合 Origin 绘制左图拟合线的数据 |
| `temperature_dependence_elsevier_double_column.png/pdf` | 双栏论文图 |

## 9. 当前结果摘要

基于 303 K 三组压力合成、并使用 `u_B` 放大两倍的保守版本，结果摘要为：

```text
波数范围: 9120-9820 cm^-1
点数: 70001
dB/dT 最大值: 1.40742625149323e-9 cm^-1 amagat^-2 K^-1
最大值位置: 9322.87 cm^-1
dB/dT 最小值: 4.39005507168653e-11 cm^-1 amagat^-2 K^-1
最小值位置: 9798.94 cm^-1
u(dB/dT) 中位数: 3.66562283630293e-11 cm^-1 amagat^-2 K^-1
u(dB/dT) 平均值: 4.77263108031724e-11 cm^-1 amagat^-2 K^-1
相对标准不确定度中位数: 7.343897136224 %
温度拟合 Birge ratio 中位数: 1
303 K 加权 B 峰值: 1.01482209936224e-6 cm^-1 amagat^-2
303 K 加权 B 峰值位置: 9386.08 cm^-1
```

代表性波数点包括：

```text
9200, 9323, 9420, 9520, 9800 cm^-1
```

其中 `9323 cm^-1` 选在 `dB/dT` 最大值附近，用于展示温度依赖最强的位置。

## 10. 图中误差表示

左图为代表性波数点处 `B` 随温度变化的结果：

```text
横坐标: T / K
纵坐标: B / (10^-6 cm^-1 amagat^-2)
误差棒: u_B
拟合线: B(T) = a + (dB/dT) T
```

右图为全波段温度依赖系数：

```text
横坐标: nu / cm^-1
纵坐标: dB/dT / (10^-9 cm^-1 amagat^-2 K^-1)
阴影: ±u(dB/dT)
黑点和虚线: 左图选取的代表性波数位置
```

如果论文中需要给出约 95% 覆盖概率的不确定度，可使用扩展不确定度：

```text
U(dB/dT) = 2 u(dB/dT)
```

但图中当前阴影默认表示标准不确定度 `±1 sigma`。如需展示 `±2 sigma`，应在绘图时将 `u(dB/dT)` 乘以 2。

