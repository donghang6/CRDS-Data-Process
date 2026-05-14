# CIA 不确定度分析方法

本文档总结 O2-O2 碰撞诱导吸收（collision-induced absorption, CIA）数据处理中不确定度的计算流程。目标是从原始衰荡时间的不确定度出发，依次传播到吸收系数、O2 分子数密度、二元碰撞吸收系数以及积分强度，为论文中的不确定度分析部分提供可直接摘取的计算步骤和公式。

本文中所有不确定度若无特别说明，均为标准不确定度。最终表述实验结果时，可进一步给出扩展不确定度：

```text
U = k u
```

其中本文采用覆盖因子 `k = 2`，近似对应 95% 覆盖概率。

## 1. 主要物理量和单位

本工作最终关注的谱量为 O2-O2 二元碰撞吸收系数：

```text
B(nu) = alpha(nu) / rho^2
```

其中：

| 符号 | 含义 | 单位 |
|---|---|---|
| `nu` | 波数 | `cm^-1` |
| `tau` | 衰荡时间 | `us` |
| `alpha` | 吸收系数 | `cm^-1` |
| `rho` | O2 分子数密度，相对于 Loschmidt 数密度的 amagat 数 | `amagat` |
| `B` | O2-O2 二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `S` | 积分强度，`S = integral B(nu)dnu` | `cm^-2 amagat^-2` |

这里的 `rho` 是 O2 的分子数密度，而不是 Ar 的分子数密度。Ar 数据只用于确定腔损耗背景或参考衰荡行为；最终的 O2-O2 CIA 系数使用 O2 样品的数密度。

## 2. 总体计算链条

不确定度传播按以下顺序进行：

1. 从 raw/CIA 原始数据中计算每个波数点平均衰荡时间的 A 类不确定度。
2. 通过 Monte Carlo 方法把原始衰荡时间不确定度传播到拟合后的衰荡时间 `tau_fit`。
3. 由 Ar 与 O2 的拟合衰荡时间计算吸收系数 `alpha`，并传播得到 `u_alpha`。
4. 由压力和温度计算 O2 数密度 `rho`，并传播得到 `u_rho`。
5. 由 `B = alpha/rho^2` 计算 `B` 的不确定度 `u_B`。
6. 对 `B(nu)` 积分得到积分强度 `S`，并通过上下边界积分法得到 `u_S`。
7. 如需论文表格中的最终形式，使用扩展不确定度 `U_S = 2u_S`，并按括号不确定度形式书写。

## 3. 衰荡时间平均值的不确定度

每个波数点通常包含多次衰荡事件。对同一波数点的原始衰荡时间集合：

```text
tau_1, tau_2, ..., tau_N
```

先进行异常值剔除，再计算平均衰荡时间：

```text
tau_mean = (1/N) sum_i tau_i
```

样本标准偏差为：

```text
s_tau = sqrt( sum_i (tau_i - tau_mean)^2 / (N - 1) )
```

平均衰荡时间的 A 类标准不确定度取平均值标准误：

```text
u_tau_mean = s_tau / sqrt(N)
```

在程序中，对衰荡时间、温度和压力均先进行 robust 异常值剔除。默认方法优先使用 MAD robust z-score；当 MAD 不可用时使用 IQR 方法。这样可以避免少数坏点影响平均衰荡时间及后续不确定度传播。

对应脚本：

```bash
python scripts/calculate_cia_tau_uncertainty.py
```

主要输出：

```text
output/results/uncertainty/CIA/cia_tau_uncertainty_all.csv
output/results/uncertainty/CIA/cia_tau_uncertainty_summary.csv
```

关键列：

| 列名 | 含义 |
|---|---|
| `tau_mean_us` | 异常值剔除后的平均衰荡时间 |
| `tau_std_us` | 衰荡时间样本标准偏差 |
| `tau_sem_us` | 平均值标准误 |
| `tau_uncertainty_us` | 平均衰荡时间标准不确定度，本步骤中等于 `tau_sem_us` |
| `n_raw` | 原始事件数量 |
| `n_kept` | 剔除异常值后保留的事件数量 |
| `n_removed` | 剔除的事件数量 |

## 4. 拟合衰荡时间的不确定度

最终用于计算吸收系数的不是单个原始波数点的 `tau_mean`，而是经过 CIA Step 2 平滑/拟合后的衰荡时间曲线。因此需要把原始平均衰荡时间的不确定度传播到拟合曲线。

本工作采用 Monte Carlo 方法：

1. 对每个波数点的平均衰荡时间 `tau_i`，以其标准不确定度 `u_tau_i` 为标准差生成扰动：

   ```text
   tau_i* = tau_i + epsilon_i
   epsilon_i ~ N(0, u_tau_i^2)
   ```

2. 对扰动后的数据执行与正式数据完全相同的 Step 2 拟合流程。
3. 重复上述过程 `M` 次，得到每个波数点的拟合衰荡时间集合：

   ```text
   tau_fit,1(nu), tau_fit,2(nu), ..., tau_fit,M(nu)
   ```

4. 每个波数点的拟合衰荡时间标准不确定度取 Monte Carlo 样本标准偏差：

   ```text
   u_tau_fit(nu) = std[tau_fit,j(nu)]
   ```

对应脚本：

```bash
python scripts/monte_carlo_tau_fit_uncertainty.py
```

主要输出：

```text
output/results/continuum/CIA/{temperature}/{gas pressure}/continuum_step2_fit_mc_uncertainty.csv
```

关键列：

| 列名 | 含义 |
|---|---|
| `tau_fit_us_mc_mean_us` | Monte Carlo 后拟合衰荡时间均值 |
| `tau_fit_us_mc_uncertainty_us` | 拟合衰荡时间标准不确定度 |
| `tau_fit_us_mc_ci_low_us` | Monte Carlo 置信区间下界 |
| `tau_fit_us_mc_ci_high_us` | Monte Carlo 置信区间上界 |
| `tau_fit_us_mc_samples` | Monte Carlo 重复次数 |

## 5. 吸收系数的不确定度

CRDS 中腔损耗与衰荡时间满足：

```text
L = 1 / (c tau)
```

其中 `c = 2.99792458e10 cm/s`。若 `tau` 使用微秒，则：

```text
L = (1e6 / c) * (1 / tau_us)
```

O2 吸收系数由 O2 和 Ar 的衰荡时间差异得到：

```text
alpha = (1e6 / c) * (1/tau_O2 - 1/tau_Ar)
```

其中 `tau_O2` 和 `tau_Ar` 均以微秒为单位，所得 `alpha` 单位为 `cm^-1`。

假设 O2 和 Ar 的拟合衰荡时间不确定度相互独立，则误差传播为：

```text
u_alpha = (1e6 / c) * sqrt[
    (u_tau_O2 / tau_O2^2)^2
  + (u_tau_Ar / tau_Ar^2)^2
]
```

对应脚本：

```bash
python scripts/calculate_final_absorption_uncertainty.py \
  '/Users/donghang/科研/实验数据/氧气连续吸收温度/最终处理数据'
```

主要输出：

```text
output/results/uncertainty/CIA/final_absorption_uncertainty/
```

关键输出表：

```text
absorption_uncertainty_all.csv
absorption_uncertainty_wide_cm_inv.csv
summary.csv
```

关键列：

| 列名 | 含义 | 单位 |
|---|---|---|
| `tau_ar_us` | Ar 拟合衰荡时间 | `us` |
| `u_tau_ar_us` | Ar 拟合衰荡时间标准不确定度 | `us` |
| `tau_o2_us` | O2 拟合衰荡时间 | `us` |
| `u_tau_o2_us` | O2 拟合衰荡时间标准不确定度 | `us` |
| `alpha_cm_inv` | 吸收系数 | `cm^-1` |
| `u_alpha_cm_inv` | 吸收系数标准不确定度 | `cm^-1` |
| `u_alpha_rel_percent` | 吸收系数相对标准不确定度 | `%` |

## 6. O2 分子数密度的不确定度

O2 分子数密度用 amagat 表示：

```text
rho = (p / p0) * (T0 / T)
```

其中：

```text
p0 = 760 Torr
T0 = 273.15 K
```

即：

```text
rho = (p / 760) * (273.15 / T)
```

压力 `p` 使用 O2 样品的压力，温度 `T` 使用 O2 样品的温度。温度和压力均先从 raw/CIA 原始数据中统计，并进行异常值剔除。

压力和温度的标准不确定度采用 B 类不确定度：

```text
u_T = 0.001 K
```

CTR100 真空计在本实验压力范围内（500-700 Torr）采用 `0.20% of reading` 的精度。该值按矩形分布半宽处理，因此压力标准不确定度为：

```text
u_p = 0.20% * p / sqrt(3)
```

由误差传播：

```text
u_rho / rho = sqrt[ (u_p / p)^2 + (u_T / T)^2 ]
```

即：

```text
u_rho = rho * sqrt[ (u_p / p)^2 + (u_T / T)^2 ]
```

本工作在最终计算中，为保证与最终数据表中已经给出的 `B = alpha/rho^2` 完全一致，实际使用的 `rho` 优先由最终表反推：

```text
rho = sqrt(alpha / B)
```

同时使用 raw/CIA 剔除异常值后的 O2 压力和温度计算 `u_rho/rho`。这样可以确保不确定度传播与最终发表的 `B` 数值严格对应。

对应脚本：

```bash
python scripts/summarize_pressure_temperature.py \
  --root data/raw/CIA \
  --output output/results/uncertainty/CIA/final_rho_uncertainty/pressure_temperature_summary_used.csv \
  --write-outliers \
  --outliers-output output/results/uncertainty/CIA/final_rho_uncertainty/pressure_temperature_outliers_removed.csv

python scripts/calculate_final_rho_uncertainty.py \
  '/Users/donghang/科研/实验数据/氧气连续吸收温度/最终处理数据' \
  --pressure-temperature-summary output/results/uncertainty/CIA/final_rho_uncertainty/pressure_temperature_summary_used.csv \
  --pressure-relative-half-width-percent 0.2 \
  --temperature-uncertainty-k 0.001
```

主要输出：

```text
output/results/uncertainty/CIA/final_rho_uncertainty/
```

关键列：

| 列名 | 含义 | 单位 |
|---|---|---|
| `rho_amagat` | O2 数密度 | `amagat` |
| `u_rho_amagat` | O2 数密度标准不确定度 | `amagat` |
| `u_rho_rel_percent` | O2 数密度相对标准不确定度 | `%` |
| `pressure_torr_summary` | 剔除异常值后的 O2 平均压力 | `Torr` |
| `temperature_k_summary` | 剔除异常值后的 O2 平均温度 | `K` |
| `pressure_uncertainty_torr` | 压力标准不确定度 | `Torr` |
| `temperature_uncertainty_k` | 温度标准不确定度 | `K` |
| `n_pressure_temperature_used` | 温度压力统计中保留的点数 | - |
| `n_pressure_temperature_removed` | 温度压力统计中剔除的点数 | - |

## 7. 二元碰撞吸收系数 B 的不确定度

二元碰撞吸收系数为：

```text
B = alpha / rho^2
```

其中 `alpha` 和 `rho` 分别来自独立的测量和处理过程，因此按独立量进行误差传播。偏导数为：

```text
partial B / partial alpha = 1 / rho^2
partial B / partial rho = -2 alpha / rho^3 = -2B / rho
```

因此：

```text
u_B = sqrt[
    (u_alpha / rho^2)^2
  + (2B u_rho / rho)^2
]
```

等价的相对形式为：

```text
u_B / B = sqrt[
    (u_alpha / alpha)^2
  + (2u_rho / rho)^2
]
```

其中：

```text
u_B,alpha = u_alpha / rho^2
u_B,rho = 2B u_rho / rho
```

最终：

```text
u_B = sqrt(u_B,alpha^2 + u_B,rho^2)
```

对应脚本：

```bash
python scripts/calculate_final_binary_coefficient_uncertainty.py
```

主要输出：

```text
output/results/uncertainty/CIA/final_binary_coefficient_uncertainty/
```

关键输出表：

```text
binary_coefficient_uncertainty_all.csv
binary_coefficient_uncertainty_wide.csv
binary_coefficient_uncertainty_summary.csv
```

关键列：

| 列名 | 含义 | 单位 |
|---|---|---|
| `binary_coeff_recomputed_cm_inv_amagat_neg2` | 重新计算得到的二元碰撞吸收系数 | `cm^-1 amagat^-2` |
| `u_binary_coeff_from_alpha` | 由吸收系数不确定度贡献的 `B` 不确定度 | `cm^-1 amagat^-2` |
| `u_binary_coeff_from_rho` | 由数密度不确定度贡献的 `B` 不确定度 | `cm^-1 amagat^-2` |
| `u_binary_coeff` | 二元碰撞吸收系数标准不确定度 | `cm^-1 amagat^-2` |
| `u_binary_coeff_rel_percent` | 二元碰撞吸收系数相对标准不确定度 | `%` |

从当前结果看，`B` 的相对标准不确定度通常约为 `0.23%`，且主要由 `rho` 的不确定度贡献。这是因为 `B` 与 `rho` 的平方成反比，密度相对不确定度在传播到 `B` 时会乘以 2。

## 8. 积分强度及其不确定度

积分强度定义为：

```text
S = integral_{nu1}^{nu2} B(nu) dnu
```

当前积分范围为：

```text
nu1 = 9120 cm^-1
nu2 = 9820 cm^-1
```

采用梯形积分。对于等间隔波数网格，可以写成：

```text
S ~= sum_i w_i B_i
```

其中 `w_i` 为梯形积分权重。内部点的权重约为 `0.01 cm^-1`，两端点权重约为 `0.005 cm^-1`。

积分强度的不确定度采用上下边界积分法。先构造：

```text
B_upper(nu) = B(nu) + u_B(nu)
B_lower(nu) = B(nu) - u_B(nu)
```

分别积分得到：

```text
S_upper = integral B_upper(nu) dnu
S_lower = integral B_lower(nu) dnu
```

积分强度的标准不确定度取上下边界积分结果的半宽：

```text
u_S = (S_upper - S_lower) / 2
```

在离散形式下：

```text
S_upper ~= sum_i w_i [B_i + u_B_i]
S_lower ~= sum_i w_i [B_i - u_B_i]
u_S = (S_upper - S_lower) / 2
```

由于上下边界是对整条曲线进行积分，这种处理相当于将 `u_B(nu)` 作为谱曲线的不确定度包络，得到较直观、偏保守的积分强度不确定度。

对应脚本：

```bash
python scripts/calculate_final_integrated_strength_uncertainty.py
```

如果需要改变积分范围，例如只积分 `9200-9800 cm^-1`：

```bash
python scripts/calculate_final_integrated_strength_uncertainty.py \
  --start 9200 \
  --end 9800
```

主要输出：

```text
output/results/uncertainty/CIA/final_integrated_strength/integrated_strength_summary.csv
```

关键列：

| 列名 | 含义 | 单位 |
|---|---|---|
| `integrated_strength` | 积分强度 `S` | `cm^-2 amagat^-2` |
| `integrated_strength_upper` | 上边界积分 `S_upper` | `cm^-2 amagat^-2` |
| `integrated_strength_lower` | 下边界积分 `S_lower` | `cm^-2 amagat^-2` |
| `u_integrated_strength` | 积分强度标准不确定度 `u_S` | `cm^-2 amagat^-2` |
| `u_integrated_strength_rel_percent` | 积分强度相对标准不确定度 | `%` |
| `u_integrated_strength_from_alpha_boundary` | 仅由 `alpha` 项边界积分得到的诊断值 | `cm^-2 amagat^-2` |
| `u_integrated_strength_from_rho_common_scale_diagnostic` | 由 `rho` 公共比例项估算的诊断值 | `cm^-2 amagat^-2` |

## 9. 当前积分强度结果

按 `9120-9820 cm^-1` 全波段积分，当前结果如下：

| 条件 | `S` | `S_upper` | `S_lower` | `u_S` | 相对标准不确定度 |
|---|---:|---:|---:|---:|---:|
| 273K/500Torr | `2.44626148495602e-4` | `2.45192961839143e-4` | `2.44059335152061e-4` | `5.66813343540936e-7` | `0.231706%` |
| 303K/500Torr | `2.57445033295379e-4` | `2.58040665553796e-4` | `2.56849401036961e-4` | `5.95632258417188e-7` | `0.231363%` |
| 303K/600Torr | `2.57606571973108e-4` | `2.58203580907018e-4` | `2.57009563039197e-4` | `5.97008933910192e-7` | `0.231752%` |
| 303K/700Torr | `2.58100517493416e-4` | `2.58701144403235e-4` | `2.57499890583597e-4` | `6.00626909819021e-7` | `0.232710%` |
| 333K/500Torr | `2.71227849349803e-4` | `2.71894401989313e-4` | `2.70561296710293e-4` | `6.66552639510106e-7` | `0.245754%` |

单位均为：

```text
cm^-2 amagat^-2
```

## 10. 括号不确定度写法

论文表格中常将结果写为括号不确定度形式。规则为：

1. 先将不确定度保留两位有效数字。
2. 主值保留到与不确定度相同的小数位。
3. 括号中写不确定度最后对应位数的数字。

若使用标准不确定度 `u_S`，结果可写为：

| 条件 | 标准不确定度写法 |
|---|---|
| 273K/500Torr | `2.4463(57) x 10^-4` |
| 303K/500Torr | `2.5745(60) x 10^-4` |
| 303K/600Torr | `2.5761(60) x 10^-4` |
| 303K/700Torr | `2.5810(60) x 10^-4` |
| 333K/500Torr | `2.7123(67) x 10^-4` |

若使用扩展不确定度 `U_S = 2u_S`，结果可写为：

| 条件 | 扩展不确定度写法，`k = 2` |
|---|---|
| 273K/500Torr | `2.446(11) x 10^-4` |
| 303K/500Torr | `2.574(12) x 10^-4` |
| 303K/600Torr | `2.576(12) x 10^-4` |
| 303K/700Torr | `2.581(12) x 10^-4` |
| 333K/500Torr | `2.712(13) x 10^-4` |

建议论文表格中使用扩展不确定度，并在表注中说明：

```text
括号内为扩展不确定度，覆盖因子 k = 2。
```

英文可写为：

```text
The values in parentheses represent expanded uncertainties with a coverage factor of k = 2.
```

## 11. 可复现计算命令汇总

完整流程可按以下顺序运行：

```bash
# 1. 统计温度和压力，剔除异常值
python scripts/summarize_pressure_temperature.py \
  --root data/raw/CIA \
  --output output/results/uncertainty/CIA/final_rho_uncertainty/pressure_temperature_summary_used.csv \
  --write-outliers \
  --outliers-output output/results/uncertainty/CIA/final_rho_uncertainty/pressure_temperature_outliers_removed.csv

# 2. 计算吸收系数 alpha 的不确定度
python scripts/calculate_final_absorption_uncertainty.py \
  '/Users/donghang/科研/实验数据/氧气连续吸收温度/最终处理数据'

# 3. 计算 O2 数密度 rho 的不确定度
python scripts/calculate_final_rho_uncertainty.py \
  '/Users/donghang/科研/实验数据/氧气连续吸收温度/最终处理数据' \
  --pressure-temperature-summary output/results/uncertainty/CIA/final_rho_uncertainty/pressure_temperature_summary_used.csv \
  --pressure-relative-half-width-percent 0.2 \
  --temperature-uncertainty-k 0.001

# 4. 计算二元碰撞吸收系数 B 的不确定度
python scripts/calculate_final_binary_coefficient_uncertainty.py

# 5. 计算积分强度 S 及其不确定度
python scripts/calculate_final_integrated_strength_uncertainty.py
```

## 12. 论文方法段落示例

以下文字可作为论文中不确定度分析部分的基础表述：

本研究的不确定度分析采用逐步误差传播方法。首先，对每个波数点的多次衰荡事件进行异常值剔除，并由保留事件的样本标准偏差计算平均衰荡时间的 A 类标准不确定度。随后，通过 Monte Carlo 方法将平均衰荡时间的不确定度传播到 CIA 基线拟合后的衰荡时间曲线。每次 Monte Carlo 计算中，按照各波数点的衰荡时间标准不确定度对衰荡时间进行随机扰动，并执行与正式数据相同的基线拟合流程；最终以多次拟合结果的标准偏差作为拟合衰荡时间的不确定度。

吸收系数由 O2 和 Ar 条件下的衰荡时间差异计算得到，即 `alpha = (1e6/c)(1/tau_O2 - 1/tau_Ar)`，其中 `tau` 以微秒为单位。假设 O2 与 Ar 衰荡时间不确定度相互独立，利用一阶误差传播得到吸收系数标准不确定度。O2 数密度以 amagat 表示，按照理想气体关系 `rho = (p/760)(273.15/T)` 计算。压力和温度的不确定度按 B 类不确定度处理，其中温度标准不确定度为 `0.001 K`，CTR100 真空计压力精度按 `0.20% of reading` 的矩形分布半宽处理，并除以 `sqrt(3)` 转换为标准不确定度。

二元碰撞吸收系数由 `B = alpha/rho^2` 得到，其标准不确定度由 `alpha` 与 `rho` 的不确定度共同传播。积分强度定义为 `S = integral B(nu)dnu`，在 `9120-9820 cm^-1` 范围内采用梯形积分计算。积分强度的不确定度通过上下边界积分法确定，即分别对 `B(nu)+u_B(nu)` 和 `B(nu)-u_B(nu)` 积分，取二者差值的一半作为 `S` 的标准不确定度。最终表格中给出的括号不确定度为扩展不确定度，覆盖因子 `k = 2`。
