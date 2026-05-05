# O2 连续吸收与 Ar 背景数据处理方法说明

本文档总结本项目中用于 O2 连续吸收（collision-induced absorption, CIA）实验及 Ar 背景数据的处理流程。内容按论文“实验数据处理/数据分析方法”写法组织，可按需要摘取、删减或改写。本文档中的变量、列名和参数与项目代码保持一致。

## 1. 数据处理总体思路

本实验记录的是不同激光波数下的 CRDS 衰荡时间。对于每一个波数点，原始文件包含多次衰荡事件的拟合结果。数据处理的核心目标是：

1. 从原始衰荡事件中得到每个波数点具有代表性的平均衰荡时间；
2. 将衰荡时间转换为腔内总损耗谱；
3. 对无窄吸收结构的 Ar 数据和含有 O2 窄线吸收的 O2 数据分别建立连续变化的 CIA 基线；
4. 对必要的局部或全局区域进行平滑样条处理；
5. 将最终结果重采样到统一的波数网格，便于后续比较、作图和定量分析。

本流程只针对连续吸收数据。与线吸收处理不同，CIA 数据不做标准具去除，也不进行 MATS 谱线拟合。Ar 数据被视为无明显窄线吸收的背景损耗数据，直接在 loss 域拟合缓慢变化的基线。O2 数据包含窄线吸收，因此先利用相同温度和压力条件下的 HITRAN2024 O2 仿真谱识别并剔除吸收区，再只用非吸收区拟合缓慢变化的 CIA 基线。

## 2. 原始数据格式

每一个原始 txt 文件对应一个波数点。程序支持两种文件名格式：

```text
1 9630.00400 20260101183304.txt
9630.00400.txt
```

第一种格式中，中间字段为波数，单位为 cm-1；第二种格式中，文件名本身即为波数。每个文件内部为四列数据：

```text
tau_us    fit_residual    temperature_c    pressure_torr
```

其中 `tau_us` 为单次衰荡事件得到的衰荡时间，单位为 microsecond；`temperature_c` 为腔体温度，单位为摄氏度；`pressure_torr` 为腔体压力，单位为 Torr。

## 3. 原始数据质量控制

在正式处理前，对明显异常的原始测量区进行质量控制。质量控制只在原始衰荡时间层面进行，不在最终 loss 谱上人为调整窄线结构。

质量控制包括：

1. 对文件名进行统一，使文件名只保留波数，避免后续读取时混淆；
2. 对重复波数只保留一个文件；
3. 对明显测量不理想的局部波数段进行剔除；
4. 对补充测量数据替换原数据时，将原始被替换文件移动到归档目录，而不是直接删除；
5. 如果某个波段整体存在已知的衰荡时间偏移，可在指定波数范围内对该范围所有原始衰荡时间加上或减去相同的常数偏移，然后重新执行 Step 1。

完成这些修正后，重新从原始 txt 运行 Step 1，以保证后续所有结果均来自同一套整理后的原始衰荡时间。

## 4. Step 1：每个波数点衰荡时间的统计

对每一个波数点，设原始衰荡时间序列为：

```text
tau_1, tau_2, ..., tau_N
```

首先对该序列进行 sigma-clip 离群值剔除。默认使用 3 sigma，最多迭代 5 次。第 k 次迭代中，计算当前保留数据的均值和标准差：

```text
mean_k = mean(tau)
std_k  = std(tau)
```

只保留满足下式的事件：

```text
|tau_i - mean_k| < 3 std_k
```

若本轮没有新的点被剔除，或标准差为 0，则迭代停止。每个波数点至少需要保留一定数量的衰荡事件，默认最少 5 个事件，否则该波数点不进入后续分析。

过滤后的衰荡时间用于计算该波数点的平均衰荡时间和离散度：

```text
tau_mean = mean(tau_filtered)
tau_std  = std(tau_filtered)
tau_sem  = tau_std / sqrt(N_filtered)
```

实际输出的 Step 1 文件为：

```text
output/results/ringdown/CIA/{temperature}/{gas pressure}/ringdown_results.csv
```

主要列包括：

```text
wavenumber, tau_mean, tau_std, temperature, pressure
```

其中 `wavenumber` 为波数，`tau_mean` 为过滤后的平均衰荡时间，`tau_std` 为同一波数点内多次衰荡事件的标准差。

## 5. 波数间隔检查

读取原始文件后，程序会按波数排序，并检查相邻波数间隔。设相邻波数间隔为：

```text
delta_nu_i = nu_{i+1} - nu_i
```

程序计算所有间隔的中位数和标准差：

```text
median_delta = median(delta_nu)
std_delta    = std(delta_nu)
```

默认使用以下阈值识别间隔异常：

```text
upper = median_delta + 3 std_delta
lower = 0.1 median_delta
```

如果某个间隔大于 `upper`，说明可能存在局部缺失或跳点，该间隔两侧的点会被标记；如果某个间隔小于 `lower`，说明可能存在重复或异常密集点，后一个点会被标记。被标记的点不进入 Step 1 的最终衰荡时间表。

## 6. 衰荡时间到腔内损耗的转换

连续吸收处理从 Step 1 的 `ringdown_results.csv` 开始。默认使用 `tau_mean` 作为每个波数点的代表性衰荡时间。腔内总损耗定义为：

```text
loss_ppm_per_cm = (1e12 / c) / tau_us
```

其中：

```text
c = 2.99792458e10 cm/s
tau_us 的单位为 us
loss_ppm_per_cm 的单位为 ppm/cm
```

这个公式等价于：

```text
loss_ppm_per_cm = 1e12 / (c * tau_us)
```

代码中常数写作：

```text
TAU_US_TO_PPM_PER_CM = 1e12 / c
loss_ppm_per_cm = TAU_US_TO_PPM_PER_CM / tau_us
```

如果输入中存在衰荡时间统计误差 `tau_stats_us`，则 loss 的误差通过一阶误差传播得到：

```text
loss_stats_ppm_per_cm = (1e12 / c) * |tau_stats_us| / tau_us^2
```

如果提供参考腔损耗，例如参考衰荡时间谱或标量空腔衰荡时间 `tau0_us`，则可进一步计算吸收系数：

```text
alpha_ppm_per_cm = loss_sample_ppm_per_cm - loss_reference_ppm_per_cm
```

当前 CIA 处理通常不强制要求参考谱；若没有参考谱，程序仍输出 `loss_ppm_per_cm`，而 `alpha_ppm_per_cm` 置为 NaN。

## 7. CIA Step 2：Ar 数据的连续基线拟合

Ar 数据不包含 O2 窄线吸收，因此将其视为只有缓慢变化背景的连续损耗谱。Ar 的 Step 2 在 `loss_ppm_per_cm` 域进行滑动窗口拟合。

### 7.1 滑动窗口局部多项式拟合

设总损耗谱为：

```text
L(nu) = loss_ppm_per_cm
```

在波数范围内以固定步长放置一系列拟合中心：

```text
nu_c = nu_min, nu_min + step, nu_min + 2 step, ...
```

对每一个中心 `nu_c`，取窗口：

```text
|nu - nu_c| <= window / 2
```

在该窗口内对 `L(nu)` 做局部多项式拟合。默认多项式阶数为 2：

```text
L(nu) = a0 + a1 x + a2 x^2
```

其中 `x` 为中心化并缩放后的局部波数坐标。为了降低离群点对拟合的影响，每个窗口内使用 robust sigma clipping。拟合后只取该局部多项式在窗口中心 `nu_c` 处的值，作为一个基线锚点：

```text
L_anchor(nu_c)
```

如果某个窗口内有效点太少，则跳过该中心。

### 7.2 锚点连接

所有局部窗口得到的锚点按照波数排序，使用 PCHIP 插值连接成连续的基线：

```text
L_fit(nu)
```

PCHIP 的优点是比普通三次样条更不容易产生过冲，适合保持缓慢变化背景的形状。为了保证端点覆盖，如果第一个或最后一个锚点没有覆盖整个有效波数范围，程序会在端点补入最近的有效值。

### 7.3 可选平滑

如果指定 `--cia-fit-smooth`，程序会对 `L_fit(nu)` 再进行一次 Savitzky-Golay 平滑。平滑窗口宽度由 `--cia-fit-smooth` 指定，单位为 cm-1。程序根据实际波数间隔自动换算为奇数个数据点，并使用与拟合阶数相容的多项式阶数。

### 7.4 Ar 使用参数

当前 Ar 500 Torr 数据推荐使用：

```text
fit_window = 40 cm-1
fit_step   = 5 cm-1
fit_order  = 2
fit_sigma  = 4
fit_smooth = 20 cm-1
```

拟合得到的 loss 基线写入：

```text
loss_fit_ppm_per_cm
```

并换算回等效衰荡时间：

```text
tau_fit_us = (1e12 / c) / loss_fit_ppm_per_cm
```

同时输出残差：

```text
loss_residual_ppm_per_cm = loss_ppm_per_cm - loss_fit_ppm_per_cm
tau_residual_us          = tau_us - tau_fit_us
```

Ar Step 2 的输出文件为：

```text
output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv
```

## 8. CIA Step 2：O2 数据的 HITRAN 辅助吸收剔除

O2 数据中存在许多窄线吸收。窄线吸收会增加损耗、降低衰荡时间。如果直接像 Ar 一样对全部点进行平滑基线拟合，拟合结果会被 O2 窄吸收线向上拉高，无法代表缓慢变化的 CIA 基线。因此，O2 的 Step 2 与 Ar 分开处理。

O2 处理的核心思想是：先在相同温度和压力条件下生成 HITRAN2024 O2 仿真吸收谱，用仿真谱定位窄线吸收区；然后从基线拟合中剔除这些吸收区，只使用非吸收区数据拟合 CIA baseline。

### 8.1 HITRAN2024 O2 仿真谱

程序从 Step 1 结果中读取该波段的温度和压力。温度使用 `temperature_c` 的中位数；压力优先使用 `pressure_torr` 的正值中位数，如果文件中没有有效压力，则从压力标签如 `O2 500Torr` 中解析。

HITRAN 仿真条件为：

```text
T = temperature_c + 273.15 K
p = pressure_torr / 760 atm
```

程序调用 HAPI 的 Voigt 线型吸收系数计算，仿真波数范围为实测波数范围两端各扩展 0.5 cm-1。HITRAN 仿真的波数步长为：

```text
0.002 cm-1
```

HAPI 输出的吸收系数单位为 cm-1，程序将其转换为 ppm/cm：

```text
hitran_o2_loss_ppm_per_cm = alpha_hitran_cm_inv * 1e6
```

随后将 HITRAN 仿真谱插值到实验波数点。

### 8.2 HITRAN 扣除预览

为了检查 HITRAN 仿真与实测损耗的相对关系，可以先只输出：

```text
loss_minus_hitran_ppm_per_cm = loss_ppm_per_cm - hitran_o2_loss_ppm_per_cm
```

该模式不进行 CIA 基线拟合，只用于观察扣除 O2 窄线吸收后的背景形状。

输出文件为：

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_hitran_subtracted.csv
```

### 8.3 O2 吸收区识别

程序根据 HITRAN 仿真损耗自动生成吸收区 mask。设 HITRAN 仿真谱最大值为：

```text
H_max = max(hitran_o2_loss_ppm_per_cm)
```

默认阈值为最大值的 1%：

```text
threshold = 0.01 * H_max
```

满足以下条件的点被认为属于 O2 窄吸收区：

```text
hitran_o2_loss_ppm_per_cm >= threshold
```

为了避免只剔除吸收线中心而保留线翼附近的残余影响，程序会将 mask 在波数轴上向两侧各扩展：

```text
0.05 cm-1
```

最终：

```text
o2_absorption_mask = True
```

表示该点被判定为 O2 吸收区，不参与 CIA baseline 拟合；

```text
o2_fit_used = True
```

表示该点为非吸收区，参与 CIA baseline 拟合。

### 8.4 O2 CIA baseline 拟合

O2 的基线拟合仍在 `loss_ppm_per_cm` 域进行，拟合算法与 Ar 的滑动窗口局部多项式/PCHIP 锚点连接一致。区别是 O2 只使用：

```text
finite data AND NOT o2_absorption_mask
```

这些非吸收区点作为拟合点。

当前 O2 500 Torr 推荐参数为：

```text
fit_window = 8 cm-1
fit_step   = 1 cm-1
fit_order  = 2
fit_sigma  = 2
fit_smooth = 2 cm-1
```

输出的 CIA 基线列为：

```text
cia_baseline_loss_ppm_per_cm
```

为了与 Ar 输出保持一致，也同步写入：

```text
loss_fit_ppm_per_cm = cia_baseline_loss_ppm_per_cm
tau_fit_us = (1e12 / c) / cia_baseline_loss_ppm_per_cm
```

残差定义为：

```text
cia_baseline_residual_ppm_per_cm = loss_ppm_per_cm - cia_baseline_loss_ppm_per_cm
loss_residual_ppm_per_cm         = loss_ppm_per_cm - loss_fit_ppm_per_cm
tau_residual_us                  = tau_us - tau_fit_us
```

O2 Step 2 的完整输出为：

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv
```

纯 CIA baseline 的简化输出为：

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_cia_baseline.csv
```

该简化文件主要保留波数、CIA baseline、等效衰荡时间、HITRAN 条件和 mask 信息，便于后续只针对 CIA 基线进行分析。

## 9. Step 2 图像输出

每个 CIA 数据集均输出一张两面板图：

```text
continuum_spectrum.png
```

上面板为处理后的衰荡时间：

```text
tau_us 和 tau_fit_us
```

下面板为 loss：

```text
loss_ppm_per_cm 和 loss_fit_ppm_per_cm
```

对于 O2 数据，下面板还会区分被用于 baseline 拟合的点和由 HITRAN 判断为 O2 吸收区的点，并可同时显示 HITRAN 仿真损耗或 HITRAN 扣除结果。

## 10. 后处理：lmfit 三次 B-spline 平滑

Step 2 后，如果处理后的 `tau_fit_us` 仍存在局部小折点，可进一步对处理后的衰荡时间进行平滑样条拟合。该步骤属于后处理，用于获得更平滑的连续基线，不改变原始 Step 1 衰荡时间。

### 10.1 B-spline 模型

使用三次 B-spline 表示平滑后的衰荡时间：

```text
tau_spline(nu) = sum_j c_j B_j(nu)
```

其中 `B_j(nu)` 为三次 B-spline 基函数，`c_j` 为待优化系数。内部节点由 `knots_every` 控制。例如 `knots_every = 15` 表示大约每 15 cm-1 放置一个内部节点。

### 10.2 最小二乘目标函数

程序先用线性最小二乘给出初始系数，然后用 lmfit 的 least-squares 优化。目标函数为：

```text
r_i = [tau_spline(nu_i) - tau_i] * w_i
```

如果提供误差列，则权重为：

```text
w_i = 1 / sigma_i
```

否则所有点权重相同。

为了抑制节点之间的锯齿状变化，可以加入二阶差分平滑惩罚：

```text
sqrt(lambda) * Delta^2 c_j
```

因此整体优化残差由数据残差和平滑惩罚共同组成。参数 `smooth_lambda` 越大，曲线越平滑；但过大的平滑会压低真实缓慢结构。

### 10.3 全局和分区域两种方式

Ar 数据目前采用全局样条：

```text
knots_every = 15 cm-1
smooth_lambda = 0.1
```

O2 数据可按不同波段使用不同节点间隔。例如：

```text
9100-9200 cm-1: knots_every = 12
9200-9600 cm-1: knots_every = 10
9600-9900 cm-1: knots_every = 15
```

分区域拟合时，每个区域独立拟合。为避免区域边界不连续，程序在每个目标区域两侧额外加入一段 buffer 作为拟合上下文：

```text
anchor_width = 5 cm-1
```

拟合时使用：

```text
[region_start - anchor_width, region_end + anchor_width]
```

但最终只把目标区域内部的数据写回输出列。

### 10.4 衍生列同步

如果样条结果覆盖 `tau_fit_us`，程序会同步更新：

```text
loss_fit_ppm_per_cm = (1e12 / c) / tau_fit_us
loss_residual_ppm_per_cm = loss_ppm_per_cm - loss_fit_ppm_per_cm
tau_residual_us = tau_us - tau_fit_us
```

如果不覆盖原列，则样条结果默认写入新列：

```text
tau_fit_us_lmfit_spline
```

这样便于与原始 Step 2 拟合结果比较。

## 11. 最终数据重采样

为了便于不同数据集比较，最终结果可重采样到统一波数网格。当前使用的网格为：

```text
9120-9820 cm-1
step = 0.01 cm-1
```

重采样默认使用 PCHIP 插值。PCHIP 在保持曲线形状方面比普通三次样条更稳定，不容易在缓慢变化的 CIA 基线上引入过冲。若输入文件只有两列，则默认第 1 列为波数，第 2 列为衰荡时间：

```text
wavenumber, tau_us
```

对于 CSV 中的数值列，程序逐列插值；对于文本列或布尔列，默认使用最近邻方式保留元数据。若输出网格端点略微超出原始数据覆盖范围，默认使用最近有效端点值填充边界。

## 12. 输出文件说明

### 12.1 Step 1 输出

```text
output/results/ringdown/CIA/{temperature}/{gas pressure}/ringdown_results.csv
```

主要列：

```text
wavenumber
tau_mean
tau_std
temperature
pressure
```

### 12.2 Step 2 初始 loss 输出

```text
output/results/continuum/CIA/{temperature}/{gas pressure}/continuum_spectrum supplement.csv
```

主要列：

```text
wavenumber
tau_us
tau_stats_us
pressure_torr
temperature_c
loss_ppm_per_cm
loss_stats_ppm_per_cm
reference_loss_ppm_per_cm
alpha_ppm_per_cm
```

### 12.3 Ar Step 2 输出

```text
output/results/continuum/CIA/273K/Ar 500Torr/continuum_step2_fit.csv
```

主要列：

```text
loss_fit_ppm_per_cm
loss_residual_ppm_per_cm
tau_fit_us
tau_residual_us
step2_fit_mode
```

其中 `step2_fit_mode = ar`。

### 12.4 O2 Step 2 输出

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_fit.csv
```

除 Ar 输出列外，O2 输出还包括：

```text
hitran_o2_loss_ppm_per_cm
hitran_o2_absorption_cm_inv
loss_minus_hitran_ppm_per_cm
tau_equiv_after_hitran_us
hitran_temperature_c
hitran_temperature_k
hitran_pressure_torr
hitran_pressure_atm
hitran_step_cm1
o2_absorption_mask
o2_fit_used
hitran_mask_threshold_ppm_per_cm
hitran_mask_ratio
hitran_mask_margin_cm1
cia_baseline_loss_ppm_per_cm
cia_baseline_residual_ppm_per_cm
```

其中 `step2_fit_mode = o2`。

### 12.5 O2 HITRAN 扣除预览输出

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_hitran_subtracted.csv
```

主要用于检查：

```text
loss_minus_hitran_ppm_per_cm
```

### 12.6 O2 纯 CIA baseline 输出

```text
output/results/continuum/CIA/273K/O2 500Torr/continuum_step2_cia_baseline.csv
```

该文件用于后续只分析 O2 CIA baseline。

### 12.7 最终 0.01 cm-1 网格输出

典型输出文件：

```text
ar_tau_lmfit_interp_9120_9820_step0p01.csv
o2_cia_baseline_interp_9120_9820_step0p01.csv
o2_tau_lmfit_interp_9120_9820_step0p01.csv
```

## 13. 可直接放入论文的方法描述草稿

下面文字可作为论文方法部分的初稿，根据实际投稿格式再精简。

原始 CRDS 数据以单个波数点为单位存储，每个文件包含同一波数下多次衰荡事件的拟合衰荡时间、拟合残差、腔体温度和压力。首先根据文件名读取激光波数，并对所有波数点按波数排序。随后检查相邻波数间隔，剔除明显异常的跳点或重复点。对于每个波数点的多次衰荡事件，采用迭代 3 sigma clipping 剔除离群值，最多迭代 5 次。剔除后若有效衰荡事件数不足 5 个，则该波数点不参与后续分析。保留事件的平均值作为该波数点的代表性衰荡时间，标准差用于表征同一波数点内的重复测量离散度。

平均衰荡时间被转换为腔内总损耗。对于衰荡时间 tau，单位为 microsecond，总损耗以 ppm/cm 表示为：

```text
loss = (1e12 / c) / tau
```

其中 `c = 2.99792458e10 cm/s`。若提供参考空腔衰荡时间或参考损耗谱，则吸收系数由样品损耗减去参考损耗得到；若未提供参考谱，则后续分析直接基于腔内总损耗进行。

Ar 数据中没有明显 O2 窄线吸收，因此其连续背景直接在 loss 域拟合。具体而言，在波数轴上以固定步长设置局部拟合中心，每个中心取有限宽度的波数窗口，并在窗口内采用 robust 局部多项式拟合。每个局部拟合只保留窗口中心处的拟合值作为基线锚点，所有锚点再通过 PCHIP 插值连接为全波段连续基线。为降低局部测量噪声造成的小尺度波动，可对该基线进一步进行 Savitzky-Golay 平滑。Ar 500 Torr 数据采用 40 cm-1 的局部窗口、5 cm-1 的锚点间隔、二阶多项式、4 sigma robust clipping 和 20 cm-1 的额外平滑宽度。

O2 数据含有大量窄线吸收，不能直接使用全部 loss 点拟合连续基线。为去除窄线吸收的影响，首先在与实验相同的温度和压力条件下使用 HITRAN2024 谱线参数和 Voigt 线型计算 O2 吸收损耗。温度和压力分别取实验记录中的中位数，HITRAN 仿真步长为 0.002 cm-1。仿真吸收谱被插值到实验波数点后，用于识别 O2 吸收区。当 HITRAN 仿真损耗大于其峰值的 1% 时，相应波数点被标记为 O2 吸收区；同时将该 mask 在波数轴上向两侧各扩展 0.05 cm-1，以剔除线翼附近可能受吸收影响的点。随后只使用未被标记为 O2 吸收区的实验点，在 loss 域按照与 Ar 相同的滑动窗口局部多项式/PCHIP 锚点方法拟合缓慢变化的 CIA baseline。O2 500 Torr 数据采用 8 cm-1 的局部窗口、1 cm-1 的锚点间隔、二阶多项式、2 sigma robust clipping 和 2 cm-1 的额外平滑宽度。

对于拟合后的连续衰荡时间曲线，必要时进一步使用三次 B-spline 进行平滑后处理。B-spline 的系数由 lmfit 最小二乘优化获得，目标函数由数据残差和可选的二阶系数差分平滑惩罚组成。该处理用于减少局部连接处的小折点，并保持宽波段 CIA 基线的缓慢变化趋势。Ar 数据使用全局三次 B-spline，内部节点间隔为 15 cm-1，平滑惩罚系数为 0.1。O2 数据可根据不同波段的数据质量和曲率使用分区域节点间隔，每个区域独立拟合，并在区域两侧加入 5 cm-1 的上下文宽度以减小边界效应。

最终，为便于不同气体和不同处理结果之间的比较，将处理后的衰荡时间或 CIA baseline 插值到统一波数网格。统一网格范围为 9120-9820 cm-1，间隔为 0.01 cm-1。插值采用 PCHIP 方法，以避免普通三次样条在缓慢变化曲线上的过冲。最终输出的等间隔数据用于后续作图和定量比较。

## 14. 参数汇总

| 项目 | Ar 500 Torr | O2 500 Torr |
| --- | --- | --- |
| Step 1 离群值剔除 | 3 sigma clipping | 3 sigma clipping |
| 最少衰荡事件数 | 5 | 5 |
| loss 计算 | `(1e12/c)/tau` | `(1e12/c)/tau` |
| Step 2 模式 | `ar` | `o2` |
| 是否用 HITRAN mask | 否 | 是 |
| HITRAN 仿真步长 | 不适用 | 0.002 cm-1 |
| HITRAN mask 阈值 | 不适用 | 峰值的 1% |
| HITRAN mask 扩展 | 不适用 | ±0.05 cm-1 |
| Step 2 拟合域 | loss | loss |
| Step 2 window | 40 cm-1 | 8 cm-1 |
| Step 2 step | 5 cm-1 | 1 cm-1 |
| Step 2 polynomial order | 2 | 2 |
| Step 2 sigma | 4 | 2 |
| Step 2 smooth | 20 cm-1 | 2 cm-1 |
| lmfit B-spline | 全局，15 cm-1 节点间隔 | 可分区域 |
| 最终插值网格 | 9120-9820 cm-1, 0.01 cm-1 | 9120-9820 cm-1, 0.01 cm-1 |
