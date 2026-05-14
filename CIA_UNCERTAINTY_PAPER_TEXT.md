# CIA 不确定度分析论文表述

以下文字按照论文“实验数据处理与不确定度分析”部分的写法组织，可根据期刊格式对公式编号和单位格式再做调整。

## 中文正文版本

本研究对 O2-O2 碰撞诱导吸收系数的不确定度进行了逐步传播分析。首先，对每个波数点处记录的多次衰荡事件进行统计处理。为避免偶发坏点对平均衰荡时间产生影响，先对同一波数点处的衰荡时间数据进行异常值剔除，然后计算平均衰荡时间及其 A 类标准不确定度。若某一波数点处保留的衰荡时间为 \(\tau_i\)，事件数为 \(N\)，则平均衰荡时间为

\[
\bar{\tau}=\frac{1}{N}\sum_{i=1}^{N}\tau_i ,
\]

其标准不确定度由平均值标准误给出：

\[
u_{\bar{\tau}}=\frac{s_{\tau}}{\sqrt{N}},
\]

其中 \(s_{\tau}\) 为该波数点处衰荡时间的样本标准偏差。

由于最终用于计算吸收系数的是经过连续基线拟合后的衰荡时间曲线，而不是原始平均衰荡时间点，因此进一步采用 Monte Carlo 方法将原始衰荡时间的不确定度传播到拟合结果。具体而言，在每一次 Monte Carlo 计算中，按照每个波数点的标准不确定度 \(u_{\bar{\tau}}\) 对平均衰荡时间进行随机扰动，并对扰动后的数据执行与正式数据完全相同的基线拟合流程。重复计算后，每个波数点处拟合衰荡时间的标准偏差被作为该点拟合衰荡时间的不确定度。

O2 样品的吸收系数由 O2 和 Ar 条件下的衰荡时间差异计算。对于波数 \(\tilde{\nu}\)，吸收系数表示为

\[
\alpha(\tilde{\nu})=\frac{1}{c}
\left[
\frac{1}{\tau_{\mathrm{O_2}}(\tilde{\nu})}
-
\frac{1}{\tau_{\mathrm{Ar}}(\tilde{\nu})}
\right],
\]

其中 \(c\) 为真空光速，\(\tau_{\mathrm{O_2}}\) 和 \(\tau_{\mathrm{Ar}}\) 分别为 O2 和 Ar 条件下的拟合衰荡时间。实际计算中衰荡时间以微秒为单位，因此上式中的单位换算已在程序中统一处理。假设 O2 和 Ar 衰荡时间的不确定度彼此独立，吸收系数的标准不确定度由一阶误差传播得到：

\[
u_{\alpha}=\frac{1}{c}
\left[
\left(\frac{u_{\tau,\mathrm{O_2}}}{\tau_{\mathrm{O_2}}^2}\right)^2
+
\left(\frac{u_{\tau,\mathrm{Ar}}}{\tau_{\mathrm{Ar}}^2}\right)^2
\right]^{1/2}.
\]

其中 \(u_{\tau,\mathrm{O_2}}\) 和 \(u_{\tau,\mathrm{Ar}}\) 分别为 O2 和 Ar 拟合衰荡时间的标准不确定度。

O2 分子数密度以 amagat 为单位表示，并由理想气体关系计算：

\[
\rho = \frac{p}{p_0}\frac{T_0}{T},
\]

其中 \(p_0=760\ \mathrm{Torr}\)，\(T_0=273.15\ \mathrm{K}\)，\(p\) 和 \(T\) 分别为 O2 样品的实验压力和温度。压力和温度均从原始实验记录中统计得到，并在统计前剔除明显偏离的异常值。温度标准不确定度取 \(0.001\ \mathrm{K}\)。压力不确定度根据 CTR100 真空计精度确定，在本实验压力范围内取 \(0.20\%\) of reading，并按矩形分布处理，因此压力标准不确定度为

\[
u_p=\frac{0.002p}{\sqrt{3}}.
\]

由 \(\rho=(p/p_0)(T_0/T)\) 可得数密度的相对标准不确定度：

\[
\frac{u_{\rho}}{\rho}
=
\left[
\left(\frac{u_p}{p}\right)^2
+
\left(\frac{u_T}{T}\right)^2
\right]^{1/2}.
\]

O2-O2 二元碰撞吸收系数定义为

\[
B(\tilde{\nu})=\frac{\alpha(\tilde{\nu})}{\rho^2}.
\]

假设吸收系数和数密度的不确定度相互独立，则二元碰撞吸收系数的标准不确定度为

\[
u_B=
\left[
\left(\frac{u_{\alpha}}{\rho^2}\right)^2
+
\left(\frac{2B u_{\rho}}{\rho}\right)^2
\right]^{1/2}.
\]

等价地，其相对标准不确定度可写为

\[
\frac{u_B}{B}
=
\left[
\left(\frac{u_{\alpha}}{\alpha}\right)^2
+
\left(\frac{2u_{\rho}}{\rho}\right)^2
\right]^{1/2}.
\]

由于 \(B\) 与 \(\rho^2\) 成反比，数密度的相对不确定度在传播到 \(B\) 时会被放大两倍。因此，在本实验条件下，二元碰撞吸收系数的不确定度主要由压力测量引起的数密度不确定度贡献。

积分强度定义为

\[
S=\int_{\tilde{\nu}_1}^{\tilde{\nu}_2}B(\tilde{\nu})\,d\tilde{\nu},
\]

其中本工作积分范围为 \(9120-9820\ \mathrm{cm^{-1}}\)。离散数据采用梯形积分计算：

\[
S \simeq \sum_i w_i B_i,
\]

其中 \(w_i\) 为梯形积分权重。积分强度的不确定度采用上下边界积分法。首先构造二元碰撞吸收系数的上下边界：

\[
B_{\mathrm{upper}}(\tilde{\nu})=B(\tilde{\nu})+u_B(\tilde{\nu}),
\]

\[
B_{\mathrm{lower}}(\tilde{\nu})=B(\tilde{\nu})-u_B(\tilde{\nu}).
\]

然后分别进行积分：

\[
S_{\mathrm{upper}}=\int B_{\mathrm{upper}}(\tilde{\nu})\,d\tilde{\nu},
\]

\[
S_{\mathrm{lower}}=\int B_{\mathrm{lower}}(\tilde{\nu})\,d\tilde{\nu}.
\]

积分强度的标准不确定度取上下边界积分差值的一半：

\[
u_S=\frac{S_{\mathrm{upper}}-S_{\mathrm{lower}}}{2}.
\]

最终结果采用扩展不确定度表示：

\[
U_S=k u_S,
\]

其中覆盖因子取 \(k=2\)。因此，表格中括号内给出的不确定度为扩展不确定度，近似对应 95% 的覆盖概率。

## 可放在表注中的说明

表中积分强度单位为 \(\mathrm{cm^{-2}\ amagat^{-2}}\)。括号内为扩展不确定度，覆盖因子 \(k=2\)。积分强度在 \(9120-9820\ \mathrm{cm^{-1}}\) 范围内由梯形积分得到，其不确定度由 \(B(\tilde{\nu})\pm u_B(\tilde{\nu})\) 的上下边界积分差值确定。

## 英文表述草稿

The uncertainty of the O2-O2 CIA coefficient was evaluated by a stepwise propagation procedure. For each wavenumber, the mean ring-down time was calculated after outlier rejection, and its type-A standard uncertainty was estimated from the standard error of the mean. The uncertainty of the fitted ring-down-time baseline was then evaluated by a Monte Carlo procedure, in which the mean ring-down times were randomly perturbed according to their standard uncertainties and the same baseline fitting procedure was repeated.

The absorption coefficient was calculated from the difference between the O2 and Ar ring-down times,

\[
\alpha(\tilde{\nu})=\frac{1}{c}
\left[
\frac{1}{\tau_{\mathrm{O_2}}(\tilde{\nu})}
-
\frac{1}{\tau_{\mathrm{Ar}}(\tilde{\nu})}
\right].
\]

Assuming independent uncertainties in the O2 and Ar fitted ring-down times, the standard uncertainty of the absorption coefficient was obtained by first-order error propagation. The O2 number density, expressed in amagat, was calculated from the ideal gas relation \(\rho=(p/760)(273.15/T)\). The temperature standard uncertainty was taken as \(0.001\ \mathrm{K}\), while the pressure uncertainty was determined from the CTR100 gauge specification of \(0.20\%\) of reading and converted to a standard uncertainty assuming a rectangular distribution.

The binary CIA coefficient was calculated as \(B(\tilde{\nu})=\alpha(\tilde{\nu})/\rho^2\). Its standard uncertainty was propagated according to

\[
u_B=
\left[
\left(\frac{u_{\alpha}}{\rho^2}\right)^2
+
\left(\frac{2B u_{\rho}}{\rho}\right)^2
\right]^{1/2}.
\]

The integrated intensity was obtained by trapezoidal integration of \(B(\tilde{\nu})\) over \(9120-9820\ \mathrm{cm^{-1}}\). Its uncertainty was evaluated from the integrated upper and lower bounds, \(B+u_B\) and \(B-u_B\), respectively, as

\[
u_S=\frac{S_{\mathrm{upper}}-S_{\mathrm{lower}}}{2}.
\]

The uncertainties reported in parentheses are expanded uncertainties with a coverage factor of \(k=2\).
