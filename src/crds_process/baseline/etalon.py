"""标准具效应去除 (Etalon Removal)

CRDS 光谱中常存在由光学元件平行面引起的标准具效应（etalon fringes），
表现为叠加在基线上的周期性正弦振荡。本模块使用 lmfit 进行单频/多频
正弦模型拟合，从光谱中识别并去除标准具干涉条纹。

核心思路:
    1. 用 exclude_regions 排除吸收峰区域，仅在基线区域拟合
    2. 正弦 + 多项式模型同时拟合标准具条纹和缓变基线
    3. 用拟合参数在全波段计算并扣除标准具分量（仅正弦部分）

模型:
    y = Σ aᵢ·sin(2π·fᵢ·x + φᵢ) + poly(x)

用法:
    from crds_process.baseline.etalon import EtalonRemover

    remover = EtalonRemover(
        n_etalons=1,
        poly_order=3,
        exclude_regions=[[9385.9, 9386.5]],  # 排除吸收峰区域
    )
    result = remover.fit(wavenumber, tau)
    tau_clean = result.corrected
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from lmfit.model import ModelResult


# ==================================================================
# 数据容器
# ==================================================================
@dataclass
class EtalonFitResult:
    """标准具拟合结果"""

    wavenumber: np.ndarray          # 波数 (全波段)
    original: np.ndarray            # 原始信号
    etalon: np.ndarray              # 标准具分量 (仅正弦部分)
    baseline: np.ndarray            # 基线分量 (多项式 / 常数)
    corrected: np.ndarray           # 去除标准具后的信号
    model_result: ModelResult       # lmfit 拟合结果 (仅基线区域)
    n_etalons: int                  # 标准具分量数
    fit_mask: np.ndarray            # 用于拟合的点 (bool)
    exclude_regions: list = field(default_factory=list)  # 实际使用的排除区域
    components: list[dict] = field(default_factory=list)

    @property
    def residual_std(self) -> float:
        """拟合区域残差的标准差"""
        return float(np.std(self.model_result.residual))

    @property
    def n_fit_points(self) -> int:
        """参与拟合的数据点数"""
        return int(np.sum(self.fit_mask))

    def summary(self) -> str:
        lines = [
            f"Etalon fit: {self.n_etalons} component(s)",
            f"  拟合点数    = {self.n_fit_points} / {len(self.wavenumber)}",
        ]
        if self.exclude_regions:
            for i, reg in enumerate(self.exclude_regions):
                lines.append(f"  排除区域 {i}: [{reg[0]:.5f}, {reg[1]:.5f}] cm⁻¹")
        lines.extend([
            f"  Reduced χ²  = {self.model_result.redchi:.6e}",
            f"  Residual σ  = {self.residual_std:.6e}",
        ])
        for i, comp in enumerate(self.components):
            lines.append(
                f"  Component {i}: "
                f"amp={comp['amplitude']:.4e}, "
                f"freq={comp['frequency']:.4f} cycles/cm⁻¹, "
                f"period={1/comp['frequency']:.5f} cm⁻¹, "
                f"phase={comp['phase']:.4f} rad"
            )
        return "\n".join(lines)


# ==================================================================
# 辅助函数
# ==================================================================
def _sine_component(x: np.ndarray, amplitude: float, frequency: float,
                    phase: float) -> np.ndarray:
    """单个正弦分量: a * sin(2π * f * x + φ)"""
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def _sine_component_am(x: np.ndarray, amp0: float, amp1: float,
                       frequency: float, phase: float,
                       x_center: float) -> np.ndarray:
    """调幅正弦分量: (a0 + a1*(x - x_center)) * sin(2π * f * x + φ)

    振幅随波数线性变化，捕捉标准具振幅的空间不均匀性。
    """
    envelope = amp0 + amp1 * (x - x_center)
    return envelope * np.sin(2.0 * np.pi * frequency * x + phase)


def _build_exclude_mask(x: np.ndarray,
                        exclude_regions: list[list[float]] | None) -> np.ndarray:
    """构建排除掩码 (True = 用于拟合)"""
    mask = np.ones(len(x), dtype=bool)
    if exclude_regions:
        for low, high in exclude_regions:
            mask &= ~((x >= low) & (x <= high))
    return mask


def hitran_detect_absorption(
    wavenumber: np.ndarray,
    temperature: float,
    pressure_torr: float,
    molecule: int = 7,
    isotopologue: int = 1,
    hitran_table: str | None = None,
    hitran_dir: str | None = None,
    threshold_ratio: float = 0.01,
    margin: float = 0.05,
    step: float = 0.002,
) -> list[list[float]]:
    """基于 HITRAN 模拟自动检测吸收峰区域

    使用 HAPI 计算给定温度/压力下的吸收系数谱，
    在吸收系数 > peak * threshold_ratio 的波数区域标记为吸收峰。

    Parameters
    ----------
    wavenumber : np.ndarray
        实测波数数组 (cm⁻¹)，用于确定模拟范围
    temperature : float
        温度 (°C)
    pressure_torr : float
        压力 (Torr)
    molecule : int
        HITRAN 分子编号 (默认 7 = O₂)
    isotopologue : int
        HITRAN 同位素编号 (默认 1 = ¹⁶O₂)
    hitran_table : str | None
        HAPI 本地表名。None 则自动搜索或下载
    hitran_dir : str | None
        HITRAN 本地数据目录。None 则使用默认路径
    threshold_ratio : float
        吸收系数阈值 = peak * threshold_ratio (0~1)
        较小的值 → 排除区域更大；推荐 0.005~0.05
    margin : float
        检测到的区域向两侧扩展的余量 (cm⁻¹)
    step : float
        模拟波数步长 (cm⁻¹)

    Returns
    -------
    list[list[float]]
        排除区域列表 [[low1, high1], ...]
    """
    import os
    import sys

    # ── 静默导入 HAPI ──
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        import hapi
    except ImportError:
        sys.stdout.close()
        sys.stdout = _stdout
        raise ImportError("需要安装 hitran-api: pip install hitran-api")
    finally:
        sys.stdout.close()
        sys.stdout = _stdout

    # HITRAN 数据目录
    if hitran_dir is None:
        # 尝试项目默认路径
        _candidates = [
            os.path.join(os.getcwd(), "data", "hitran"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hitran"),
        ]
        for c in _candidates:
            c = os.path.abspath(c)
            if os.path.isdir(c):
                hitran_dir = c
                break
        if hitran_dir is None:
            hitran_dir = os.path.join(os.getcwd(), "data", "hitran")
            os.makedirs(hitran_dir, exist_ok=True)

    # 保存/恢复当前目录 (hapi 会 chdir)
    original_dir = os.getcwd()
    os.chdir(hitran_dir)

    try:
        # 静默初始化 hapi
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            hapi.db_begin(hitran_dir)
        finally:
            sys.stdout.close()
            sys.stdout = _stdout

        # 确定表名
        if hitran_table is None:
            # 自动搜索已有表
            mol_prefix = {7: "O2", 6: "CH4", 2: "CO2", 1: "H2O"}.get(molecule, f"M{molecule}")
            for tname in hapi.LOCAL_TABLE_CACHE:
                if tname.startswith(mol_prefix):
                    hitran_table = tname
                    break

            if hitran_table is None:
                # 需要下载
                hitran_table = f"{mol_prefix}_iso{isotopologue}"
                wn_min = float(np.min(wavenumber)) - 2.0
                wn_max = float(np.max(wavenumber)) + 2.0
                print(f"  [HITRAN] 下载 {mol_prefix} 数据 ({wn_min:.0f}-{wn_max:.0f} cm⁻¹)...")
                hapi.fetch(
                    TableName=hitran_table,
                    M=molecule,
                    I=isotopologue,
                    numin=wn_min,
                    numax=wn_max,
                )

        # ── 模拟吸收系数 ──
        wn_min = float(np.min(wavenumber)) - 0.5
        wn_max = float(np.max(wavenumber)) + 0.5

        T_kelvin = temperature + 273.15
        p_atm = pressure_torr / 760.0  # Torr → atm

        _stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            nu_sim, alpha_sim = hapi.absorptionCoefficient_Voigt(
                SourceTables=hitran_table,
                Environment={"T": T_kelvin, "p": p_atm},
                WavenumberRange=[wn_min, wn_max],
                WavenumberStep=step,
                HITRAN_units=False,
            )
        finally:
            sys.stdout.close()
            sys.stdout = _stdout

        alpha_sim = alpha_sim.flatten()

    finally:
        os.chdir(original_dir)

    # ── 检测吸收峰区域 ──
    alpha_peak = float(np.max(alpha_sim))
    if alpha_peak <= 0:
        return []

    threshold = alpha_peak * threshold_ratio
    above = nu_sim[alpha_sim > threshold]

    if len(above) == 0:
        return []

    # 合并连续区域
    dx_sim = step * 3
    regions: list[list[float]] = []
    region_start = float(above[0])
    region_end = float(above[0])

    for i in range(1, len(above)):
        if above[i] - above[i - 1] > dx_sim:
            regions.append([region_start - margin, region_end + margin])
            region_start = float(above[i])
        region_end = float(above[i])

    regions.append([region_start - margin, region_end + margin])

    return regions


def _estimate_dominant_frequency(x: np.ndarray, y: np.ndarray,
                                 n_freq: int = 1) -> list[float]:
    """通过 Lomb-Scargle 周期图估计信号中的主要频率

    适用于非均匀采样（如排除吸收峰后不连续的基线区域）。
    """
    from scipy.signal import detrend, lombscargle

    y_dt = detrend(y, type='linear')

    # 频率搜索范围
    dx_median = np.median(np.diff(x))
    f_max = 0.5 / dx_median          # Nyquist
    f_min = 1.0 / (x[-1] - x[0])    # 最低可分辨频率
    test_freqs = np.linspace(f_min, f_max, 10000)

    # Lomb-Scargle (angular frequency)
    angular_freqs = 2.0 * np.pi * test_freqs
    power = lombscargle(x, y_dt, angular_freqs, normalize=True)

    result = []
    for _ in range(n_freq):
        idx = int(np.argmax(power))
        if power[idx] > 0:
            result.append(float(test_freqs[idx]))
        # 屏蔽该峰附近
        delta = max(3, int(len(test_freqs) * 0.02))
        low = max(0, idx - delta)
        high = min(len(power), idx + delta + 1)
        power[low:high] = 0

    if not result:
        result = [1.0]
    return result


# ==================================================================
# 标准具去除器
# ==================================================================
class EtalonRemover:
    """标准具效应拟合与去除 (迭代交替优化)

    策略:
        1. 多项式去趋势 → 正弦拟合 → 扣除正弦 → 重新拟合多项式 → …
        2. 迭代收敛后，仅从原始信号中扣除正弦分量

    Parameters
    ----------
    n_etalons : int
        标准具正弦分量数目 (通常 1~3)
    freq_hints : list[float] | None
        频率初始猜测值 (cycles/cm⁻¹)，None 则自动估计
    exclude_regions : list[list[float]] | str | None
        排除的波数区域。支持:
        - list: 手动指定 [[low1, high1], ...]
        - "hitran": 基于 HITRAN 模拟自动检测吸收峰区域 (推荐)
        - None: 不排除任何区域
    poly_order : int
        去趋势多项式阶数 (推荐 1~3; 1 通常最优)
    n_iter : int
        交替迭代次数 (推荐 3~5)
    flatten_baseline : bool
        若为 True，corrected 中同时扣除基线趋势（保留均值），
        使输出基线平坦化。默认 True。
    hitran_kwargs : dict | None
        传给 hitran_detect_absorption 的额外参数
        (molecule, isotopologue, threshold_ratio, margin 等)
    max_iter : int
        lmfit 每次正弦拟合的最大迭代次数
    """

    def __init__(
        self,
        n_etalons: int = 1,
        freq_hints: list[float] | None = None,
        exclude_regions: list[list[float]] | str | None = "hitran",
        poly_order: int = 1,
        n_iter: int = 5,
        flatten_baseline: bool = True,
        hitran_kwargs: dict | None = None,
        max_iter: int = 20000,
    ):
        if n_etalons < 1:
            raise ValueError("n_etalons 必须 >= 1")
        self.n_etalons = n_etalons
        self.freq_hints = freq_hints
        self.exclude_regions = exclude_regions
        self.poly_order = poly_order
        self.n_iter = n_iter
        self.flatten_baseline = flatten_baseline
        self.hitran_kwargs = hitran_kwargs or {}
        self.max_iter = max_iter

    def _build_sine_model(
        self, x: np.ndarray, y: np.ndarray,
        freq_init: list[float] | None = None,
    ) -> tuple[Model, Parameters]:
        """构建纯正弦 lmfit 模型"""

        if freq_init is None:
            if self.freq_hints and len(self.freq_hints) >= self.n_etalons:
                freq_init = self.freq_hints[:self.n_etalons]
            else:
                freq_init = _estimate_dominant_frequency(x, y, self.n_etalons)

        model = Model(_sine_component, prefix="e0_")
        for i in range(1, self.n_etalons):
            model += Model(_sine_component, prefix=f"e{i}_")

        params = model.make_params()

        amp_est = float(np.std(y) * np.sqrt(2))
        if amp_est < 1e-10:
            amp_est = float((np.max(y) - np.min(y)) / 2.0)

        for i in range(self.n_etalons):
            pfx = f"e{i}_"
            f0 = freq_init[i] if i < len(freq_init) else freq_init[-1] * (i + 1)
            params[f"{pfx}amplitude"].set(value=amp_est / (i + 1), min=0)
            params[f"{pfx}frequency"].set(value=f0, min=f0 * 0.5, max=f0 * 2.0)
            params[f"{pfx}phase"].set(value=0, min=-np.pi, max=np.pi)

        return model, params

    def fit(
        self, wavenumber: np.ndarray, signal: np.ndarray,
        exclude_regions: list[list[float]] | None = None,
        temperature: float | None = None,
        pressure_torr: float | None = None,
    ) -> EtalonFitResult:
        """迭代交替优化拟合

        Parameters
        ----------
        wavenumber : np.ndarray
            波数数组 (cm⁻¹)
        signal : np.ndarray
            待处理信号
        exclude_regions : list[list[float]] | None
            临时排除区域，与构造函数中的合并
        temperature : float | None
            温度 (°C)，仅 exclude_regions="hitran" 时需要
        pressure_torr : float | None
            压力 (Torr)，仅 exclude_regions="hitran" 时需要

        Returns
        -------
        EtalonFitResult
        """
        x = np.asarray(wavenumber, dtype=float)
        y = np.asarray(signal, dtype=float)

        if len(x) != len(y):
            raise ValueError("wavenumber 和 signal 长度不一致")

        # 解析排除区域
        if self.exclude_regions == "hitran":
            if temperature is None or pressure_torr is None:
                raise ValueError(
                    "exclude_regions='hitran' 需要提供 temperature (°C) "
                    "和 pressure_torr (Torr)"
                )
            auto_regions = hitran_detect_absorption(
                x, temperature=temperature, pressure_torr=pressure_torr,
                **self.hitran_kwargs,
            )
        elif isinstance(self.exclude_regions, list):
            auto_regions = list(self.exclude_regions)
        else:
            auto_regions = []

        # 合并临时排除区域
        all_exclude = auto_regions[:]
        if exclude_regions:
            all_exclude.extend(exclude_regions)

        fit_mask = _build_exclude_mask(x, all_exclude)
        x_fit = x[fit_mask]
        y_fit = y[fit_mask]

        if len(x_fit) < 20:
            raise ValueError(f"排除后仅剩 {len(x_fit)} 个拟合点，不足以拟合")

        # 波数中心化
        x_center = float(np.mean(x))
        x_c = x - x_center
        x_fit_c = x_fit - x_center

        # ── 迭代交替优化 ──
        etalon_on_fit = np.zeros_like(x_fit)  # 当前基线区域上的标准具估计
        last_result = None
        prev_params: Parameters | None = None  # 上一轮拟合参数（热启动）

        for iteration in range(self.n_iter):
            # Step A: 多项式去趋势 (在扣除当前标准具估计后的数据上)
            y_for_poly = y_fit - etalon_on_fit
            poly_coeffs = np.polyfit(x_fit_c, y_for_poly, self.poly_order)
            baseline_fit = np.polyval(poly_coeffs, x_fit_c)

            # Step B: 正弦拟合 (在去趋势残差上)
            residual = y_fit - baseline_fit

            if prev_params is None:
                # 第一轮：自动估计频率
                model, params = self._build_sine_model(x_fit, residual)
            else:
                # 后续轮：用上一轮结果热启动
                model, params = self._build_sine_model(x_fit, residual)
                for pname in prev_params:
                    if pname in params:
                        params[pname].set(value=prev_params[pname].value)

            last_result = model.fit(residual, params, x=x_fit, max_nfev=self.max_iter)
            prev_params = last_result.params

            # 更新标准具估计
            etalon_on_fit = np.zeros_like(x_fit)
            for i in range(self.n_etalons):
                pfx = f"e{i}_"
                amp = last_result.params[f"{pfx}amplitude"].value
                freq = last_result.params[f"{pfx}frequency"].value
                phase = last_result.params[f"{pfx}phase"].value
                etalon_on_fit += _sine_component(x_fit, amp, freq, phase)

        # ── 最终结果 ──
        # 最终多项式基线（全波段）
        y_for_poly_final = y_fit - etalon_on_fit
        poly_coeffs_final = np.polyfit(x_fit_c, y_for_poly_final, self.poly_order)
        baseline_full = np.polyval(poly_coeffs_final, x_c)

        # 标准具分量（全波段）
        etalon_full = np.zeros_like(x)
        components_info: list[dict] = []
        for i in range(self.n_etalons):
            pfx = f"e{i}_"
            amp = last_result.params[f"{pfx}amplitude"].value
            freq = last_result.params[f"{pfx}frequency"].value
            phase = last_result.params[f"{pfx}phase"].value
            etalon_full += _sine_component(x, amp, freq, phase)
            components_info.append({
                "amplitude": amp,
                "frequency": freq,
                "phase": phase,
            })

        corrected = y - etalon_full

        # 可选：扣除基线趋势（保留均值），使输出平坦
        if self.flatten_baseline:
            baseline_trend = baseline_full - float(np.mean(baseline_full))
            corrected = corrected - baseline_trend

        return EtalonFitResult(
            wavenumber=x,
            original=y,
            etalon=etalon_full,
            baseline=baseline_full,
            corrected=corrected,
            model_result=last_result,
            n_etalons=self.n_etalons,
            fit_mask=fit_mask,
            exclude_regions=all_exclude,
            components=components_info,
        )

    def fit_df(
        self,
        df: pd.DataFrame,
        wavenumber_col: str = "wavenumber",
        signal_col: str = "alpha",
        temperature_col: str = "temperature",
        pressure_col: str = "pressure",
        inplace: bool = False,
    ) -> tuple[pd.DataFrame, EtalonFitResult]:
        """对 DataFrame 去除标准具效应

        当 exclude_regions="hitran" 时，自动从 DataFrame 中提取
        温度和压力的平均值用于 HITRAN 模拟。
        """
        out = df if inplace else df.copy()

        # 自动提取温度和压力
        temperature = None
        pressure_torr = None
        if self.exclude_regions == "hitran":
            if temperature_col in df.columns:
                temperature = float(df[temperature_col].mean())
            if pressure_col in df.columns:
                pressure_torr = float(df[pressure_col].mean())

        result = self.fit(
            out[wavenumber_col].values,
            out[signal_col].values,
            temperature=temperature,
            pressure_torr=pressure_torr,
        )
        out[f"{signal_col}_etalon"] = result.etalon
        out[f"{signal_col}_no_etalon"] = result.corrected
        return out, result


# ==================================================================
# 可视化
# ==================================================================
def plot_etalon_removal(
    result: EtalonFitResult,
    title: str = "Etalon Removal",
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 10),
):
    """绘制标准具去除前后的对比图 (5 面板)

    1. 原始信号 + 全拟合 (标注排除区域)
    2. 标准具分量 (正弦)
    3. 去除标准具后的信号
    4. 拟合区域残差
    5. FFT 功率谱对比
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wn = result.wavenumber
    mask = result.fit_mask

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # (0,0) 原始 + 全拟合 (基线 + 标准具)
    ax = axes[0, 0]
    ax.plot(wn, result.original, ".", ms=2, alpha=0.4, label="Original")
    full_fit = result.baseline + result.etalon
    ax.plot(wn, full_fit, "r-", lw=1.2, label="Baseline + Etalon")
    ax.plot(wn, result.baseline, "--", color="orange", lw=1, alpha=0.7, label="Baseline (poly)")
    # 标注排除区域
    if not np.all(mask):
        excluded = wn[~mask]
        if len(excluded) > 0:
            ax.axvspan(excluded.min(), excluded.max(),
                       alpha=0.1, color="red", label="Excluded region")
    ax.set_ylabel("Signal")
    ax.set_title("Original + Fit")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # (0,1) 标准具分量
    ax = axes[0, 1]
    ax.plot(wn, result.etalon, "g-", lw=1)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("Etalon component")
    ax.set_title(f"Etalon ({result.n_etalons} sine component)")
    ax.grid(True, alpha=0.3)

    # (1,0) 去除标准具后 (全波段)
    ax = axes[1, 0]
    ax.plot(wn, result.corrected, ".", ms=2, color="steelblue", alpha=0.5)
    ax.set_ylabel("Corrected signal")
    ax.set_title("After etalon removal (full)")
    ax.grid(True, alpha=0.3)

    # (1,1) 拟合区域残差
    ax = axes[1, 1]
    residual = result.original[mask] - (result.baseline[mask] + result.etalon[mask])
    ax.plot(wn[mask], residual, ".", ms=2, color="tomato", alpha=0.5)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("Residual")
    ax.set_title(f"Fit residual (σ={np.std(residual):.4f})")
    ax.grid(True, alpha=0.3)

    # (2,0) FFT 对比
    ax = axes[2, 0]
    dx = np.mean(np.diff(wn))
    for label, data, color in [
        ("Before", result.original, "salmon"),
        ("After", result.corrected, "steelblue"),
    ]:
        n = len(data)
        freqs = np.fft.rfftfreq(n, d=dx)
        power = np.abs(np.fft.rfft(data - np.mean(data))) ** 2
        ax.semilogy(freqs[1:], power[1:], "-", lw=0.8, color=color,
                     alpha=0.7, label=label)
    ax.set_xlabel("Frequency (cycles/cm⁻¹)")
    ax.set_ylabel("Power")
    ax.set_title("FFT power spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,1) 基线区域 zoom-in
    ax = axes[2, 1]
    corr_bl = result.corrected[mask]
    wn_bl = wn[mask]
    ax.plot(wn_bl, corr_bl, ".", ms=2, color="steelblue", alpha=0.5)
    bl_mean = np.mean(corr_bl)
    bl_std = np.std(corr_bl)
    ax.axhline(bl_mean, color="orange", lw=1, ls="--", label=f"mean={bl_mean:.3f}")
    ax.set_ylim(bl_mean - 5 * bl_std, bl_mean + 5 * bl_std)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Corrected signal")
    ax.set_title(f"Baseline zoom-in (σ={bl_std:.4f})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图表已保存: {save_path}")

    plt.close(fig)
    return fig


# ==================================================================
# 项目级常量
# ==================================================================
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent.parent
RINGDOWN_ROOT = _PROJECT_ROOT / "output" / "results" / "ringdown"
ETALON_ROOT = _PROJECT_ROOT / "output" / "results" / "etalon"
_CSV_NAME = "ringdown_results.csv"


# ==================================================================
# 自动发现 & 批量处理
# ==================================================================
def _discover_ringdown_results(
    ringdown_root: _Path | None = None,
) -> list[tuple[str, str, _Path]]:
    """自动发现 ringdown_root/{跃迁波数}/{压力}/ringdown_results.csv"""
    root = ringdown_root or RINGDOWN_ROOT
    tasks: list[tuple[str, str, _Path]] = []
    for transition_dir in sorted(root.iterdir()):
        if not transition_dir.is_dir() or transition_dir.name.startswith("."):
            continue
        for pressure_dir in sorted(transition_dir.iterdir()):
            if not pressure_dir.is_dir() or pressure_dir.name.startswith("."):
                continue
            csv_path = pressure_dir / _CSV_NAME
            if csv_path.exists():
                tasks.append((transition_dir.name, pressure_dir.name, csv_path))
    return tasks


def _process_single(
    csv_path: _Path,
    output_dir: _Path,
    label: str = "",
    hitran_kwargs: dict | None = None,
) -> bool:
    """处理单个 ringdown_results.csv 的标准具去除"""
    df = pd.read_csv(csv_path)
    wn = df["wavenumber"].values
    tau = df["tau_mean"].values
    temperature = float(df["temperature"].mean())
    pressure = float(df["pressure"].mean())

    print(f"\n  数据: {csv_path}")
    print(f"    点数: {len(wn)}, 波数: {wn.min():.5f} ~ {wn.max():.5f} cm⁻¹")
    print(f"    温度: {temperature:.1f} °C, 压力: {pressure:.1f} Torr")

    output_dir.mkdir(parents=True, exist_ok=True)

    hkw = hitran_kwargs or {
        "molecule": 7,
        "isotopologue": 1,
        "threshold_ratio": 0.01,
        "margin": 0.05,
    }
    remover = EtalonRemover(
        n_etalons=1,
        poly_order=1,
        exclude_regions="hitran",
        hitran_kwargs=hkw,
    )

    try:
        result = remover.fit(wn, tau, temperature=temperature, pressure_torr=pressure)
    except Exception as e:
        print(f"    [ERROR] 拟合失败: {e}")
        return False

    print(f"    {result.summary()}")
    print(f"    拟合成功: {result.model_result.success}, "
          f"迭代: {result.model_result.nfev}")

    title = f"Etalon Removal — {label}" if label else "Etalon Removal"
    plot_etalon_removal(
        result,
        title=title,
        save_path=str(output_dir / "etalon_removal.png"),
    )

    df_out = df.copy()
    df_out["tau_mean_etalon"] = result.etalon
    df_out["tau_mean_no_etalon"] = result.corrected
    df_out.to_csv(output_dir / "tau_etalon_corrected.csv", index=False)

    return True


def batch_etalon_removal(
    ringdown_root: _Path | None = None,
    etalon_root: _Path | None = None,
    hitran_kwargs: dict | None = None,
):
    """批量处理所有自动发现的 ringdown 数据集的标准具去除

    自动扫描 ringdown_root/{跃迁波数}/{压力}/ringdown_results.csv，
    基于 HITRAN 模拟自动检测吸收峰排除区域，去除标准具效应，
    结果输出到 etalon_root/{跃迁波数}/{压力}/。

    Parameters
    ----------
    ringdown_root : Path, optional
        ringdown 结果根目录，默认 output/results/ringdown/
    etalon_root : Path, optional
        etalon 输出根目录，默认 output/results/etalon/
    hitran_kwargs : dict, optional
        传给 hitran_detect_absorption 的参数
    """
    r_root = ringdown_root or RINGDOWN_ROOT
    e_root = etalon_root or ETALON_ROOT

    tasks = _discover_ringdown_results(r_root)
    if not tasks:
        print(f"[ERROR] 未在 {r_root} 下找到 {{跃迁波数}}/{{压力}}/{_CSV_NAME}")
        return

    print(f"{'#' * 60}")
    print(f"  CRDS 标准具效应批量去除 (HITRAN 自动检测)")
    print(f"  输入根目录: {r_root}")
    print(f"  输出根目录: {e_root}")
    print(f"  发现 {len(tasks)} 个数据集:")
    for transition, pressure, _ in tasks:
        print(f"    {transition}/{pressure}/")
    print(f"{'#' * 60}")

    success = 0
    for i, (transition, pressure, csv_path) in enumerate(tasks, 1):
        output_dir = e_root / transition / pressure
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(tasks)}] {transition} / {pressure}")
        print(f"{'=' * 60}")

        ok = _process_single(
            csv_path, output_dir,
            label=f"{transition} / {pressure}",
            hitran_kwargs=hitran_kwargs,
        )
        if ok:
            success += 1

    print(f"\n\n{'#' * 60}")
    print(f"  全部处理完成! {success}/{len(tasks)} 成功")
    print(f"{'#' * 60}")
    for transition, pressure, _ in tasks:
        d = e_root / transition / pressure
        if d.exists():
            print(f"\n  {transition}/{pressure}/")
            for f in sorted(d.glob("*")):
                print(f"    {f.name:<45s} {f.stat().st_size:>10,} bytes")


