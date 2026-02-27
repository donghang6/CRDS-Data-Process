"""标准具效应去除 (Etalon Removal)

CRDS 光谱中常存在由光学元件平行面引起的标准具效应（etalon fringes），
表现为叠加在基线上的周期性正弦振荡。本模块使用 lmfit 进行单频/多频
正弦模型拟合，从光谱中识别并去除标准具干涉条纹。

核心类:
    HitranAbsorptionDetector — 基于 HITRAN 模拟检测吸收峰区域
    EtalonRemover            — 标准具拟合与去除
    EtalonFitResult          — 拟合结果 (含绘图/保存方法)
    EtalonBatchProcessor     — 自动发现 & 批量处理

用法:
    # 单个数据集
    remover = EtalonRemover(exclude_regions="hitran")
    result = remover.fit(wn, tau, temperature=30.0, pressure_torr=100.0)
    result.plot(save_path="etalon.png")

    # 批量处理
    from crds_process.baseline.etalon import batch_etalon_removal
    batch_etalon_removal()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from lmfit.model import ModelResult

from crds_process.log import logger


# ==================================================================
# 项目级常量
# ==================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RINGDOWN_ROOT = PROJECT_ROOT / "output" / "results" / "ringdown"
ETALON_ROOT = PROJECT_ROOT / "output" / "results" / "etalon"
HITRAN_DIR = PROJECT_ROOT / "data" / "hitran"
_CSV_NAME = "ringdown_results.csv"


# ==================================================================
# 内部工具函数
# ==================================================================
def _sine_component(x: np.ndarray, amplitude: float, frequency: float,
                    phase: float) -> np.ndarray:
    """单个正弦分量: a · sin(2π·f·x + φ)"""
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def _build_exclude_mask(x: np.ndarray,
                        regions: list[list[float]] | None) -> np.ndarray:
    """构建排除掩码 (True = 保留用于拟合)"""
    mask = np.ones(len(x), dtype=bool)
    if regions:
        for lo, hi in regions:
            mask &= ~((x >= lo) & (x <= hi))
    return mask


def _estimate_dominant_frequency(x: np.ndarray, y: np.ndarray,
                                 n_freq: int = 1) -> list[float]:
    """Lomb-Scargle 周期图估计主要频率 (适用于非均匀采样)"""
    from scipy.signal import detrend, lombscargle

    y_dt = detrend(y, type="linear")
    dx_med = float(np.median(np.diff(x)))
    f_min = 1.0 / (x[-1] - x[0])
    f_max = 0.5 / dx_med
    test_f = np.linspace(f_min, f_max, 10000)
    power = lombscargle(x, y_dt, 2.0 * np.pi * test_f, normalize=True)

    result: list[float] = []
    for _ in range(n_freq):
        idx = int(np.argmax(power))
        if power[idx] > 0:
            result.append(float(test_f[idx]))
        delta = max(3, int(len(test_f) * 0.02))
        power[max(0, idx - delta): min(len(power), idx + delta + 1)] = 0

    return result or [1.0]


def _detect_significant_frequencies(
    x: np.ndarray, y: np.ndarray,
    max_components: int = 5,
    snr_threshold: float = 3.0,
) -> list[tuple[float, float]]:
    """自动检测显著的标准具频率分量

    使用 Lomb-Scargle 周期图迭代检测：
      1. 找到最大功率的频率
      2. 检查该频率的 SNR (peak / median)
      3. 若 SNR > 阈值，记录该频率并从谱中减去
      4. 重复直到没有显著分量或达到上限

    Returns
    -------
    list[tuple[float, float]]
        [(freq, power), ...] 按功率降序
    """
    from scipy.signal import detrend, lombscargle

    y_work = detrend(y, type="linear").copy()
    dx_med = float(np.median(np.diff(x)))
    f_min = 2.0 / (x[-1] - x[0])  # 至少 2 个周期
    f_max = 0.5 / dx_med
    test_f = np.linspace(f_min, f_max, 10000)

    detected: list[tuple[float, float]] = []
    for _ in range(max_components):
        power = lombscargle(x, y_work, 2.0 * np.pi * test_f, normalize=False)
        idx_peak = int(np.argmax(power))
        peak_power = float(power[idx_peak])

        # SNR: 峰值 / 中位数
        median_power = float(np.median(power))
        snr = peak_power / median_power if median_power > 0 else 0

        if snr < snr_threshold:
            break

        freq = float(test_f[idx_peak])
        detected.append((freq, peak_power))

        # 从信号中减去这个频率分量 (简单正弦拟合)
        omega = 2.0 * np.pi * freq
        sin_vals = np.sin(omega * x)
        cos_vals = np.cos(omega * x)
        a = 2.0 * np.mean(y_work * sin_vals)
        b = 2.0 * np.mean(y_work * cos_vals)
        y_work -= a * sin_vals + b * cos_vals

    return detected


def _suppress_stdout(func, *args, **kwargs):
    """静默执行函数 (抑制 stdout 输出)"""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = old


# ==================================================================
# HITRAN 吸收峰检测器
# ==================================================================
class HitranAbsorptionDetector:
    """基于 HITRAN 模拟自动检测吸收峰区域

    使用 HAPI 计算给定温度/压力下的吸收系数谱，
    将吸收系数 > peak × threshold_ratio 的波数区域标记为吸收峰。

    Parameters
    ----------
    molecule : int
        HITRAN 分子编号 (默认 7 = O₂)
    isotopologue : int
        HITRAN 同位素编号 (默认 1 = ¹⁶O₂)
    hitran_table : str | None
        HAPI 本地表名，None 则自动搜索
    hitran_dir : str | Path | None
        HITRAN 数据目录，None 则使用项目默认路径
    threshold_ratio : float
        吸收系数阈值 = peak × threshold_ratio (0~1)
    margin : float
        检测到的区域向两侧扩展的余量 (cm⁻¹)
    step : float
        模拟波数步长 (cm⁻¹)
    """

    _MOL_PREFIX = {1: "H2O", 2: "CO2", 6: "CH4", 7: "O2"}

    def __init__(
        self,
        molecule: int = 7,
        isotopologue: int = 1,
        hitran_table: str | None = None,
        hitran_dir: str | Path | None = None,
        threshold_ratio: float = 0.01,
        margin: float = 0.05,
        step: float = 0.002,
    ):
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.hitran_table = hitran_table
        self.hitran_dir = str(hitran_dir) if hitran_dir else None
        self.threshold_ratio = threshold_ratio
        self.margin = margin
        self.step = step
        self._hapi = None

    @property
    def mol_prefix(self) -> str:
        return self._MOL_PREFIX.get(self.molecule, f"M{self.molecule}")

    def _ensure_hapi(self):
        """确保 HAPI 已导入并初始化"""
        if self._hapi is not None:
            return

        self._hapi = _suppress_stdout(__import__, "hapi")

        if self.hitran_dir is None:
            candidates = [
                str(HITRAN_DIR),
                os.path.join(os.getcwd(), "data", "hitran"),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    self.hitran_dir = c
                    break
            if self.hitran_dir is None:
                self.hitran_dir = str(HITRAN_DIR)
                os.makedirs(self.hitran_dir, exist_ok=True)

        saved_dir = os.getcwd()
        os.chdir(self.hitran_dir)
        try:
            _suppress_stdout(self._hapi.db_begin, self.hitran_dir)
        finally:
            os.chdir(saved_dir)

    def _resolve_table(self, wn_min: float, wn_max: float) -> str:
        """确定 HAPI 表名，必要时下载"""
        hapi = self._hapi

        if self.hitran_table:
            return self.hitran_table

        for tname in hapi.LOCAL_TABLE_CACHE:
            if tname.startswith(self.mol_prefix):
                self.hitran_table = tname
                return tname

        tname = f"{self.mol_prefix}_iso{self.isotopologue}"
        saved_dir = os.getcwd()
        os.chdir(self.hitran_dir)
        try:
            logger.info(f"  [HITRAN] 下载 {self.mol_prefix} "
                  f"({wn_min:.0f}-{wn_max:.0f} cm⁻¹)...")
            hapi.fetch(TableName=tname, M=self.molecule,
                       I=self.isotopologue, numin=wn_min, numax=wn_max)
        finally:
            os.chdir(saved_dir)
        self.hitran_table = tname
        return tname

    def detect(
        self,
        wavenumber: np.ndarray,
        temperature: float,
        pressure_torr: float,
    ) -> list[list[float]]:
        """检测吸收峰区域

        Parameters
        ----------
        wavenumber : np.ndarray
            实测波数 (cm⁻¹)
        temperature : float
            温度 (°C)
        pressure_torr : float
            压力 (Torr)

        Returns
        -------
        list[list[float]]
            排除区域 [[lo, hi], ...]
        """
        self._ensure_hapi()
        hapi = self._hapi

        wn_min = float(np.min(wavenumber)) - 0.5
        wn_max = float(np.max(wavenumber)) + 0.5
        table = self._resolve_table(wn_min - 1.5, wn_max + 1.5)

        T_k = temperature + 273.15
        p_atm = pressure_torr / 760.0

        saved_dir = os.getcwd()
        os.chdir(self.hitran_dir)
        try:
            nu_sim, alpha_sim = _suppress_stdout(
                hapi.absorptionCoefficient_Voigt,
                SourceTables=table,
                Environment={"T": T_k, "p": p_atm},
                WavenumberRange=[wn_min, wn_max],
                WavenumberStep=self.step,
                HITRAN_units=False,
            )
        finally:
            os.chdir(saved_dir)

        return self._find_regions(nu_sim, alpha_sim.flatten())

    def _find_regions(
        self, nu: np.ndarray, alpha: np.ndarray,
    ) -> list[list[float]]:
        """从吸收系数谱中提取排除区域"""
        peak = float(np.max(alpha))
        if peak <= 0:
            return []

        above = nu[alpha > peak * self.threshold_ratio]
        if len(above) == 0:
            return []

        gap = self.step * 3
        regions: list[list[float]] = []
        start = end = float(above[0])
        for v in above[1:]:
            if v - end > gap:
                regions.append([start - self.margin, end + self.margin])
                start = float(v)
            end = float(v)
        regions.append([start - self.margin, end + self.margin])
        return regions


# ==================================================================
# 拟合结果
# ==================================================================
@dataclass
class EtalonFitResult:
    """标准具拟合结果"""

    wavenumber: np.ndarray
    original: np.ndarray
    etalon: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    model_result: ModelResult
    n_etalons: int
    fit_mask: np.ndarray
    exclude_regions: list = field(default_factory=list)
    components: list[dict] = field(default_factory=list)

    @property
    def residual_std(self) -> float:
        return float(np.std(self.model_result.residual))

    @property
    def n_fit_points(self) -> int:
        return int(np.sum(self.fit_mask))

    def summary(self) -> str:
        lines = [
            f"Etalon fit: {self.n_etalons} component(s)",
            f"  拟合点数    = {self.n_fit_points} / {len(self.wavenumber)}",
        ]
        for i, reg in enumerate(self.exclude_regions):
            lines.append(
                f"  排除区域 {i}: [{reg[0]:.5f}, {reg[1]:.5f}] cm⁻¹"
            )
        lines += [
            f"  Reduced χ²  = {self.model_result.redchi:.6e}",
            f"  Residual σ  = {self.residual_std:.6e}",
        ]
        for i, c in enumerate(self.components):
            lines.append(
                f"  Component {i}: "
                f"amp={c['amplitude']:.4e}, "
                f"freq={c['frequency']:.4f} cycles/cm⁻¹, "
                f"period={1 / c['frequency']:.5f} cm⁻¹, "
                f"phase={c['phase']:.4f} rad"
            )
        return "\n".join(lines)

    def plot(
        self,
        title: str = "Etalon Removal",
        save_path: str | Path | None = None,
        figsize: tuple[float, float] = (14, 10),
    ):
        """绘制 6 面板对比图"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wn, mask = self.wavenumber, self.fit_mask
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # (0,0) 原始 + 全拟合
        ax = axes[0, 0]
        ax.plot(wn, self.original, ".", ms=2, alpha=0.4, label="Original")
        ax.plot(wn, self.baseline + self.etalon, "r-", lw=1.2,
                label="Baseline + Etalon")
        ax.plot(wn, self.baseline, "--", color="orange", lw=1, alpha=0.7,
                label="Baseline (poly)")
        if not np.all(mask):
            exc = wn[~mask]
            if len(exc):
                ax.axvspan(exc.min(), exc.max(), alpha=0.1, color="red",
                           label="Excluded")
        ax.set_ylabel("Signal")
        ax.set_title("Original + Fit")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

        # (0,1) 标准具分量
        ax = axes[0, 1]
        ax.plot(wn, self.etalon, "g-", lw=1)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel("Etalon component")
        ax.set_title(f"Etalon ({self.n_etalons} sine)")
        ax.grid(True, alpha=0.3)

        # (1,0) 去除标准具后
        ax = axes[1, 0]
        ax.plot(wn, self.corrected, ".", ms=2, color="steelblue", alpha=0.5)
        ax.set_ylabel("Corrected signal")
        ax.set_title("After etalon removal")
        ax.grid(True, alpha=0.3)

        # (1,1) 残差
        ax = axes[1, 1]
        res = self.original[mask] - (self.baseline[mask] + self.etalon[mask])
        ax.plot(wn[mask], res, ".", ms=2, color="tomato", alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel("Residual")
        ax.set_title(f"Fit residual (σ={np.std(res):.4f})")
        ax.grid(True, alpha=0.3)

        # (2,0) FFT 对比
        ax = axes[2, 0]
        dx = float(np.mean(np.diff(wn)))
        for lbl, data, clr in [("Before", self.original, "salmon"),
                                ("After", self.corrected, "steelblue")]:
            freqs = np.fft.rfftfreq(len(data), d=dx)
            pwr = np.abs(np.fft.rfft(data - np.mean(data))) ** 2
            ax.semilogy(freqs[1:], pwr[1:], "-", lw=0.8, color=clr,
                        alpha=0.7, label=lbl)
        ax.set_xlabel("Frequency (cycles/cm⁻¹)")
        ax.set_ylabel("Power")
        ax.set_title("FFT power spectrum")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (2,1) 基线 zoom-in
        ax = axes[2, 1]
        c_bl, w_bl = self.corrected[mask], wn[mask]
        ax.plot(w_bl, c_bl, ".", ms=2, color="steelblue", alpha=0.5)
        mu, sig = float(np.mean(c_bl)), float(np.std(c_bl))
        ax.axhline(mu, color="orange", lw=1, ls="--",
                    label=f"mean={mu:.3f}")
        ax.set_ylim(mu - 5 * sig, mu + 5 * sig)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Corrected signal")
        ax.set_title(f"Baseline zoom-in (σ={sig:.4f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info(f"  图表已保存: {save_path}")
        plt.close(fig)
        return fig

    def save_csv(
        self,
        df: pd.DataFrame,
        save_path: str | Path,
        signal_col: str = "tau_mean",
    ) -> pd.DataFrame:
        """将 etalon/corrected 列追加到 DataFrame 并保存"""
        out = df.copy()
        out[f"{signal_col}_etalon"] = self.etalon
        out[f"{signal_col}_no_etalon"] = self.corrected
        out.to_csv(str(save_path), index=False)
        return out


# ==================================================================
# 标准具去除器
# ==================================================================
class EtalonRemover:
    """标准具效应拟合与去除 (迭代交替优化)

    支持两种模式:
    1. 固定分量数: n_etalons=N (手动指定)
    2. 自动检测:   n_etalons="auto" (根据 SNR 自动确定分量数)

    Parameters
    ----------
    n_etalons : int | str
        正弦分量数 (1~5) 或 "auto" (自动检测)
    freq_hints : list[float] | None
        频率初始猜测 (cycles/cm⁻¹)
    exclude_regions : list | str | None
        - list: 手动指定 [[lo, hi], ...]
        - "hitran": HITRAN 模拟自动检测 (默认)
        - None: 不排除
    poly_order : int
        去趋势多项式阶数 (推荐 1)
    n_iter : int
        交替迭代次数 (推荐 3~5)
    flatten_baseline : bool
        是否扣除基线趋势使输出平坦
    hitran_detector : HitranAbsorptionDetector | None
        自定义检测器，None 则使用默认参数
    max_nfev : int
        lmfit 每次拟合最大函数求值次数
    auto_max_components : int
        自动模式最多检测的分量数 (默认 5)
    auto_snr_threshold : float
        自动模式 SNR 阈值 (默认 3.0)
    residual_improvement_threshold : float
        残差改善比例阈值，低于此值停止添加分量 (默认 0.05 = 5%)
    """

    def __init__(
        self,
        n_etalons: int | str = "auto",
        freq_hints: list[float] | None = None,
        exclude_regions: list[list[float]] | str | None = "hitran",
        poly_order: int = 1,
        n_iter: int = 5,
        flatten_baseline: bool = True,
        hitran_detector: HitranAbsorptionDetector | None = None,
        max_nfev: int = 20000,
        auto_max_components: int = 5,
        auto_snr_threshold: float = 3.0,
        residual_improvement_threshold: float = 0.05,
    ):
        if isinstance(n_etalons, int) and n_etalons < 1:
            raise ValueError("n_etalons 必须 >= 1 或 'auto'")
        self.n_etalons = n_etalons
        self.freq_hints = freq_hints
        self.exclude_regions = exclude_regions
        self.poly_order = poly_order
        self.n_iter = n_iter
        self.flatten_baseline = flatten_baseline
        self.hitran_detector = hitran_detector or HitranAbsorptionDetector()
        self.max_nfev = max_nfev
        self.auto_max_components = auto_max_components
        self.auto_snr_threshold = auto_snr_threshold
        self.residual_improvement_threshold = residual_improvement_threshold

    def _build_model(
        self, x: np.ndarray, y: np.ndarray,
    ) -> tuple[Model, Parameters]:
        """构建 lmfit 正弦模型"""
        freq_init = (
            self.freq_hints[:self.n_etalons]
            if self.freq_hints and len(self.freq_hints) >= self.n_etalons
            else _estimate_dominant_frequency(x, y, self.n_etalons)
        )

        model = Model(_sine_component, prefix="e0_")
        for i in range(1, self.n_etalons):
            model += Model(_sine_component, prefix=f"e{i}_")

        params = model.make_params()
        amp_est = max(float(np.std(y) * np.sqrt(2)), 1e-10)

        for i in range(self.n_etalons):
            pfx = f"e{i}_"
            f0 = freq_init[i] if i < len(freq_init) else freq_init[-1] * (i + 1)
            params[f"{pfx}amplitude"].set(value=amp_est / (i + 1), min=0)
            params[f"{pfx}frequency"].set(value=f0, min=f0 * 0.5, max=f0 * 2.0)
            params[f"{pfx}phase"].set(value=0, min=-np.pi, max=np.pi)

        return model, params

    def _resolve_exclude(
        self, x: np.ndarray,
        temperature: float | None,
        pressure_torr: float | None,
        extra: list[list[float]] | None,
    ) -> list[list[float]]:
        """解析排除区域"""
        if self.exclude_regions == "hitran":
            if temperature is None or pressure_torr is None:
                raise ValueError(
                    "exclude_regions='hitran' 需要提供 "
                    "temperature (°C) 和 pressure_torr (Torr)"
                )
            regions = self.hitran_detector.detect(x, temperature, pressure_torr)
        elif isinstance(self.exclude_regions, list):
            regions = list(self.exclude_regions)
        else:
            regions = []
        if extra:
            regions.extend(extra)
        return regions

    def fit(
        self,
        wavenumber: np.ndarray,
        signal: np.ndarray,
        temperature: float | None = None,
        pressure_torr: float | None = None,
        exclude_regions: list[list[float]] | None = None,
    ) -> EtalonFitResult:
        """迭代交替优化拟合

        auto 模式流程:
          1. 去趋势 + 排除吸收区
          2. 对残差做 Lomb-Scargle 找最强频率
          3. 拟合 1 个正弦 → 减去 → 检查残差改善
          4. 对新残差再找频率 → 添加分量 → 联合拟合
          5. 残差不再改善时停止
        """
        x = np.asarray(wavenumber, dtype=float)
        y = np.asarray(signal, dtype=float)
        if len(x) != len(y):
            raise ValueError("wavenumber 和 signal 长度不一致")

        all_exclude = self._resolve_exclude(
            x, temperature, pressure_torr, exclude_regions
        )
        fit_mask = _build_exclude_mask(x, all_exclude)
        x_fit, y_fit = x[fit_mask], y[fit_mask]
        if len(x_fit) < 20:
            raise ValueError(f"排除后仅剩 {len(x_fit)} 个点，不足以拟合")

        x_center = float(np.mean(x))
        x_c = x - x_center
        x_fit_c = x_fit - x_center

        # 确定分量数
        if self.n_etalons == "auto":
            return self._fit_auto(
                x, y, x_fit, y_fit, x_c, x_fit_c, x_center,
                fit_mask, all_exclude,
            )
        else:
            return self._fit_fixed(
                x, y, x_fit, y_fit, x_c, x_fit_c, x_center,
                fit_mask, all_exclude, self.n_etalons,
            )

    def _fit_fixed(
        self, x, y, x_fit, y_fit, x_c, x_fit_c, x_center,
        fit_mask, all_exclude, n_etalons,
        freq_hints=None,
    ) -> EtalonFitResult:
        """固定分量数的迭代交替拟合"""
        etalon_fit = np.zeros_like(x_fit)
        last_result = None
        prev_params: Parameters | None = None

        # 使用提供的频率初始值或自动估计
        saved_n = self.n_etalons
        saved_hints = self.freq_hints
        self.n_etalons = n_etalons
        if freq_hints:
            self.freq_hints = freq_hints

        try:
            for _ in range(self.n_iter):
                poly_c = np.polyfit(x_fit_c, y_fit - etalon_fit,
                                    self.poly_order)
                baseline_fit = np.polyval(poly_c, x_fit_c)
                residual = y_fit - baseline_fit

                model, params = self._build_model(x_fit, residual)
                if prev_params is not None:
                    for pn in prev_params:
                        if pn in params:
                            params[pn].set(value=prev_params[pn].value)

                last_result = model.fit(residual, params, x=x_fit,
                                        max_nfev=self.max_nfev)
                prev_params = last_result.params

                etalon_fit = np.zeros_like(x_fit)
                for i in range(n_etalons):
                    pfx = f"e{i}_"
                    etalon_fit += _sine_component(
                        x_fit,
                        last_result.params[f"{pfx}amplitude"].value,
                        last_result.params[f"{pfx}frequency"].value,
                        last_result.params[f"{pfx}phase"].value,
                    )
        finally:
            self.n_etalons = saved_n
            self.freq_hints = saved_hints

        # 全波段结果
        poly_final = np.polyfit(x_fit_c, y_fit - etalon_fit, self.poly_order)
        baseline_full = np.polyval(poly_final, x_c)

        etalon_full = np.zeros_like(x)
        components: list[dict] = []
        for i in range(n_etalons):
            pfx = f"e{i}_"
            amp = last_result.params[f"{pfx}amplitude"].value
            freq = last_result.params[f"{pfx}frequency"].value
            phase = last_result.params[f"{pfx}phase"].value
            etalon_full += _sine_component(x, amp, freq, phase)
            components.append({"amplitude": amp, "frequency": freq,
                               "phase": phase})

        corrected = y - etalon_full
        if self.flatten_baseline:
            corrected -= baseline_full - float(np.mean(baseline_full))

        return EtalonFitResult(
            wavenumber=x, original=y, etalon=etalon_full,
            baseline=baseline_full, corrected=corrected,
            model_result=last_result, n_etalons=n_etalons,
            fit_mask=fit_mask, exclude_regions=all_exclude,
            components=components,
        )

    def _fit_auto(
        self, x, y, x_fit, y_fit, x_c, x_fit_c, x_center,
        fit_mask, all_exclude,
    ) -> EtalonFitResult:
        """自动检测分量数的贪心迭代拟合

        每轮:
          1. 对当前残差用 Lomb-Scargle 找显著频率
          2. 用已有频率 + 新频率联合拟合
          3. 检查残差是否改善 > threshold
          4. 不改善则回退，返回上轮结果
        """
        # 初始: 去趋势得到残差
        poly_init = np.polyfit(x_fit_c, y_fit, self.poly_order)
        residual = y_fit - np.polyval(poly_init, x_fit_c)
        prev_std = float(np.std(residual))

        best_result = None
        accumulated_freqs: list[float] = []

        for comp_idx in range(self.auto_max_components):
            # 检测当前残差中最显著的频率
            detected = _detect_significant_frequencies(
                x_fit, residual,
                max_components=1,
                snr_threshold=self.auto_snr_threshold,
            )
            if not detected:
                logger.info(f"    auto: 第 {comp_idx + 1} 个分量 — "
                             f"无显著频率, 停止")
                break

            new_freq = detected[0][0]
            trial_freqs = accumulated_freqs + [new_freq]

            logger.info(f"    auto: 尝试第 {comp_idx + 1} 个分量, "
                         f"freq={new_freq:.2f} cycles/cm⁻¹")

            # 联合拟合所有分量
            trial_result = self._fit_fixed(
                x, y, x_fit, y_fit, x_c, x_fit_c, x_center,
                fit_mask, all_exclude, len(trial_freqs),
                freq_hints=trial_freqs,
            )

            # 计算改善
            new_std = trial_result.residual_std
            improvement = (prev_std - new_std) / prev_std if prev_std > 0 else 0

            logger.info(f"    auto: σ {prev_std:.6f} → {new_std:.6f} "
                         f"(改善 {improvement * 100:.1f}%)")

            if improvement < self.residual_improvement_threshold and comp_idx > 0:
                logger.info(f"    auto: 改善 < {self.residual_improvement_threshold * 100:.0f}%, "
                             f"停止, 使用 {len(accumulated_freqs)} 个分量")
                break

            # 接受这个分量
            accumulated_freqs = trial_freqs
            best_result = trial_result
            prev_std = new_std

            # 更新残差: 从 corrected 信号中去趋势
            corrected_fit = trial_result.corrected[fit_mask]
            poly_new = np.polyfit(x_fit_c, corrected_fit, self.poly_order)
            residual = corrected_fit - np.polyval(poly_new, x_fit_c)

        if best_result is None:
            # 没有检测到任何分量，返回仅基线的结果
            logger.info(f"    auto: 未检测到标准具, 仅去除基线")
            return self._fit_fixed(
                x, y, x_fit, y_fit, x_c, x_fit_c, x_center,
                fit_mask, all_exclude, 1,
            )

        logger.info(f"    auto: 最终使用 {best_result.n_etalons} 个标准具分量")
        return best_result

    def fit_df(
        self,
        df: pd.DataFrame,
        wavenumber_col: str = "wavenumber",
        signal_col: str = "tau_mean",
        temperature_col: str = "temperature",
        pressure_col: str = "pressure",
    ) -> tuple[pd.DataFrame, EtalonFitResult]:
        """对 DataFrame 去除标准具效应 (自动提取温度/压力)"""
        temperature = (
            float(df[temperature_col].mean())
            if self.exclude_regions == "hitran" and temperature_col in df.columns
            else None
        )
        pressure = (
            float(df[pressure_col].mean())
            if self.exclude_regions == "hitran" and pressure_col in df.columns
            else None
        )
        result = self.fit(
            df[wavenumber_col].values, df[signal_col].values,
            temperature=temperature, pressure_torr=pressure,
        )
        out = df.copy()
        out[f"{signal_col}_etalon"] = result.etalon
        out[f"{signal_col}_no_etalon"] = result.corrected
        return out, result


# ==================================================================
# 批量处理器
# ==================================================================
class EtalonBatchProcessor:
    """标准具效应批量处理器

    自动扫描 ringdown_root/{跃迁波数}/{压力}/ringdown_results.csv，
    逐数据集去除标准具效应，输出到 etalon_root/{跃迁波数}/{压力}/。

    Parameters
    ----------
    ringdown_root : Path | None
        输入根目录
    etalon_root : Path | None
        输出根目录
    remover : EtalonRemover | None
        自定义 EtalonRemover
    """

    def __init__(
        self,
        ringdown_root: Path | None = None,
        etalon_root: Path | None = None,
        remover: EtalonRemover | None = None,
    ):
        self.ringdown_root = ringdown_root or RINGDOWN_ROOT
        self.etalon_root = etalon_root or ETALON_ROOT
        self.remover = remover or EtalonRemover()

    def discover(self) -> list[tuple[str, str, Path]]:
        """自动发现数据集"""
        tasks: list[tuple[str, str, Path]] = []
        if not self.ringdown_root.exists():
            return tasks
        for t_dir in sorted(self.ringdown_root.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            for p_dir in sorted(t_dir.iterdir()):
                if not p_dir.is_dir() or p_dir.name.startswith("."):
                    continue
                csv = p_dir / _CSV_NAME
                if csv.exists():
                    tasks.append((t_dir.name, p_dir.name, csv))
        return tasks

    def process_one(
        self, csv_path: Path, output_dir: Path, label: str = "",
    ) -> bool:
        """处理单个数据集"""
        df = pd.read_csv(csv_path)
        wn = df["wavenumber"].values
        tau = df["tau_mean"].values
        temp = float(df["temperature"].mean())
        pres = float(df["pressure"].mean())

        logger.info(f"\n  数据: {csv_path}")
        logger.info(f"    点数: {len(wn)}, 波数: {wn.min():.5f} ~ "
              f"{wn.max():.5f} cm⁻¹")
        logger.info(f"    温度: {temp:.1f} °C, 压力: {pres:.1f} Torr")

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self.remover.fit(
                wn, tau, temperature=temp, pressure_torr=pres
            )
        except Exception as e:
            logger.error(f"    拟合失败: {e}")
            return False

        logger.info(f"    {result.summary()}")
        logger.info(f"    拟合成功: {result.model_result.success}, "
              f"迭代: {result.model_result.nfev}")

        title = f"Etalon Removal — {label}" if label else "Etalon Removal"
        result.plot(title=title, save_path=output_dir / "etalon_removal.png")
        result.save_csv(df, output_dir / "tau_etalon_corrected.csv")
        return True

    def run(self):
        """执行批量处理"""
        tasks = self.discover()
        if not tasks:
            logger.error(f"未在 {self.ringdown_root} 下找到 "
                  f"{{跃迁波数}}/{{压力}}/{_CSV_NAME}")
            return

        logger.info(f"{'#' * 60}")
        logger.info(f"  CRDS 标准具效应批量去除 (HITRAN 自动检测)")
        logger.info(f"  输入: {self.ringdown_root}")
        logger.info(f"  输出: {self.etalon_root}")
        logger.info(f"  发现 {len(tasks)} 个数据集:")
        for t, p, _ in tasks:
            logger.info(f"    {t}/{p}/")
        logger.info(f"{'#' * 60}")

        ok = 0
        for i, (transition, pressure, csv_path) in enumerate(tasks, 1):
            out_dir = self.etalon_root / transition / pressure
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  [{i}/{len(tasks)}] {transition} / {pressure}")
            logger.info(f"{'=' * 60}")
            if self.process_one(csv_path, out_dir,
                                label=f"{transition} / {pressure}"):
                ok += 1

        logger.info(f"\n\n{'#' * 60}")
        logger.info(f"  全部完成! {ok}/{len(tasks)} 成功")
        logger.info(f"{'#' * 60}")
        for t, p, _ in tasks:
            d = self.etalon_root / t / p
            if d.exists():
                logger.info(f"\n  {t}/{p}/")
                for f in sorted(d.glob("*")):
                    logger.info(f"    {f.name:<45s} "
                          f"{f.stat().st_size:>10,} bytes")


# ==================================================================
# 便捷函数 (供脚本一行调用)
# ==================================================================
def batch_etalon_removal(
    ringdown_root: Path | None = None,
    etalon_root: Path | None = None,
    **remover_kwargs,
):
    """批量去除标准具效应 (便捷入口)"""
    remover = EtalonRemover(**remover_kwargs) if remover_kwargs else None
    EtalonBatchProcessor(
        ringdown_root=ringdown_root,
        etalon_root=etalon_root,
        remover=remover,
    ).run()


# 向后兼容
def plot_etalon_removal(result: EtalonFitResult, **kwargs):
    """向后兼容: 等价于 result.plot(**kwargs)"""
    return result.plot(**kwargs)


def hitran_detect_absorption(wavenumber, temperature, pressure_torr, **kwargs):
    """向后兼容: 等价于 HitranAbsorptionDetector(**kwargs).detect(...)"""
    return HitranAbsorptionDetector(**kwargs).detect(
        wavenumber, temperature, pressure_torr
    )

