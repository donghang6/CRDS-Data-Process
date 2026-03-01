"""线性回归提取 N₂ 展宽参数

物理基础:
    对于 O₂+N₂ 混合气体，总的 Lorentzian 展宽满足:
        γ₀_total = γ₀_O₂ · x_O₂ + γ₀_N₂ · x_N₂

    其中 x_O₂, x_N₂ 是摩尔分数。

    从纯 O₂ 多光谱联合拟合获得可靠的 γ₀_O₂ (已固定)，
    O₂+N₂ 单光谱拟合获得各混合比下的 γ₀_air (即 γ₀_total)。

    定义残差:
        y = γ₀_air - γ₀_O₂ · x_O₂

    则:
        y = γ₀_N₂ · x_N₂   (过原点线性回归)

    加权最小二乘 (WLS) 解:
        γ₀_N₂ = Σ(w·x·y) / Σ(w·x²),  w = 1/σ²

    同理适用于 SD_gamma, delta0, SD_delta 等参数。

用法:
    from crds_process.spectral.linear_regression import N2BroadeningExtractor
    extractor = N2BroadeningExtractor(transition="9386.2076")
    results = extractor.run(mix_stats_csv, o2_multi_csv)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from crds_process.gas_config import parse_gas_dir
from crds_process.log import logger


# ==================================================================
# 数据结构
# ==================================================================
@dataclass
class LinearRegressionResult:
    """单个参数的线性回归结果

    Attributes
    ----------
    param_name : str
        参数名 (如 "gamma0", "SD_gamma", "delta0", "SD_delta")
    value_N2 : float
        N₂ 展宽/位移值 (回归斜率)
    uncertainty_N2 : float
        N₂ 值不确定度 (斜率标准误差)
    value_O2_fixed : float
        固定的 O₂ 值 (来自纯 O₂ 多光谱联合拟合)
    value_O2_fixed_err : float
        O₂ 值不确定度
    R_squared : float
        R² 决定系数
    n_points : int
        数据点数
    x_N2 : np.ndarray
        N₂ 摩尔分数
    y_obs : np.ndarray
        观测的 γ₀_air
    y_err : np.ndarray
        γ₀_air 不确定度
    y_pred : np.ndarray
        模型预测值
    residuals : np.ndarray
        残差 (y_obs - y_pred)
    value_N2_free : float
        自由二参数回归的 N₂ 值 (交叉验证)
    value_O2_free : float
        自由二参数回归的 O₂ 值
    """
    param_name: str = ""
    value_N2: float = 0.0
    uncertainty_N2: float = 0.0
    value_O2_fixed: float = 0.0
    value_O2_fixed_err: float = 0.0
    R_squared: float = 0.0
    n_points: int = 0
    x_N2: np.ndarray = field(default_factory=lambda: np.array([]))
    x_O2: np.ndarray = field(default_factory=lambda: np.array([]))
    y_obs: np.ndarray = field(default_factory=lambda: np.array([]))
    y_err: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    # 自由拟合 (两参数) 的交叉验证
    value_N2_free: float = 0.0
    value_O2_free: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  {self.param_name}:",
            f"    {self.param_name}_N2  = {self.value_N2:.6f}"
            f" ± {self.uncertainty_N2:.6f} cm⁻¹/atm",
            f"    {self.param_name}_O2  = {self.value_O2_fixed:.6f}"
            f" (固定, ±{self.value_O2_fixed_err:.6f})",
            f"    R²        = {self.R_squared:.6f}",
            f"    N_points  = {self.n_points}",
        ]
        if self.value_N2_free != 0:
            lines.append(
                f"    [交叉验证] 自由拟合: "
                f"{self.param_name}_O2={self.value_O2_free:.6f}, "
                f"{self.param_name}_N2={self.value_N2_free:.6f}"
            )
        return "\n".join(lines)


# ==================================================================
# 回归参数配置
# ==================================================================
# 每个需要回归的参数: (参数基名, air列名, O2列名, err后缀)
_REGRESSION_PARAMS = [
    ("gamma0", "gamma0_air", "gamma0_O2", "gamma0_air_err", "gamma0_O2_err"),
    ("SD_gamma", "SD_gamma_air", "SD_gamma_O2", "SD_gamma_air_err", "SD_gamma_O2_err"),
    ("delta0", "delta0_air", "delta0_O2", "delta0_air_err", "delta0_O2_err"),
    ("SD_delta", "SD_delta_air", "SD_delta_O2", "SD_delta_air_err", "SD_delta_O2_err"),
]


# ==================================================================
# 核心回归逻辑
# ==================================================================
def _weighted_zero_intercept_regression(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """过原点加权线性回归 y = slope · x

    Parameters
    ----------
    x : 自变量 (x_N₂)
    y : 因变量 (γ₀_air - γ₀_O₂ · x_O₂)
    w : 权重 (1/σ²)，None 时等权

    Returns
    -------
    slope : 斜率 (γ₀_N₂)
    slope_err : 斜率标准误差
    R_squared : R²
    """
    n = len(x)
    if w is None:
        w = np.ones(n)

    # WLS: slope = Σ(w·x·y) / Σ(w·x²)
    wxxy = np.sum(w * x * y)
    wxx = np.sum(w * x * x)
    if wxx == 0:
        return 0.0, 0.0, 0.0

    slope = wxxy / wxx

    # 残差
    y_pred = slope * x
    residuals = y - y_pred

    # 斜率误差: σ_slope = sqrt(Σ(w·r²) / ((n-1)·Σ(w·x²)))
    # 对于只有 1 个自由参数的过原点回归, 自由度 = n - 1
    wresid_sq = np.sum(w * residuals**2)
    dof = max(n - 1, 1)
    slope_err = np.sqrt(wresid_sq / (dof * wxx)) if wxx > 0 else 0.0

    # R²: 相对于 y 的均值
    ss_res = np.sum(w * residuals**2)
    y_mean = np.average(y, weights=w) if np.sum(w) > 0 else np.mean(y)
    ss_tot = np.sum(w * (y - y_mean)**2)
    R_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return slope, slope_err, R_sq


def _two_param_regression(
    x_O2: np.ndarray,
    x_N2: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[float, float]:
    """两参数无截距回归 y = a · x_O₂ + b · x_N₂

    返回 (a=γ₀_O₂, b=γ₀_N₂) 的自由拟合值 (交叉验证用)
    """
    n = len(y)
    if w is None:
        w = np.ones(n)

    # 加权最小二乘: [x_O2, x_N2]^T W [x_O2, x_N2] β = [x_O2, x_N2]^T W y
    W = np.diag(w)
    X = np.column_stack([x_O2, x_N2])
    XtW = X.T @ W
    beta = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
    return float(beta[0]), float(beta[1])


# ==================================================================
# 主提取器
# ==================================================================
class N2BroadeningExtractor:
    """通过线性回归从 O₂+N₂ 混合气单光谱拟合结果中提取 N₂ 展宽

    Parameters
    ----------
    transition : str
        跃迁标识 (如 "9386.2076")
    outlier_sigma : float
        离群点筛选阈值 (MAD 倍数)，0 表示不筛选
    """

    def __init__(self, transition: str = "", outlier_sigma: float = 3.0):
        self.transition = transition
        self.outlier_sigma = outlier_sigma

    def run(
        self,
        mix_stats_csv: Path | str,
        o2_multi_csv: Path | str,
        output_dir: Path | str | None = None,
    ) -> dict[str, LinearRegressionResult]:
        """执行线性回归提取所有 N₂ 参数

        Parameters
        ----------
        mix_stats_csv : Path
            O₂+N₂ 单光谱拟合统计表
            (output/results/final/O2_N2/{transition}/fit_summary_statistics.csv)
        o2_multi_csv : Path
            纯 O₂ 多光谱联合拟合结果
            (output/results/final/O2/{transition}/multi_fit_result.csv)
        output_dir : Path, optional
            输出目录 (保存 CSV 和图表)

        Returns
        -------
        dict[str, LinearRegressionResult]
            各参数的回归结果
        """
        mix_df = pd.read_csv(str(mix_stats_csv))
        o2_df = pd.read_csv(str(o2_multi_csv))

        if o2_df.empty:
            logger.error("  纯 O₂ 多光谱拟合结果为空")
            return {}

        o2_row = o2_df.iloc[0]

        # 解析每个混合气的 O₂/N₂ 摩尔分数
        x_O2_arr = []
        x_N2_arr = []
        for pres_label in mix_df["pressure"]:
            gc = parse_gas_dir(pres_label, "O2_N2")
            x_O2_arr.append(gc.o2_fraction)
            x_N2_arr.append(gc.n2_fraction)
        x_O2 = np.array(x_O2_arr)
        x_N2 = np.array(x_N2_arr)

        results: dict[str, LinearRegressionResult] = {}

        for (param_base, air_col, o2_col,
             air_err_col, o2_err_col) in _REGRESSION_PARAMS:

            # 检查列是否存在
            if air_col not in mix_df.columns:
                logger.warning(f"  {air_col} 列不存在于混合气统计表中，跳过")
                continue
            if o2_col not in o2_row:
                logger.warning(f"  {o2_col} 列不存在于纯 O₂ 结果中，跳过")
                continue

            gamma_air = mix_df[air_col].values.astype(float)
            gamma_air_err = (mix_df[air_err_col].values.astype(float)
                             if air_err_col in mix_df.columns
                             else np.ones_like(gamma_air))
            gamma_O2_fixed = float(o2_row[o2_col])
            gamma_O2_fixed_err = float(o2_row.get(o2_err_col, 0))

            result = self._regress_one_param(
                param_base, x_O2, x_N2,
                gamma_air, gamma_air_err,
                gamma_O2_fixed, gamma_O2_fixed_err,
            )
            results[param_base] = result

        # 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.save_csv(results, output_dir / "linear_regression_n2.csv")
            self.plot_results(results, output_dir)

        return results

    def _regress_one_param(
        self,
        param_name: str,
        x_O2: np.ndarray,
        x_N2: np.ndarray,
        y_obs: np.ndarray,
        y_err: np.ndarray,
        gamma_O2_fixed: float,
        gamma_O2_fixed_err: float,
    ) -> LinearRegressionResult:
        """对单个参数执行回归"""
        n = len(y_obs)

        # 定义 y = γ_air - γ_O₂ · x_O₂ (消去已知 O₂ 贡献)
        y = y_obs - gamma_O2_fixed * x_O2

        # 权重 (1/σ²), 零误差时等权
        safe_err = np.where(y_err > 0, y_err, 1.0)
        w = np.where(y_err > 0, 1.0 / safe_err**2, np.ones(n))

        # 离群点筛选 (如果数据点 >= 4)
        mask = np.ones(n, dtype=bool)
        if self.outlier_sigma > 0 and n >= 4:
            # 先做一次无筛选回归
            slope0, _, _ = _weighted_zero_intercept_regression(x_N2, y, w)
            pred0 = slope0 * x_N2
            resid0 = y - pred0
            mad = np.median(np.abs(resid0 - np.median(resid0)))
            if mad < 1e-10:
                mad = np.std(resid0)
            if mad > 0:
                deviation = np.abs(resid0 - np.median(resid0)) / mad
                mask = deviation <= self.outlier_sigma
                n_removed = np.sum(~mask)
                if n_removed > 0:
                    logger.info(f"  [{param_name}] 剔除 {n_removed} 个离群点"
                                f" (MAD σ > {self.outlier_sigma})")

        # 最终回归 (使用筛选后的数据)
        x_fit = x_N2[mask]
        y_fit = y[mask]
        w_fit = w[mask]

        slope, slope_err, R_sq = _weighted_zero_intercept_regression(
            x_fit, y_fit, w_fit,
        )

        # 模型预测 (全部数据点)
        y_pred = gamma_O2_fixed * x_O2 + slope * x_N2

        # 交叉验证: 两参数自由回归
        o2_free, n2_free = 0.0, 0.0
        if n >= 3:
            try:
                o2_free, n2_free = _two_param_regression(
                    x_O2[mask], x_N2[mask], y_obs[mask], w_fit,
                )
            except Exception:
                pass

        return LinearRegressionResult(
            param_name=param_name,
            value_N2=slope,
            uncertainty_N2=slope_err,
            value_O2_fixed=gamma_O2_fixed,
            value_O2_fixed_err=gamma_O2_fixed_err,
            R_squared=R_sq,
            n_points=int(np.sum(mask)),
            x_N2=x_N2,
            x_O2=x_O2,
            y_obs=y_obs,
            y_err=y_err,
            y_pred=y_pred,
            residuals=y_obs - y_pred,
            value_N2_free=n2_free,
            value_O2_free=o2_free,
        )

    # ==============================================================
    # 输出
    # ==============================================================
    @staticmethod
    def save_csv(
        results: dict[str, LinearRegressionResult],
        output_path: Path | str,
    ) -> None:
        """保存回归结果 CSV"""
        rows = []
        for name, r in results.items():
            rows.append({
                "parameter": r.param_name,
                "value_N2": r.value_N2,
                "uncertainty_N2": r.uncertainty_N2,
                "value_O2_fixed": r.value_O2_fixed,
                "value_O2_fixed_err": r.value_O2_fixed_err,
                "R_squared": r.R_squared,
                "n_points": r.n_points,
                "value_N2_free_check": r.value_N2_free,
                "value_O2_free_check": r.value_O2_free,
            })
        df = pd.DataFrame(rows)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(output_path), index=False)
        logger.info(f"  线性回归结果已保存: {output_path}")

    def plot_results(
        self,
        results: dict[str, LinearRegressionResult],
        output_dir: Path | str,
    ) -> None:
        """绘制回归诊断图

        每个参数一张图: 左 = 散点 + 回归线, 右 = 残差
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 中文标签映射
        _LABELS = {
            "gamma0": ("γ₀", "cm⁻¹/atm"),
            "SD_gamma": ("SD_γ", ""),
            "delta0": ("δ₀", "cm⁻¹/atm"),
            "SD_delta": ("SD_δ", ""),
        }

        for name, r in results.items():
            sym, unit = _LABELS.get(name, (name, ""))
            unit_str = f" ({unit})" if unit else ""

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # ── 左图: γ_air vs x_N₂ ──
            # 误差棒
            ax1.errorbar(
                r.x_N2, r.y_obs, yerr=r.y_err,
                fmt="o", ms=8, capsize=4, color="steelblue",
                label=f"Single-spectrum {sym}_air",
            )

            # 回归线: 从 x_N₂=0 到 x_N₂=max
            x_line = np.linspace(0, max(r.x_N2) * 1.1, 100)
            # 对应的 x_O₂ = 1 - x_N₂ (对于二元混合气)
            x_O2_line = 1.0 - x_line
            y_line = r.value_O2_fixed * x_O2_line + r.value_N2 * x_line
            ax1.plot(x_line, y_line, "r-", lw=2,
                     label=(f"Linear model: {sym}_O₂={r.value_O2_fixed:.4f} (fixed)"
                            f"\n{sym}_N₂={r.value_N2:.4f} ± {r.uncertainty_N2:.4f}"))

            # 纯 O₂ 点 (x_N₂=0)
            ax1.plot(0, r.value_O2_fixed, "s", ms=10, color="crimson",
                     zorder=5, label=f"Pure O₂ multi-fit: {r.value_O2_fixed:.4f}")

            ax1.set_xlabel("x_N₂ (N₂ mole fraction)")
            ax1.set_ylabel(f"{sym}_total{unit_str}")
            ax1.set_title(
                f"{sym}: Linear Regression"
                + (f" — {self.transition}" if self.transition else "")
            )
            ax1.legend(fontsize=9, loc="best")
            ax1.grid(True, alpha=0.3)

            # 标注 R²
            ax1.text(
                0.02, 0.02,
                f"R² = {r.R_squared:.4f}, n = {r.n_points}",
                transform=ax1.transAxes, fontsize=10,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
            )

            # ── 右图: 残差 ──
            ax2.errorbar(
                r.x_N2, r.residuals, yerr=r.y_err,
                fmt="o", ms=8, capsize=4, color="tomato",
            )
            ax2.axhline(0, color="gray", lw=1, ls="--")
            ax2.set_xlabel("x_N₂ (N₂ mole fraction)")
            ax2.set_ylabel(f"Residual{unit_str}")
            res_std = float(np.std(r.residuals))
            ax2.set_title(f"Residuals (σ = {res_std:.6f})")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            save_path = output_dir / f"linear_regression_{name}.png"
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  图表已保存: {save_path}")

        # ── 汇总图: 所有参数在一张图上 ──
        valid_results = {k: v for k, v in results.items()
                         if v.n_points >= 2}
        if len(valid_results) >= 2:
            n_params = len(valid_results)
            fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))
            if n_params == 1:
                axes = axes[np.newaxis, :]

            for i, (name, r) in enumerate(valid_results.items()):
                sym, unit = _LABELS.get(name, (name, ""))
                unit_str = f" ({unit})" if unit else ""

                ax1, ax2 = axes[i]

                ax1.errorbar(r.x_N2, r.y_obs, yerr=r.y_err,
                             fmt="o", ms=6, capsize=3, color="steelblue")
                x_line = np.linspace(0, max(r.x_N2) * 1.1, 100)
                x_O2_line = 1.0 - x_line
                y_line = r.value_O2_fixed * x_O2_line + r.value_N2 * x_line
                ax1.plot(x_line, y_line, "r-", lw=1.5)
                ax1.plot(0, r.value_O2_fixed, "s", ms=8, color="crimson",
                         zorder=5)
                ax1.set_ylabel(f"{sym}{unit_str}")
                ax1.set_title(
                    f"{sym}_N₂ = {r.value_N2:.4f} ± {r.uncertainty_N2:.4f}"
                    f"  (R²={r.R_squared:.3f})"
                )
                ax1.grid(True, alpha=0.3)

                ax2.errorbar(r.x_N2, r.residuals, yerr=r.y_err,
                             fmt="o", ms=6, capsize=3, color="tomato")
                ax2.axhline(0, color="gray", lw=1, ls="--")
                ax2.set_ylabel("Residual")
                ax2.grid(True, alpha=0.3)

            axes[-1][0].set_xlabel("x_N₂")
            axes[-1][1].set_xlabel("x_N₂")
            fig.suptitle(
                f"N₂ Broadening — Linear Regression Summary"
                + (f" ({self.transition})" if self.transition else ""),
                fontsize=14, y=1.01,
            )
            fig.tight_layout()
            save_path = output_dir / "linear_regression_summary.png"
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  汇总图已保存: {save_path}")

