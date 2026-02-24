"""绘图工具

提供 CRDS 数据处理各阶段的标准可视化函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 统一图表样式
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_ringdown_histogram(
    tau: np.ndarray,
    tau_filtered: Optional[np.ndarray] = None,
    wavenumber: Optional[float] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """绘制衰荡时间直方图

    Parameters
    ----------
    tau : np.ndarray
        原始衰荡时间
    tau_filtered : np.ndarray, optional
        过滤后的衰荡时间
    wavenumber : float, optional
        波数，用于标题
    save_path : str or Path, optional
        保存路径

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots()
    ax.hist(tau, bins=30, alpha=0.5, label="原始数据", color="gray")
    if tau_filtered is not None:
        ax.hist(tau_filtered, bins=30, alpha=0.7, label="过滤后", color="steelblue")
    ax.set_xlabel("衰荡时间 (μs)")
    ax.set_ylabel("计数")
    title = "衰荡时间分布"
    if wavenumber is not None:
        title += f" @ {wavenumber:.5f} cm⁻¹"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_tau_spectrum(
    wavenumber: np.ndarray,
    tau: np.ndarray,
    tau_err: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """绘制衰荡时间随波数变化的光谱

    Parameters
    ----------
    wavenumber : np.ndarray
        波数数组
    tau : np.ndarray
        平均衰荡时间数组
    tau_err : np.ndarray, optional
        误差棒
    save_path : str or Path, optional
        保存路径

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots()
    if tau_err is not None:
        ax.errorbar(wavenumber, tau, yerr=tau_err, fmt="o-", ms=3, capsize=2, color="steelblue")
    else:
        ax.plot(wavenumber, tau, "o-", ms=3, color="steelblue")
    ax.set_xlabel("波数 (cm⁻¹)")
    ax.set_ylabel("衰荡时间 (μs)")
    ax.set_title("衰荡时间光谱")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_absorption_spectrum(
    spectrum_df: pd.DataFrame,
    show_baseline: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """绘制吸收光谱（含基线）

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        光谱数据，须包含 'wavenumber' 和 'alpha' 列
    show_baseline : bool
        是否显示基线
    save_path : str or Path, optional
        保存路径

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(spectrum_df["wavenumber"], spectrum_df["alpha"], "o-", ms=2, label="吸收系数", color="steelblue")
    if show_baseline and "baseline" in spectrum_df.columns:
        ax1.plot(spectrum_df["wavenumber"], spectrum_df["baseline"], "--", label="基线", color="red", lw=1.5)
    ax1.set_ylabel("吸收系数 (cm⁻¹)")
    ax1.set_title("吸收光谱")
    ax1.legend()

    ax2 = axes[1]
    if "alpha_corrected" in spectrum_df.columns:
        ax2.plot(spectrum_df["wavenumber"], spectrum_df["alpha_corrected"], "o-", ms=2, color="steelblue")
    ax2.set_xlabel("波数 (cm⁻¹)")
    ax2.set_ylabel("基线扣除后 (cm⁻¹)")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_fit_result(
    wavenumber: np.ndarray,
    alpha_exp: np.ndarray,
    alpha_fit: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """绘制拟合结果对比图

    Parameters
    ----------
    wavenumber : np.ndarray
        波数
    alpha_exp : np.ndarray
        实验吸收系数
    alpha_fit : np.ndarray
        拟合吸收系数
    residuals : np.ndarray, optional
        残差
    save_path : str or Path, optional
        保存路径

    Returns
    -------
    plt.Figure
    """
    if residuals is None:
        residuals = alpha_exp - alpha_fit

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(wavenumber, alpha_exp, "o", ms=3, label="实验值", color="steelblue")
    ax1.plot(wavenumber, alpha_fit, "-", label="拟合值", color="red", lw=1.5)
    ax1.set_ylabel("吸收系数 (cm⁻¹)")
    ax1.set_title("MATS 光谱拟合结果")
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(wavenumber, residuals, "o", ms=2, color="gray")
    ax2.axhline(0, color="red", ls="--", lw=1)
    ax2.set_xlabel("波数 (cm⁻¹)")
    ax2.set_ylabel("残差 (cm⁻¹)")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig

