"""基线拟合与扣除

支持多项式拟合和样条拟合两种基线拟合方法。
通过选取无吸收峰的波数区域来确定基线。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def _select_baseline_points(
    wavenumber: np.ndarray,
    alpha: np.ndarray,
    regions: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """根据指定波数区间选取基线数据点

    Parameters
    ----------
    wavenumber : np.ndarray
        波数数组
    alpha : np.ndarray
        吸收系数数组
    regions : list[list[float]]
        基线区域列表，每个元素 [low, high]

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        基线区域的 (wavenumber, alpha)
    """
    mask = np.zeros(len(wavenumber), dtype=bool)
    for low, high in regions:
        mask |= (wavenumber >= low) & (wavenumber <= high)
    return wavenumber[mask], alpha[mask]


def fit_polynomial_baseline(
    wavenumber: np.ndarray,
    alpha: np.ndarray,
    regions: list[list[float]],
    order: int = 3,
) -> np.ndarray:
    """使用多项式拟合基线

    Parameters
    ----------
    wavenumber : np.ndarray
        完整波数数组
    alpha : np.ndarray
        完整吸收系数数组
    regions : list[list[float]]
        用于拟合的基线区域
    order : int
        多项式阶数

    Returns
    -------
    np.ndarray
        基线值（与 wavenumber 等长）
    """
    wn_base, alpha_base = _select_baseline_points(wavenumber, alpha, regions)
    if len(wn_base) < order + 1:
        raise ValueError(f"基线数据点不足（{len(wn_base)}个），需要至少 {order + 1} 个点")

    coeffs = np.polyfit(wn_base, alpha_base, order)
    return np.polyval(coeffs, wavenumber)


def fit_spline_baseline(
    wavenumber: np.ndarray,
    alpha: np.ndarray,
    regions: list[list[float]],
    smoothing: float = 1e-6,
) -> np.ndarray:
    """使用样条拟合基线

    Parameters
    ----------
    wavenumber : np.ndarray
        完整波数数组
    alpha : np.ndarray
        完整吸收系数数组
    regions : list[list[float]]
        用于拟合的基线区域
    smoothing : float
        样条平滑因子

    Returns
    -------
    np.ndarray
        基线值（与 wavenumber 等长）
    """
    wn_base, alpha_base = _select_baseline_points(wavenumber, alpha, regions)
    if len(wn_base) < 4:
        raise ValueError(f"基线数据点不足（{len(wn_base)}个），样条拟合至少需要4个点")

    # 确保排序
    sort_idx = np.argsort(wn_base)
    wn_base = wn_base[sort_idx]
    alpha_base = alpha_base[sort_idx]

    spline = UnivariateSpline(wn_base, alpha_base, s=smoothing)
    return spline(wavenumber)


def subtract_baseline(
    spectrum_df: pd.DataFrame,
    regions: list[list[float]],
    method: str = "polynomial",
    poly_order: int = 3,
    spline_smoothing: float = 1e-6,
) -> pd.DataFrame:
    """从吸收光谱中扣除基线

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        吸收光谱，须包含 'wavenumber' 和 'alpha' 列
    regions : list[list[float]]
        基线区域
    method : str
        拟合方法 ("polynomial" 或 "spline")
    poly_order : int
        多项式阶数
    spline_smoothing : float
        样条平滑因子

    Returns
    -------
    pd.DataFrame
        新增 'baseline' 和 'alpha_corrected' 列
    """
    df = spectrum_df.copy()
    wn = df["wavenumber"].values
    alpha = df["alpha"].values

    if method == "polynomial":
        baseline = fit_polynomial_baseline(wn, alpha, regions, order=poly_order)
    elif method == "spline":
        baseline = fit_spline_baseline(wn, alpha, regions, smoothing=spline_smoothing)
    else:
        raise ValueError(f"未知的基线拟合方法: {method}")

    df["baseline"] = baseline
    df["alpha_corrected"] = alpha - baseline
    return df

