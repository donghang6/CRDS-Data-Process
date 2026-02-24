"""衰荡时间离群值过滤

支持 sigma-clip 和 IQR 两种离群值剔除方法。
"""

from __future__ import annotations

import numpy as np


def sigma_clip_filter(tau: np.ndarray, sigma: float = 3.0, max_iter: int = 5) -> np.ndarray:
    """使用 sigma-clip 方法过滤衰荡时间离群值

    Parameters
    ----------
    tau : np.ndarray
        衰荡时间数组
    sigma : float
        裁剪阈值（标准差倍数）
    max_iter : int
        最大迭代次数

    Returns
    -------
    np.ndarray
        过滤后的衰荡时间数组
    """
    filtered = tau.copy()
    for _ in range(max_iter):
        mean = np.mean(filtered)
        std = np.std(filtered, ddof=1)
        if std == 0:
            break
        mask = np.abs(filtered - mean) < sigma * std
        if np.sum(mask) == len(filtered):
            break
        filtered = filtered[mask]
    return filtered


def iqr_filter(tau: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """使用 IQR 方法过滤衰荡时间离群值

    Parameters
    ----------
    tau : np.ndarray
        衰荡时间数组
    factor : float
        IQR 因子

    Returns
    -------
    np.ndarray
        过滤后的衰荡时间数组
    """
    q1 = np.percentile(tau, 25)
    q3 = np.percentile(tau, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (tau >= lower) & (tau <= upper)
    return tau[mask]


def filter_ringdown_times(
    tau: np.ndarray,
    method: str = "sigma_clip",
    sigma: float = 3.0,
    iqr_factor: float = 1.5,
) -> np.ndarray:
    """统一接口：过滤衰荡时间离群值

    Parameters
    ----------
    tau : np.ndarray
        衰荡时间数组
    method : str
        过滤方法 ("sigma_clip" 或 "iqr")
    sigma : float
        sigma-clip 阈值
    iqr_factor : float
        IQR 因子

    Returns
    -------
    np.ndarray
        过滤后的衰荡时间数组
    """
    if method == "sigma_clip":
        return sigma_clip_filter(tau, sigma=sigma)
    elif method == "iqr":
        return iqr_filter(tau, factor=iqr_factor)
    else:
        raise ValueError(f"未知的过滤方法: {method}，支持 'sigma_clip' 或 'iqr'")

