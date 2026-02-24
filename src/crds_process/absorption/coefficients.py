"""衰荡时间 → 吸收系数转换

核心公式: α = (1/c) × (1/τ - 1/τ₀)

其中:
    α : 吸收系数 (cm⁻¹)
    c : 光速 (cm/s)
    τ : 衰荡时间 (s)
    τ₀: 空腔衰荡时间 (s)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crds_process.ringdown.processing import RingdownResult


def tau_to_alpha(
    tau_us: float | np.ndarray,
    tau0_us: float,
    c_cm_s: float = 2.99792458e10,
) -> float | np.ndarray:
    """将衰荡时间转换为吸收系数

    Parameters
    ----------
    tau_us : float or np.ndarray
        衰荡时间 (μs)
    tau0_us : float
        空腔（基线）衰荡时间 (μs)
    c_cm_s : float
        光速 (cm/s)

    Returns
    -------
    float or np.ndarray
        吸收系数 (cm⁻¹)
    """
    tau_s = np.asarray(tau_us) * 1e-6
    tau0_s = tau0_us * 1e-6
    return (1.0 / c_cm_s) * (1.0 / tau_s - 1.0 / tau0_s)


def ringdown_results_to_spectrum(
    results: list[RingdownResult],
    tau0_us: float,
    c_cm_s: float = 2.99792458e10,
) -> pd.DataFrame:
    """将衰荡处理结果转换为吸收光谱

    Parameters
    ----------
    results : list[RingdownResult]
        衰荡时间处理结果列表
    tau0_us : float
        空腔衰荡时间 (μs)
    c_cm_s : float
        光速 (cm/s)

    Returns
    -------
    pd.DataFrame
        包含 wavenumber, alpha, alpha_err 等列的光谱数据
    """
    wavenumbers = np.array([r.wavenumber for r in results])
    tau_means = np.array([r.tau_mean for r in results])
    tau_sems = np.array([r.tau_sem for r in results])

    alpha = tau_to_alpha(tau_means, tau0_us, c_cm_s)

    # 误差传播: δα = (1/c) × (δτ / τ²)
    tau_s = tau_means * 1e-6
    tau_sem_s = tau_sems * 1e-6
    alpha_err = (1.0 / c_cm_s) * (tau_sem_s / tau_s**2)

    return pd.DataFrame({
        "wavenumber": wavenumbers,
        "alpha": alpha,
        "alpha_err": alpha_err,
        "tau_mean": tau_means,
        "tau_std": [r.tau_std for r in results],
        "temperature": [r.temperature for r in results],
        "pressure": [r.pressure for r in results],
    })

