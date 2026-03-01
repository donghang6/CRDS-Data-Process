"""衰荡时间统计处理

对每个波数点的衰荡事件进行统计处理，获取代表性的衰荡时间。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crds_process.io.readers import ScanData
from crds_process.ringdown.filtering import filter_ringdown_times


@dataclass
class RingdownResult:
    """单个波数点的衰荡时间处理结果"""
    wavenumber: float       # 波数 (cm⁻¹)
    tau_mean: float         # 过滤后的平均衰荡时间 (μs)
    tau_std: float          # 标准差 (μs)
    tau_sem: float          # 标准误差 (μs)
    n_raw: int              # 原始事件数
    n_filtered: int         # 过滤后事件数
    temperature: float      # 平均温度 (°C)
    pressure: float         # 平均压力 (Torr)


def process_single_scan(
    scan: ScanData,
    filter_method: str = "sigma_clip",
    sigma: float = 3.0,
    iqr_factor: float = 1.5,
    min_events: int = 10,
) -> RingdownResult | None:
    """处理单个扫描点的衰荡时间

    Parameters
    ----------
    scan : ScanData
        扫描数据
    filter_method : str
        离群值过滤方法
    sigma : float
        sigma-clip 阈值
    iqr_factor : float
        IQR 因子
    min_events : int
        最少事件数，不足则返回 None

    Returns
    -------
    RingdownResult or None
    """
    tau_filtered = filter_ringdown_times(
        scan.tau, method=filter_method, sigma=sigma, iqr_factor=iqr_factor
    )

    if len(tau_filtered) < min_events:
        return None

    return RingdownResult(
        wavenumber=scan.meta.wavenumber,
        tau_mean=float(np.mean(tau_filtered)),
        tau_std=float(np.std(tau_filtered, ddof=1)),
        tau_sem=float(np.std(tau_filtered, ddof=1) / np.sqrt(len(tau_filtered))),
        n_raw=len(scan.tau),
        n_filtered=len(tau_filtered),
        temperature=float(np.mean(scan.temperature)),
        pressure=float(np.mean(scan.pressure)),
    )


def process_all_scans(
    scans: list[ScanData],
    filter_method: str = "sigma_clip",
    sigma: float = 3.0,
    iqr_factor: float = 1.5,
    min_events: int = 10,
) -> list[RingdownResult]:
    """批量处理所有扫描点

    Parameters
    ----------
    scans : list[ScanData]
        扫描数据列表
    filter_method, sigma, iqr_factor, min_events
        同 process_single_scan

    Returns
    -------
    list[RingdownResult]
        处理结果列表（按波数排序，已剔除无效点）
    """
    results = []
    for scan in scans:
        r = process_single_scan(
            scan, filter_method=filter_method, sigma=sigma,
            iqr_factor=iqr_factor, min_events=min_events,
        )
        if r is not None:
            results.append(r)

    results.sort(key=lambda r: r.wavenumber)
    return results

