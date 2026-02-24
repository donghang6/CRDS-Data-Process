"""原始数据读取模块

解析 CRDS 原始数据文件名和内容。
文件名格式: "  1 9290.08204 20260121100904.txt"
    - 字段1: 扫描序号
    - 字段2: 波数 (cm⁻¹)
    - 字段3: 时间戳 (YYYYMMDDHHmmss)

文件内容: 4列空格分隔
    - 列1: 衰荡时间 (μs)
    - 列2: 拟合残差
    - 列3: 温度 (°C)
    - 列4: 压力 (Torr)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ScanMeta:
    """单个扫描点的元信息"""
    index: int              # 扫描序号
    wavenumber: float       # 波数 (cm⁻¹)
    timestamp: datetime     # 采集时间戳
    filepath: Path          # 文件路径


@dataclass
class ScanData:
    """单个扫描点的完整数据"""
    meta: ScanMeta
    tau: np.ndarray         # 衰荡时间数组 (μs)
    residual: np.ndarray    # 拟合残差数组
    temperature: np.ndarray # 温度数组 (°C)
    pressure: np.ndarray    # 压力数组 (Torr)


# ============================================================
# 文件名解析
# ============================================================

_FILENAME_PATTERN = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+(\d{14})\.txt$"
)


def parse_filename(filepath: str | Path) -> ScanMeta:
    """解析数据文件名提取扫描元信息

    Parameters
    ----------
    filepath : str or Path
        数据文件路径

    Returns
    -------
    ScanMeta
        解析后的元信息

    Raises
    ------
    ValueError
        文件名格式不符合预期
    """
    filepath = Path(filepath)
    match = _FILENAME_PATTERN.match(filepath.name)
    if not match:
        raise ValueError(f"无法解析文件名: {filepath.name}")

    index = int(match.group(1))
    wavenumber = float(match.group(2))
    timestamp = datetime.strptime(match.group(3), "%Y%m%d%H%M%S")

    return ScanMeta(
        index=index,
        wavenumber=wavenumber,
        timestamp=timestamp,
        filepath=filepath,
    )


# ============================================================
# 数据读取
# ============================================================

def read_ringdown_file(filepath: str | Path) -> ScanData:
    """读取单个衰荡数据文件

    Parameters
    ----------
    filepath : str or Path
        数据文件路径

    Returns
    -------
    ScanData
        包含元信息和衰荡数据的完整数据对象
    """
    filepath = Path(filepath)
    meta = parse_filename(filepath)

    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return ScanData(
        meta=meta,
        tau=data[:, 0],
        residual=data[:, 1],
        temperature=data[:, 2],
        pressure=data[:, 3],
    )


def load_scan_directory(directory: str | Path) -> list[ScanData]:
    """批量读取目录下所有扫描数据文件

    Parameters
    ----------
    directory : str or Path
        数据目录路径

    Returns
    -------
    list[ScanData]
        按波数排序的扫描数据列表
    """
    directory = Path(directory)
    scan_files = sorted(directory.glob("*.txt"))

    scans = []
    for f in scan_files:
        try:
            scans.append(read_ringdown_file(f))
        except (ValueError, Exception) as e:
            print(f"[WARNING] 跳过文件 {f.name}: {e}")

    # 按波数排序
    scans.sort(key=lambda s: s.meta.wavenumber)
    return scans


def scans_to_dataframe(scans: list[ScanData]) -> pd.DataFrame:
    """将扫描数据列表转换为 DataFrame 汇总表

    每行代表一个扫描点，包含波数、平均衰荡时间、标准差等统计量。

    Parameters
    ----------
    scans : list[ScanData]
        扫描数据列表

    Returns
    -------
    pd.DataFrame
        汇总数据表
    """
    records = []
    for s in scans:
        records.append({
            "index": s.meta.index,
            "wavenumber": s.meta.wavenumber,
            "timestamp": s.meta.timestamp,
            "tau_mean": np.mean(s.tau),
            "tau_std": np.std(s.tau, ddof=1),
            "tau_count": len(s.tau),
            "temperature_mean": np.mean(s.temperature),
            "pressure_mean": np.mean(s.pressure),
        })

    return pd.DataFrame(records)

