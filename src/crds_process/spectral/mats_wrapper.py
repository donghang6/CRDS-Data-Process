"""MATS (Multi-spectrum Analysis Tool for Spectroscopy) 封装

将处理后的吸收光谱数据转换为 MATS 所需格式，
执行多光谱拟合，提取线强、展宽系数等光谱参数。

References
----------
- MATS GitHub: https://github.com/usnistgov/MATS
- MATS Documentation: https://pages.nist.gov/MATS/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    """MATS 拟合结果"""
    parameters: dict[str, float] = field(default_factory=dict)
    uncertainties: dict[str, float] = field(default_factory=dict)
    residuals: np.ndarray | None = None
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    fitted_spectrum: np.ndarray | None = None
    raw_result: Any = None  # MATS 原始返回对象


def prepare_mats_input(
    spectrum_df: pd.DataFrame,
    wavenumber_col: str = "wavenumber",
    alpha_col: str = "alpha_corrected",
    alpha_err_col: str = "alpha_err",
) -> dict:
    """将 DataFrame 转换为 MATS 所需的输入格式

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        基线扣除后的吸收光谱
    wavenumber_col : str
        波数列名
    alpha_col : str
        吸收系数列名
    alpha_err_col : str
        吸收系数误差列名

    Returns
    -------
    dict
        MATS 格式的输入数据
    """
    data = {
        "wavenumber": spectrum_df[wavenumber_col].values,
        "alpha": spectrum_df[alpha_col].values,
    }
    if alpha_err_col in spectrum_df.columns:
        data["alpha_err"] = spectrum_df[alpha_err_col].values

    return data


def run_mats_fit(
    spectrum_data: dict,
    temperature_K: float,
    pressure_torr: float,
    species: str = "H2O",
    isotopologue: int = 1,
    database: str = "HITRAN",
    wavenumber_range: Optional[list[float]] = None,
    fit_parameters: Optional[list[str]] = None,
    max_iterations: int = 100,
) -> FitResult:
    """执行 MATS 光谱拟合

    Parameters
    ----------
    spectrum_data : dict
        由 prepare_mats_input 准备的输入数据
    temperature_K : float
        温度 (K)
    pressure_torr : float
        压力 (Torr)
    species : str
        气体种类
    isotopologue : int
        同位素编号
    database : str
        数据库名称 ("HITRAN", "HITEMP" 等)
    wavenumber_range : list[float], optional
        波数范围 [min, max]
    fit_parameters : list[str], optional
        拟合参数列表
    max_iterations : int
        最大迭代次数

    Returns
    -------
    FitResult

    Notes
    -----
    此函数为 MATS 库的封装接口。使用前需安装 MATS:
        pip install MATS-venv
    或从 GitHub 安装:
        pip install git+https://github.com/usnistgov/MATS.git
    """
    try:
        from MATS import Spectrum as MATSSpectrum
        from MATS import Dataset as MATSDataset
        from MATS import Fit as MATSFit
    except ImportError:
        raise ImportError(
            "MATS 未安装。请使用以下命令安装:\n"
            "  pip install MATS-venv\n"
            "或:\n"
            "  pip install git+https://github.com/usnistgov/MATS.git"
        )

    # TODO: 根据 MATS 实际 API 实现具体拟合逻辑
    # 以下为拟合框架的骨架代码，需根据 MATS 版本和 API 调整

    raise NotImplementedError(
        "MATS 拟合逻辑需要根据具体的 MATS 版本和数据格式进行实现。"
        "请参考 MATS 文档: https://pages.nist.gov/MATS/"
    )

