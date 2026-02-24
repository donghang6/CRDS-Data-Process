"""实验配置管理模块

使用 Pydantic 模型加载和验证 YAML 配置文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ============================================================
# 配置数据模型
# ============================================================

class CavityConfig(BaseModel):
    """光腔参数"""
    length_cm: float = Field(42.0, description="腔长 (cm)")
    mirror_reflectivity: float = Field(0.99997, description="镜面反射率")


class RingdownConfig(BaseModel):
    """衰荡时间处理参数"""
    filter_method: str = Field("sigma_clip", description="离群值过滤方法")
    sigma_clip_threshold: float = Field(3.0, description="sigma-clip 阈值")
    iqr_factor: float = Field(1.5, description="IQR 因子")
    min_events_per_point: int = Field(10, description="每个波数点最少衰荡事件数")


class BaselineConfig(BaseModel):
    """基线拟合参数"""
    method: str = Field("polynomial", description="拟合方法")
    poly_order: int = Field(3, description="多项式阶数")
    spline_smoothing: float = Field(1.0e-6, description="样条平滑因子")
    regions: list[list[float]] = Field(default_factory=list, description="基线区域 (波数范围)")


class AbsorptionConfig(BaseModel):
    """吸收系数计算参数"""
    speed_of_light_cm_s: float = Field(2.99792458e10, description="光速 (cm/s)")
    tau0_us: float = Field(90.5, description="空腔衰荡时间 (μs)")


class GasConfig(BaseModel):
    """气体条件"""
    species: str = Field("H2O", description="气体种类")
    temperature_K: float = Field(296.0, description="温度 (K)")
    pressure_torr: float = Field(30.0, description="压力 (Torr)")
    isotopologue: int = Field(1, description="同位素编号")


class MATSConfig(BaseModel):
    """MATS 拟合参数"""
    database: str = Field("HITRAN", description="光谱数据库")
    wavenumber_range: list[float] = Field(default_factory=lambda: [9290.0, 9290.6])
    fit_parameters: list[str] = Field(
        default_factory=lambda: ["nu", "sw", "gamma_self", "gamma_air", "delta_air", "n_air"]
    )
    max_iterations: int = Field(100)
    convergence_threshold: float = Field(1.0e-8)


class PathsConfig(BaseModel):
    """输入输出路径"""
    raw_data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "output"
    figures_dir: str = "output/figures"
    results_dir: str = "output/results"


class Settings(BaseModel):
    """全局配置"""
    cavity: CavityConfig = Field(default_factory=CavityConfig)
    ringdown: RingdownConfig = Field(default_factory=RingdownConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    absorption: AbsorptionConfig = Field(default_factory=AbsorptionConfig)
    gas: GasConfig = Field(default_factory=GasConfig)
    mats: MATSConfig = Field(default_factory=MATSConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


# ============================================================
# 配置加载函数
# ============================================================

def load_config(config_path: Optional[str | Path] = None) -> Settings:
    """从 YAML 文件加载配置

    Parameters
    ----------
    config_path : str or Path, optional
        配置文件路径，默认使用 config/default.yaml

    Returns
    -------
    Settings
        验证后的配置对象
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return Settings()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(**raw) if raw else Settings()

