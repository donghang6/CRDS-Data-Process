"""气体类型配置

解析原始数据目录名，区分纯 O₂ 和 O₂/N₂ 混合气，
并生成对应的 MATS 拟合参数。

目录命名约定:
    纯 O₂:   data/raw/O2/{跃迁}/{压力Torr}/
    O₂+N₂:  data/raw/O2_N2/{跃迁}/O2 300Torr N2 100Torr/
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GasConfig:
    """气体类型配置

    Attributes
    ----------
    gas_type : str
        气体类型标识 ("O2" 或 "O2_N2")
    o2_pressure : float
        O₂ 分压 (Torr)
    n2_pressure : float
        N₂ 分压 (Torr)，纯 O₂ 时为 0
    total_pressure : float
        总压力 (Torr)
    o2_fraction : float
        O₂ 摩尔分数 (0~1)
    """
    gas_type: str
    o2_pressure: float
    n2_pressure: float = 0.0
    total_pressure: float = 0.0
    o2_fraction: float = 1.0

    def __post_init__(self):
        if self.total_pressure == 0:
            self.total_pressure = self.o2_pressure + self.n2_pressure
        if self.total_pressure > 0:
            self.o2_fraction = self.o2_pressure / self.total_pressure

    @property
    def n2_fraction(self) -> float:
        """N₂ 摩尔分数 (0~1)"""
        if self.total_pressure > 0:
            return self.n2_pressure / self.total_pressure
        return 0.0

    @property
    def diluent(self) -> str:
        """MATS diluent 参数

        纯 O₂ → "O2"
        O₂+N₂ 混合 → "air" (仅用于 MATS Spectrum 参数, 实际展宽由 Diluent 控制)
        """
        return "O2" if self.gas_type == "O2" else "air"

    @property
    def molefraction(self) -> dict:
        """MATS molefraction 参数 (分子7 = O₂)"""
        return {7: self.o2_fraction}

    @property
    def Diluent(self) -> dict:
        """MATS Diluent 参数 (单光谱拟合用)

        纯 O₂:  单一稀释气 O₂
        O₂+N₂: 用 air 近似 (单光谱无法分离 γ₀_O₂ 和 γ₀_N₂)
        """
        if self.gas_type == "O2":
            return {"O2": {"composition": 1, "m": 31.9988}}
        else:
            return {"air": {"composition": 1, "m": 28.964}}

    @property
    def Diluent_dual(self) -> dict:
        """MATS Diluent 参数 (多光谱联合拟合用, 双稀释气)

        纯 O₂:  单一稀释气 O₂
        O₂+N₂: 双稀释气, 按摩尔分数加权, 分别拟合 γ₀_O₂ 和 γ₀_N₂
        """
        if self.gas_type == "O2":
            return {"O2": {"composition": 1, "m": 31.9988}}
        else:
            # O₂ + N₂ 混合: 双 diluent 分别对应 γ₀_O₂ 和 γ₀_N₂
            o2_frac = self.o2_fraction
            n2_frac = self.n2_fraction
            return {
                "O2": {"composition": o2_frac, "m": 31.9988},
                "N2": {"composition": n2_frac, "m": 28.014},
            }

    @property
    def label(self) -> str:
        """简短标签，用于输出目录名和日志"""
        if self.gas_type == "O2":
            return f"{self.o2_pressure:.0f}Torr"
        else:
            return f"O2_{self.o2_pressure:.0f}Torr_N2_{self.n2_pressure:.0f}Torr"

    def to_fitter_kwargs(self) -> dict:
        """生成 MATSFitter 的 diluent/molefraction/Diluent 参数"""
        return dict(
            diluent=self.diluent,
            molefraction=self.molefraction,
            Diluent=self.Diluent,
            gas_type=self.gas_type,
        )


# ==================================================================
# 目录名解析
# ==================================================================
_RE_PURE_O2 = re.compile(r"^(\d+)Torr$")
_RE_MIX_O2_N2 = re.compile(
    r"O2\s+(\d+)\s*Torr\s+N2\s+(\d+)\s*Torr", re.IGNORECASE
)


def parse_gas_dir(dir_name: str, gas_type: str) -> GasConfig:
    """从目录名解析气体配置

    Parameters
    ----------
    dir_name : str
        压力目录名 (如 "100Torr" 或 "O2 300Torr N2 100Torr")
    gas_type : str
        气体类型 ("O2" 或 "O2_N2")

    Returns
    -------
    GasConfig
    """
    if gas_type == "O2":
        m = _RE_PURE_O2.match(dir_name.strip())
        if m:
            p = float(m.group(1))
            return GasConfig(gas_type="O2", o2_pressure=p)
        # 回退: 目录名可能不规范，尝试提取数字
        nums = re.findall(r"(\d+)", dir_name)
        if nums:
            p = float(nums[0])
            return GasConfig(gas_type="O2", o2_pressure=p)
        raise ValueError(f"无法从 '{dir_name}' 解析纯 O₂ 压力")

    elif gas_type == "O2_N2":
        m = _RE_MIX_O2_N2.search(dir_name)
        if m:
            o2_p = float(m.group(1))
            n2_p = float(m.group(2))
            return GasConfig(
                gas_type="O2_N2",
                o2_pressure=o2_p,
                n2_pressure=n2_p,
            )
        raise ValueError(f"无法从 '{dir_name}' 解析 O₂/N₂ 混合气压力")

    else:
        raise ValueError(f"未知气体类型: {gas_type}")

