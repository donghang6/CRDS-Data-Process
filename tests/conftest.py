"""pytest 共享 fixtures"""

import numpy as np
import pytest


@pytest.fixture
def sample_tau():
    """生成模拟衰荡时间数据（含少量离群值）"""
    rng = np.random.default_rng(42)
    tau_normal = rng.normal(loc=90.0, scale=0.1, size=50)
    tau_outliers = np.array([85.0, 95.0])  # 离群值
    return np.concatenate([tau_normal, tau_outliers])


@pytest.fixture
def sample_spectrum():
    """生成模拟吸收光谱数据"""
    wn = np.linspace(9290.0, 9290.6, 200)
    # 模拟一个 Lorentzian 吸收峰 + 线性基线
    nu0 = 9290.3
    gamma = 0.02
    peak = 1e-7 * gamma / ((wn - nu0) ** 2 + gamma ** 2)
    baseline = 1e-9 * (wn - 9290.0) + 5e-8
    alpha = peak + baseline
    return wn, alpha, baseline

