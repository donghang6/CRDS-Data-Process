"""吸收系数计算模块测试"""

import numpy as np

from crds_process.absorption.coefficients import tau_to_alpha


class TestTauToAlpha:
    def test_basic_conversion(self):
        """τ == τ₀ 时 α 应为 0"""
        alpha = tau_to_alpha(90.0, tau0_us=90.0)
        assert np.isclose(alpha, 0.0)

    def test_positive_absorption(self):
        """τ < τ₀ 时 α 应为正"""
        alpha = tau_to_alpha(89.0, tau0_us=90.0)
        assert alpha > 0

    def test_array_input(self):
        """支持数组输入"""
        tau = np.array([88.0, 89.0, 90.0, 91.0])
        alpha = tau_to_alpha(tau, tau0_us=90.0)
        assert alpha.shape == (4,)
        assert alpha[2] == 0.0  # τ == τ₀

