"""基线拟合模块测试"""

import numpy as np
import pandas as pd

from crds_process.baseline.fitting import fit_polynomial_baseline, subtract_baseline


class TestPolynomialBaseline:
    def test_linear_baseline(self):
        wn = np.linspace(0, 10, 100)
        alpha = 2.0 * wn + 1.0  # 纯线性
        regions = [[0, 3], [7, 10]]

        baseline = fit_polynomial_baseline(wn, alpha, regions, order=1)
        np.testing.assert_allclose(baseline, alpha, atol=1e-10)


class TestSubtractBaseline:
    def test_baseline_subtraction(self, sample_spectrum):
        wn, alpha, true_baseline = sample_spectrum
        df = pd.DataFrame({"wavenumber": wn, "alpha": alpha})

        # 使用光谱两端作为基线区域
        regions = [[9290.0, 9290.05], [9290.55, 9290.6]]
        result = subtract_baseline(df, regions=regions, method="polynomial", poly_order=1)

        assert "baseline" in result.columns
        assert "alpha_corrected" in result.columns

