"""衰荡时间过滤模块测试"""

import numpy as np

from crds_process.ringdown.filtering import (
    filter_ringdown_times,
    iqr_filter,
    sigma_clip_filter,
)


class TestSigmaClip:
    def test_removes_outliers(self, sample_tau):
        filtered = sigma_clip_filter(sample_tau, sigma=3.0)
        assert len(filtered) < len(sample_tau)
        assert np.all(np.abs(filtered - np.mean(filtered)) < 5 * np.std(filtered))

    def test_preserves_clean_data(self):
        clean = np.ones(50) * 90.0
        filtered = sigma_clip_filter(clean, sigma=3.0)
        assert len(filtered) == 50


class TestIQR:
    def test_removes_outliers(self, sample_tau):
        filtered = iqr_filter(sample_tau, factor=1.5)
        assert len(filtered) < len(sample_tau)


class TestUnifiedFilter:
    def test_sigma_clip_method(self, sample_tau):
        filtered = filter_ringdown_times(sample_tau, method="sigma_clip")
        assert len(filtered) > 0

    def test_iqr_method(self, sample_tau):
        filtered = filter_ringdown_times(sample_tau, method="iqr")
        assert len(filtered) > 0

    def test_invalid_method_raises(self, sample_tau):
        import pytest
        with pytest.raises(ValueError, match="未知的过滤方法"):
            filter_ringdown_times(sample_tau, method="invalid")

