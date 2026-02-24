"""配置加载模块测试"""

from crds_process.config import Settings, load_config


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.cavity.length_cm == 42.0
        assert s.gas.species == "H2O"

    def test_load_nonexistent_config(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, Settings)

