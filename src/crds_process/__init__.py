"""CRDS Data Process - 腔衰荡光谱数据处理工具包

从原始衰荡信号到 MATS 光谱参数拟合的完整处理流程。
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    """延迟导入 pipeline 函数，避免循环导入"""
    _exports = {
        "run_pipeline",
        "step1_ringdown",
        "step2_etalon",
        "step3_mats",
        "step4_multi_fit",
    }
    if name in _exports:
        from crds_process import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'crds_process' has no attribute {name!r}")
