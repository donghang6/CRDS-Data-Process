"""基线处理模块"""

from crds_process.baseline.etalon import (
    HitranAbsorptionDetector,
    EtalonRemover,
    EtalonFitResult,
    EtalonBatchProcessor,
    batch_etalon_removal,
    # 向后兼容
    hitran_detect_absorption,
    plot_etalon_removal,
)

__all__ = [
    "HitranAbsorptionDetector",
    "EtalonRemover",
    "EtalonFitResult",
    "EtalonBatchProcessor",
    "batch_etalon_removal",
    "hitran_detect_absorption",
    "plot_etalon_removal",
]
