"""基线处理模块"""

from crds_process.baseline.etalon import (
    EtalonRemover,
    EtalonFitResult,
    hitran_detect_absorption,
    plot_etalon_removal,
)

__all__ = [
    "EtalonRemover",
    "EtalonFitResult",
    "hitran_detect_absorption",
    "plot_etalon_removal",
]
