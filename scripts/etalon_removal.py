#!/usr/bin/env python3
"""标准具效应批量去除 — 基于 HITRAN 模拟自动检测吸收峰

自动扫描 output/results/ringdown/{跃迁波数}/{压力}/ 下的衰荡时间数据，
使用 HITRAN 模拟确定吸收峰排除区域，去除标准具效应。

目录结构:
    output/results/ringdown/{跃迁波数}/{压力}/ringdown_results.csv  ← 输入
    output/results/etalon/{跃迁波数}/{压力}/                       ← 输出

运行: python scripts/etalon_removal.py
"""

from crds_process.baseline.etalon import batch_etalon_removal

if __name__ == "__main__":
    batch_etalon_removal()
