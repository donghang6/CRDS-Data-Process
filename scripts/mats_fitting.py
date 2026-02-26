#!/usr/bin/env python3
"""MATS 光谱拟合 — 自动发现 etalon corrected 数据并拟合

自动扫描 output/results/etalon/{跃迁波数}/{压力}/ 下的标准具去除数据，
使用 MATS (Voigt 线形) 拟合吸收光谱，提取线参数。

目录结构:
    output/results/etalon/{跃迁波数}/{压力}/tau_etalon_corrected.csv  ← 输入
    output/results/mats/{跃迁波数}/{压力}/                           ← 输出

运行: python scripts/mats_fitting.py
"""

from crds_process.spectral.mats_wrapper import batch_mats_fitting

if __name__ == "__main__":
    batch_mats_fitting()

