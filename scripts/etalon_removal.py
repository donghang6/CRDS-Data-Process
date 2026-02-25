#!/usr/bin/env python3
"""标准具效应去除 — 基于 HITRAN 模拟自动检测吸收峰

使用 HITRAN 光谱数据库模拟吸收系数谱，自动确定吸收峰排除区域，
仅用基线区域数据拟合标准具参数，然后在全波段扣除标准具正弦分量。

用法:
    python scripts/etalon_removal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crds_process.baseline.etalon import EtalonRemover, plot_etalon_removal

# ==================================================================
# 配置
# ==================================================================
RINGDOWN_CSV = PROJECT_ROOT / "output" / "results" / "ringdown" / "9386.2076" / "100Torr" / "ringdown_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "results" / "etalon_test"


def main():
    print("=" * 60)
    print("  标准具效应去除 — HITRAN 模拟自动检测吸收峰")
    print("=" * 60)

    if not RINGDOWN_CSV.exists():
        print(f"\n[ERROR] 找不到数据文件: {RINGDOWN_CSV}")
        print("请先运行: python scripts/process_raw.py")
        return

    df = pd.read_csv(RINGDOWN_CSV)
    wn = df["wavenumber"].values
    tau = df["tau_mean"].values
    temperature = float(df["temperature"].mean())
    pressure = float(df["pressure"].mean())

    print(f"\n数据: {RINGDOWN_CSV.name}")
    print(f"  点数: {len(wn)}")
    print(f"  波数范围: {wn.min():.5f} ~ {wn.max():.5f} cm⁻¹")
    print(f"  τ 范围: {tau.min():.4f} ~ {tau.max():.4f} μs")
    print(f"  温度均值: {temperature:.1f} °C")
    print(f"  压力均值: {pressure:.1f} Torr")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- HITRAN 模拟自动检测 + 标准具去除 ---
    print(f"\n{'─' * 50}")
    print("  排除区域: HITRAN 模拟自动检测")
    print("  模型: a·sin(2πfx + φ), 线性去趋势, 迭代交替优化")
    print(f"{'─' * 50}")

    remover = EtalonRemover(
        n_etalons=1,
        poly_order=1,
        exclude_regions="hitran",
        hitran_kwargs={
            "molecule": 7,           # O₂
            "isotopologue": 1,       # ¹⁶O₂
            "threshold_ratio": 0.01, # 吸收系数 > peak*1% 的区域
            "margin": 0.05,          # 两侧扩展 0.05 cm⁻¹
        },
    )
    result = remover.fit(wn, tau, temperature=temperature, pressure_torr=pressure)

    print(result.summary())
    print(f"\n  拟合成功: {result.model_result.success}")
    print(f"  迭代次数: {result.model_result.nfev}")

    # 保存图
    plot_etalon_removal(
        result,
        title="Etalon Removal — HITRAN auto-detect",
        save_path=str(OUTPUT_DIR / "etalon_removal.png"),
    )

    # 保存 CSV
    df_out, _ = remover.fit_df(df, signal_col="tau_mean")
    csv_path = OUTPUT_DIR / "tau_etalon_corrected.csv"
    df_out.to_csv(csv_path, index=False)

    print(f"\n  CSV 已保存: {csv_path.name}")
    print(f"\n  前 5 行:")
    print(df_out[["wavenumber", "tau_mean", "tau_mean_etalon", "tau_mean_no_etalon"]].head().to_string(index=False))

    # 输出汇总
    print(f"\n{'=' * 60}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name:<40s} {f.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()

