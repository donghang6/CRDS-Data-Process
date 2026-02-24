#!/usr/bin/env python3
"""标准具效应去除 — 单频模型拟合测试

排除吸收峰区域，仅用基线区域数据拟合标准具参数，
然后在全波段扣除标准具正弦分量。

用法:
    python scripts/test_etalon_removal.py
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

# 吸收峰区域（排除，不参与拟合）
EXCLUDE_REGIONS = [[9385.9, 9386.5]]


def main():
    print("=" * 60)
    print("  标准具效应去除 — 单频模型 (排除吸收峰区域)")
    print("=" * 60)

    if not RINGDOWN_CSV.exists():
        print(f"\n[ERROR] 找不到数据文件: {RINGDOWN_CSV}")
        print("请先运行: python scripts/process_raw.py")
        return

    df = pd.read_csv(RINGDOWN_CSV)
    wn = df["wavenumber"].values
    tau = df["tau_mean"].values

    print(f"\n数据: {RINGDOWN_CSV.name}")
    print(f"  点数: {len(wn)}")
    print(f"  波数范围: {wn.min():.5f} ~ {wn.max():.5f} cm⁻¹")
    print(f"  τ 范围: {tau.min():.4f} ~ {tau.max():.4f} μs")
    print(f"  排除区域: {EXCLUDE_REGIONS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 单频标准具 + 多项式基线，排除吸收峰 ---
    print(f"\n{'─' * 50}")
    print("  模型: a·sin(2πfx + φ) + poly(x, order=3)")
    print(f"  排除吸收峰: {EXCLUDE_REGIONS}")
    print(f"{'─' * 50}")

    remover = EtalonRemover(
        n_etalons=1,
        poly_order=3,
        exclude_regions=EXCLUDE_REGIONS,
        max_iter=20000,
    )
    result = remover.fit(wn, tau)

    print(result.summary())
    print(f"\n  拟合成功: {result.model_result.success}")
    print(f"  迭代次数: {result.model_result.nfev}")

    # 保存图
    plot_etalon_removal(
        result,
        title="Etalon Removal — τ(ν), single-freq (exclude absorption)",
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

