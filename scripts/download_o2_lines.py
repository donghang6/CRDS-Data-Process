#!/usr/bin/env python3
"""下载 HITRAN 氧气吸收线光谱参数

使用 HAPI 从 HITRAN 数据库下载 9000-10000 cm⁻¹ 波数范围内
线强 > 1e-29 cm/molecule 的 O₂ 吸收线参数。

氧气 (O₂) 在 HITRAN 中的分子编号:  M = 7
主要同位素:
    1 — ¹⁶O₂      (iso_id = 36)
    2 — ¹⁶O¹⁸O    (iso_id = 37)
    3 — ¹⁶O¹⁷O    (iso_id = 38)

用法:
    python scripts/download_o2_lines.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "hitran"

# ── 参数配置 ──
MOLECULE = 7               # O₂ 的 HITRAN 分子编号
ISOTOPOLOGUES = [1, 2, 3]  # 同位素编号
NUMIN = 9000.0             # 波数下限 (cm⁻¹)
NUMAX = 10000.0            # 波数上限 (cm⁻¹)
SW_THRESHOLD = 1e-29       # 线强阈值 (cm/molecule)

# 需要下载的参数列
PARAMS = [
    "molec_id", "local_iso_id", "nu", "sw", "a",
    "gamma_air", "gamma_self", "elower", "n_air", "delta_air",
]


def main():
    print("=" * 60)
    print("  HITRAN O₂ 吸收线参数下载")
    print("=" * 60)
    print(f"  分子: O₂ (M={MOLECULE})")
    print(f"  同位素: {ISOTOPOLOGUES}")
    print(f"  波数范围: {NUMIN} ~ {NUMAX} cm⁻¹")
    print(f"  线强阈值: {SW_THRESHOLD:.0e} cm/molecule")
    print(f"  数据目录: {DATA_DIR}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 切换到数据目录 (hapi 在当前目录创建文件)
    original_dir = os.getcwd()
    os.chdir(DATA_DIR)

    try:
        import hapi

        # ── Step 1: 初始化 HAPI 本地数据库 ──
        print("[Step 1] 初始化 HAPI 本地数据库...")
        hapi.db_begin(str(DATA_DIR))

        # ── Step 2: 逐同位素下载 ──
        print(f"\n[Step 2] 从 HITRAN 下载 O₂ ({NUMIN}-{NUMAX} cm⁻¹)...")
        print("  (首次下载可能需要几分钟，请耐心等待...)\n")

        all_dfs = []
        for iso in ISOTOPOLOGUES:
            tname = f"O2_iso{iso}"
            print(f"  同位素 {iso}: 下载中...")
            try:
                hapi.fetch(
                    TableName=tname,
                    M=MOLECULE,
                    I=iso,
                    numin=NUMIN,
                    numax=NUMAX,
                )
            except Exception as e:
                print(f"           下载失败 (该同位素在此波段可能无数据): {e}")
                continue

            # 提取参数列
            row = {}
            for col in PARAMS:
                try:
                    row[col] = hapi.getColumn(tname, col)
                except Exception:
                    pass

            if row and len(row.get("nu", [])) > 0:
                df_iso = pd.DataFrame(row)
                n_total = len(df_iso)
                # 筛选线强
                df_iso = df_iso[df_iso["sw"] >= SW_THRESHOLD].reset_index(drop=True)
                print(f"           总谱线: {n_total}, 线强 ≥ {SW_THRESHOLD:.0e}: {len(df_iso)}")
                all_dfs.append(df_iso)
            else:
                print(f"           无数据")

        # ── Step 3: 合并所有同位素 ──
        if not all_dfs:
            print("\n[ERROR] 未下载到任何数据")
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all = df_all.sort_values("nu").reset_index(drop=True)

        print(f"\n[Step 3] 合并结果:")
        print(f"  总谱线数: {len(df_all)}")
        print(f"  波数范围: {df_all['nu'].min():.5f} ~ {df_all['nu'].max():.5f} cm⁻¹")
        print(f"  线强范围: {df_all['sw'].min():.3e} ~ {df_all['sw'].max():.3e} cm/molecule")

        # ── Step 4: 保存 CSV ──
        csv_name = f"O2_{int(NUMIN)}_{int(NUMAX)}_sw_ge_{SW_THRESHOLD:.0e}.csv"
        csv_path = DATA_DIR / csv_name
        df_all.to_csv(csv_path, index=False)
        print(f"\n[Step 4] CSV 已保存: {csv_path.name}")

        # 前 15 行预览
        print(f"\n  前 15 行:")
        pd.set_option("display.float_format", lambda x: f"{x:.6e}" if abs(x) < 0.01 or abs(x) > 1e4 else f"{x:.6f}")
        print(df_all.head(15).to_string(index=False))

        # ── 输出文件汇总 ──
        print(f"\n{'=' * 60}")
        print(f"  数据目录: {DATA_DIR}")
        print(f"{'=' * 60}")
        for f in sorted(DATA_DIR.glob("*")):
            if f.is_file():
                print(f"  {f.name:<50s} {f.stat().st_size:>10,} bytes")

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

