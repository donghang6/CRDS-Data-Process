"""CRDS 数据处理 — 完整三步流水线

    Step 1: 衰荡时间处理   (raw → ringdown_results.csv)
    Step 2: 去除标准具      (ringdown_results → tau_etalon_corrected.csv)
    Step 3: MATS SDVP 拟合  (etalon_corrected → 线强 / 自展宽)

运行方式:
    python main.py                      # 完整三步流水线
    python main.py --step 2 3           # 仅执行 Step 2 & 3
    python main.py --step 3             # 仅执行 Step 3 (MATS 拟合)
    python main.py --lineprofile SDVP   # 指定线形 (默认 SDVP)

目录约定:
    data/raw/{跃迁波数}/{压力}/*.txt                       ← 原始数据
    output/results/ringdown/{跃迁波数}/{压力}/              ← Step 1 输出
    output/results/etalon/{跃迁波数}/{压力}/                ← Step 2 输出
    output/results/mats/{跃迁波数}/{压力}/                  ← Step 3 输出
    output/results/final/spectral_parameters_summary.csv   ← 最终汇总
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


# ==================================================================
# 项目路径常量
# ==================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
RINGDOWN_ROOT = PROJECT_ROOT / "output" / "results" / "ringdown"
ETALON_ROOT = PROJECT_ROOT / "output" / "results" / "etalon"
MATS_ROOT = PROJECT_ROOT / "output" / "results" / "mats"
FINAL_ROOT = PROJECT_ROOT / "output" / "results" / "final"


# ==================================================================
# Step 1: 衰荡时间处理
# ==================================================================
def step1_ringdown(raw_root: Path | None = None,
                   result_root: Path | None = None) -> None:
    """Step 1: 原始衰荡数据 → ringdown_results.csv"""
    from crds_process.preprocessing import batch_preprocess_ringdown

    print("\n" + "=" * 60)
    print("  Step 1 / 3 — 衰荡时间处理")
    print("=" * 60)

    batch_preprocess_ringdown(
        raw_root=raw_root or RAW_ROOT,
        result_root=result_root or RINGDOWN_ROOT,
    )


# ==================================================================
# Step 2: 去除标准具
# ==================================================================
def step2_etalon(ringdown_root: Path | None = None,
                 etalon_root: Path | None = None) -> None:
    """Step 2: ringdown_results → tau_etalon_corrected.csv"""
    from crds_process.baseline.etalon import batch_etalon_removal

    print("\n" + "=" * 60)
    print("  Step 2 / 3 — 去除标准具效应")
    print("=" * 60)

    batch_etalon_removal(
        ringdown_root=ringdown_root or RINGDOWN_ROOT,
        etalon_root=etalon_root or ETALON_ROOT,
    )


# ==================================================================
# Step 3: MATS 拟合 + 汇总保存
# ==================================================================
def step3_mats(etalon_root: Path | None = None,
               mats_root: Path | None = None,
               lineprofile: str = "SDVP") -> None:
    """Step 3: etalon_corrected → MATS SDVP 拟合 → 线强 & 自展宽"""
    from crds_process.spectral.mats_wrapper import MATSFitter, MATSBatchProcessor

    print("\n" + "=" * 60)
    print(f"  Step 3 / 3 — MATS 光谱拟合 (线形: {lineprofile})")
    print("=" * 60)

    fitter = MATSFitter(
        molecule=7,
        isotopologue=1,
        molefraction={7: 1.0},
        diluent="self",
        lineprofile=lineprofile,
        baseline_order=1,
        fit_intensity=1e-30,
        threshold_intensity=1e-35,
    )

    etalon_root = etalon_root or ETALON_ROOT
    mats_root = mats_root or MATS_ROOT

    processor = MATSBatchProcessor(
        etalon_root=etalon_root,
        mats_root=mats_root,
        fitter=fitter,
    )
    processor.run()

    # ---- 汇总: 从 MATS 输出中提取线强 & 自展宽 ----
    _collect_final_summary(mats_root)


def _collect_final_summary(mats_root: Path) -> None:
    """从 MATS 拟合结果中提取关键参数并保存汇总表"""
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if not mats_root.exists():
        return

    for t_dir in sorted(mats_root.iterdir()):
        if not t_dir.is_dir() or t_dir.name.startswith("."):
            continue
        transition = t_dir.name
        for p_dir in sorted(t_dir.iterdir()):
            if not p_dir.is_dir() or p_dir.name.startswith("."):
                continue
            pressure_label = p_dir.name

            # 查找 MATS 输出的 Parameter_LineList CSV
            param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
            if not param_files:
                continue

            param_df = pd.read_csv(param_files[0], index_col=0)

            # 读取 baseline_paramlist 获取 x_shift
            x_shift = 0.0
            baseline_files = list(p_dir.glob("*baseline_paramlist*.csv"))
            if baseline_files:
                try:
                    bl_df = pd.read_csv(baseline_files[0])
                    if "x_shift" in bl_df.columns:
                        x_shift = float(bl_df["x_shift"].iloc[0])
                except Exception:
                    pass

            # 查找 MATS summary CSV (含残差等信息)
            # MATS 输出的 summary 文件名: {dataset_name}.csv
            # 排除已知的其他 CSV 后缀文件
            import numpy as np
            residual_std = 0.0
            for csv_file in sorted(p_dir.glob("*.csv")):
                name = csv_file.name
                if any(kw in name for kw in [
                    "Parameter_LineList", "baseline_paramlist",
                    "_linelist", "_spectrum",
                ]):
                    continue
                try:
                    sdf = pd.read_csv(csv_file)
                    res_col = next((c for c in sdf.columns if "Residual" in c), None)
                    if res_col:
                        residual_std = float(np.std(sdf[res_col].values))
                    break
                except Exception:
                    pass

            for _, row in param_df.iterrows():
                nu = row.get("nu", 0)
                sw = row.get("sw", 0)
                sw_scale = row.get("sw_scale_factor", 1.0)
                if sw * sw_scale < 1e-35:
                    continue
                all_rows.append({
                    "transition": transition,
                    "pressure_label": pressure_label,
                    "nu": nu,
                    "sw": sw * sw_scale,
                    "sw_raw": sw,
                    "sw_scale_factor": sw_scale,
                    "gamma0_self": row.get("gamma0_self", 0),
                    "gamma0_self_err": row.get("gamma0_self_err", 0),
                    "n_gamma0_self": row.get("n_gamma0_self", 0),
                    "delta0_self": row.get("delta0_self", 0),
                    "delta0_self_err": row.get("delta0_self_err", 0),
                    "SD_gamma_self": row.get("SD_gamma_self", 0),
                    "SD_gamma_self_err": row.get("SD_gamma_self_err", 0),
                    "SD_delta_self": row.get("SD_delta_self", 0),
                    "x_shift": x_shift,
                    "residual_std": residual_std,
                    "sw_vary": row.get("sw_vary", False),
                })

    if not all_rows:
        print("\n  [WARN] 未找到任何有效拟合结果，跳过汇总")
        return

    df = pd.DataFrame(all_rows)
    if "nu" in df.columns:
        df = df.sort_values("nu").reset_index(drop=True)

    # 保存汇总表
    summary_path = FINAL_ROOT / "spectral_parameters_summary.csv"
    df.to_csv(summary_path, index=False)

    # 按跃迁波数分组保存
    for transition in df["transition"].unique():
        sub = df[df["transition"] == transition]
        trans_dir = FINAL_ROOT / str(transition)
        trans_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(trans_dir / "fitted_parameters.csv", index=False)

    # 打印汇总
    print(f"\n{'#' * 60}")
    print(f"  最终拟合结果汇总 (SDVP 线形)")
    print(f"{'#' * 60}")

    # 打印 x_shift
    x_shifts = df[["transition", "pressure_label", "x_shift"]].drop_duplicates()
    for _, xs in x_shifts.iterrows():
        print(f"\n  [{xs['transition']}/{xs['pressure_label']}] "
              f"x_shift = {xs['x_shift']:.6f} cm⁻¹ (波数计系统偏差)")

    # 只显示被拟合的线 (sw_vary=True)
    fitted = df[df["sw_vary"] == True]
    if fitted.empty:
        fitted = df  # 如果没有 vary 标记则全部显示

    print(f"\n  {'─' * 100}")
    print(f"  {'跃迁':<12s} {'压力':<10s} {'ν (cm⁻¹)':<16s} "
          f"{'S (cm/molec)':<14s} "
          f"{'γ₀_self (cm⁻¹/atm)':<20s} "
          f"{'δ₀_self':<12s} "
          f"{'SD_γ':<10s} "
          f"{'Res. σ':<12s}")
    print(f"  {'─' * 100}")

    for _, row in fitted.iterrows():
        gamma_str = f"{row['gamma0_self']:.6f}"
        if row.get('gamma0_self_err', 0) > 0:
            gamma_str += f" ± {row['gamma0_self_err']:.6f}"
        print(
            f"  {row['transition']:<12s} {row['pressure_label']:<10s} "
            f"{row['nu']:>14.6f}  "
            f"{row['sw']:>12.4e}  "
            f"{gamma_str:<20s} "
            f"{row.get('delta0_self', 0):>10.6f}  "
            f"{row.get('SD_gamma_self', 0):>8.4f}  "
            f"{row['residual_std']:>10.4e}"
        )

    print(f"  {'─' * 100}")
    print(f"\n  汇总表: {summary_path}")
    print(f"  详细目录: {FINAL_ROOT}")

    # ---- 按跃迁生成统计 CSV (只含被拟合的目标线, 按压力排序) ----
    _generate_fit_statistics(mats_root)


def _generate_fit_statistics(mats_root: Path) -> None:
    """从 MATS 拟合结果中提取被拟合目标线的完整参数，生成统计 CSV

    只保留 sw_vary=True 的谱线（实际被拟合的目标线），
    按压力排序，包含参数值及其不确定度。

    输出: output/results/final/{跃迁}/fit_summary_statistics.csv
    """
    import numpy as np

    if not mats_root.exists():
        return

    for t_dir in sorted(mats_root.iterdir()):
        if not t_dir.is_dir() or t_dir.name.startswith("."):
            continue
        transition = t_dir.name
        rows: list[dict] = []

        for p_dir in sorted(t_dir.iterdir()):
            if not p_dir.is_dir() or p_dir.name.startswith("."):
                continue
            pressure_label = p_dir.name

            # Parameter_LineList
            param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
            if not param_files:
                continue
            param_df = pd.read_csv(param_files[0], index_col=0)

            # baseline_paramlist → x_shift
            x_shift = 0.0
            bl_files = list(p_dir.glob("*baseline_paramlist*.csv"))
            if bl_files:
                try:
                    bl = pd.read_csv(bl_files[0])
                    if "x_shift" in bl.columns:
                        x_shift = float(bl["x_shift"].iloc[0])
                except Exception:
                    pass

            # summary → residual_std
            residual_std = 0.0
            for csv_file in sorted(p_dir.glob("*.csv")):
                if any(kw in csv_file.name for kw in [
                    "Parameter_LineList", "baseline_paramlist",
                    "_linelist", "_spectrum",
                ]):
                    continue
                try:
                    sdf = pd.read_csv(csv_file)
                    res_col = next((c for c in sdf.columns if "Residual" in c), None)
                    if res_col:
                        residual_std = float(np.std(sdf[res_col].values))
                    break
                except Exception:
                    pass

            # 只取被拟合的线
            fitted = param_df[param_df["sw_vary"] == True]
            for _, row in fitted.iterrows():
                scale = row.get("sw_scale_factor", 1.0)
                rows.append({
                    "pressure": pressure_label,
                    "nu_HITRAN": row["nu"],
                    "sw": row["sw"] * scale,
                    "sw_err": row.get("sw_err", 0) * scale,
                    "gamma0_self": row["gamma0_self"],
                    "gamma0_self_err": row.get("gamma0_self_err", 0),
                    "SD_gamma_self": row.get("SD_gamma_self", 0),
                    "SD_gamma_self_err": row.get("SD_gamma_self_err", 0),
                    "delta0_self": row.get("delta0_self", 0),
                    "delta0_self_err": row.get("delta0_self_err", 0),
                    "SD_delta_self": row.get("SD_delta_self", 0),
                    "SD_delta_self_err": row.get("SD_delta_self_err", 0),
                    "x_shift": x_shift,
                    "residual_std": residual_std,
                })

        if not rows:
            continue

        stat_df = pd.DataFrame(rows)
        # 按压力数值排序
        stat_df["_p"] = stat_df["pressure"].str.extract(r"(\d+)").astype(float)
        stat_df = stat_df.sort_values("_p").drop(columns="_p").reset_index(drop=True)

        out_dir = FINAL_ROOT / transition
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fit_summary_statistics.csv"
        stat_df.to_csv(out_path, index=False)
        print(f"\n  统计表: {out_path}")


# ==================================================================
# 主入口
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CRDS 完整三步流水线: 衰荡时间 → 去除标准具 → MATS 拟合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                      # 完整三步
  python main.py --step 2 3           # 仅 Step 2 & 3
  python main.py --step 3             # 仅 Step 3
  python main.py --lineprofile VP     # 使用 Voigt 线形
        """,
    )
    parser.add_argument(
        "--step", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="执行哪些步骤 (默认: 1 2 3 全部执行)",
    )
    parser.add_argument(
        "--lineprofile", default="SDVP",
        choices=["VP", "SDVP", "NGP", "SDNGP", "HTP"],
        help="MATS 线形 (默认: SDVP)",
    )
    parser.add_argument(
        "--raw-dir", default=None,
        help=f"原始数据目录 (默认: {RAW_ROOT})",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="输出根目录 (默认: output/results/)",
    )
    args = parser.parse_args()

    steps = sorted(set(args.step))

    # 自定义路径
    raw_root = Path(args.raw_dir) if args.raw_dir else RAW_ROOT
    if args.output_dir:
        out = Path(args.output_dir)
        ringdown_root = out / "ringdown"
        etalon_root = out / "etalon"
        mats_root = out / "mats"
    else:
        ringdown_root = RINGDOWN_ROOT
        etalon_root = ETALON_ROOT
        mats_root = MATS_ROOT

    t0 = time.time()

    print("=" * 60)
    print("  CRDS 完整处理流水线")
    print("  Step 1: 衰荡时间处理")
    print("  Step 2: 去除标准具效应")
    print("  Step 3: MATS 光谱拟合 (SDVP → 线强 & 自展宽)")
    print("=" * 60)
    print(f"  执行步骤: {steps}")
    print(f"  线形:     {args.lineprofile}")
    print(f"  原始数据: {raw_root}")
    print("=" * 60)

    # ---- Step 1 ----
    if 1 in steps:
        step1_ringdown(raw_root=raw_root, result_root=ringdown_root)

    # ---- Step 2 ----
    if 2 in steps:
        step2_etalon(ringdown_root=ringdown_root, etalon_root=etalon_root)

    # ---- Step 3 ----
    if 3 in steps:
        step3_mats(etalon_root=etalon_root, mats_root=mats_root,
                   lineprofile=args.lineprofile)

    elapsed = time.time() - t0
    print(f"\n{'#' * 60}")
    print(f"  全部完成! 耗时 {elapsed:.1f} s")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()

