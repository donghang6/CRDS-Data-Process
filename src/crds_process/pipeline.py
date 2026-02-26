"""CRDS 四步处理流水线

Step 1: 衰荡时间处理       (raw → ringdown_results.csv)
Step 2: 去除标准具          (ringdown_results → tau_etalon_corrected.csv)
Step 3: MATS 单光谱拟合    (etalon_corrected → 各压力独立拟合)
Step 4: 筛选 + 多光谱联合拟合  (剔除线强离群点 → 联合拟合 → 最终参数)

目录约定:
    data/raw/{跃迁波数}/{压力}/*.txt                       ← 原始数据
    output/results/ringdown/{跃迁波数}/{压力}/              ← Step 1 输出
    output/results/etalon/{跃迁波数}/{压力}/                ← Step 2 输出
    output/results/mats/{跃迁波数}/{压力}/                  ← Step 3 输出
    output/results/mats_multi/{跃迁波数}/                   ← Step 4 输出
    output/results/final/                                   ← 最终汇总
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================================
# 默认路径 (基于项目根目录)
# ==================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_RAW_ROOT = _PROJECT_ROOT / "data" / "raw"
DEFAULT_RINGDOWN_ROOT = _PROJECT_ROOT / "output" / "results" / "ringdown"
DEFAULT_ETALON_ROOT = _PROJECT_ROOT / "output" / "results" / "etalon"
DEFAULT_MATS_ROOT = _PROJECT_ROOT / "output" / "results" / "mats"
DEFAULT_MATS_MULTI_ROOT = _PROJECT_ROOT / "output" / "results" / "mats_multi"
DEFAULT_FINAL_ROOT = _PROJECT_ROOT / "output" / "results" / "final"


# ==================================================================
# Step 1: 衰荡时间处理
# ==================================================================
def step1_ringdown(raw_root: Path | None = None,
                   result_root: Path | None = None) -> None:
    """Step 1: 原始衰荡数据 → ringdown_results.csv"""
    from crds_process.preprocessing import batch_preprocess_ringdown

    print("\n" + "=" * 60)
    print("  Step 1 / 4 — 衰荡时间处理")
    print("=" * 60)

    batch_preprocess_ringdown(
        raw_root=raw_root or DEFAULT_RAW_ROOT,
        result_root=result_root or DEFAULT_RINGDOWN_ROOT,
    )


# ==================================================================
# Step 2: 去除标准具
# ==================================================================
def step2_etalon(ringdown_root: Path | None = None,
                 etalon_root: Path | None = None) -> None:
    """Step 2: ringdown_results → tau_etalon_corrected.csv"""
    from crds_process.baseline.etalon import batch_etalon_removal

    print("\n" + "=" * 60)
    print("  Step 2 / 4 — 去除标准具效应")
    print("=" * 60)

    batch_etalon_removal(
        ringdown_root=ringdown_root or DEFAULT_RINGDOWN_ROOT,
        etalon_root=etalon_root or DEFAULT_ETALON_ROOT,
    )


# ==================================================================
# Step 3: MATS 单光谱拟合 + 汇总
# ==================================================================
def step3_mats(etalon_root: Path | None = None,
               mats_root: Path | None = None,
               lineprofile: str = "SDVP") -> None:
    """Step 3: etalon_corrected → MATS SDVP 拟合 → 线强 & 自展宽"""
    from crds_process.spectral.mats_wrapper import MATSFitter, MATSBatchProcessor

    print("\n" + "=" * 60)
    print(f"  Step 3 / 4 — MATS 光谱拟合 (线形: {lineprofile})")
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

    etalon_root = etalon_root or DEFAULT_ETALON_ROOT
    mats_root = mats_root or DEFAULT_MATS_ROOT

    processor = MATSBatchProcessor(
        etalon_root=etalon_root,
        mats_root=mats_root,
        fitter=fitter,
    )
    processor.run()

    # 汇总: 从 MATS 输出中提取线强 & 自展宽
    _collect_final_summary(mats_root)


# ==================================================================
# Step 4: 筛选线强离群点 + 多光谱联合拟合
# ==================================================================
def step4_multi_fit(etalon_root: Path | None = None,
                    mats_root: Path | None = None,
                    mats_multi_root: Path | None = None,
                    lineprofile: str = "SDVP",
                    sw_sigma: float = 2.0) -> None:
    """Step 4: 根据 Step 3 单光谱结果筛除线强离群点，
    用剩余光谱做多光谱联合拟合。

    Parameters
    ----------
    etalon_root : Path
        etalon corrected 数据目录
    mats_root : Path
        Step 3 单光谱拟合输出目录 (读取线强用于筛选)
    mats_multi_root : Path
        Step 4 多光谱联合拟合输出目录
    lineprofile : str
        线形 (默认 SDVP)
    sw_sigma : float
        线强筛选阈值，偏离中位数超过 sw_sigma 倍 MAD 的点被剔除
    """
    from crds_process.spectral.mats_wrapper import MATSFitter

    print("\n" + "=" * 60)
    print(f"  Step 4 / 4 — 筛选 + 多光谱联合拟合 (σ={sw_sigma})")
    print("=" * 60)

    etalon_root = etalon_root or DEFAULT_ETALON_ROOT
    mats_root = mats_root or DEFAULT_MATS_ROOT
    mats_multi_root = mats_multi_root or DEFAULT_MATS_MULTI_ROOT

    if not mats_root.exists():
        print(f"  [ERROR] Step 3 结果不存在: {mats_root}")
        print(f"         请先运行 Step 3")
        return

    for t_dir in sorted(mats_root.iterdir()):
        if not t_dir.is_dir() or t_dir.name.startswith("."):
            continue
        transition = t_dir.name

        # ---- 1. 收集 Step 3 各压力点的拟合线强 ----
        records: list[dict] = []
        for p_dir in sorted(t_dir.iterdir()):
            if not p_dir.is_dir() or p_dir.name.startswith("."):
                continue
            param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
            if not param_files:
                continue
            param_df = pd.read_csv(param_files[0], index_col=0)
            fitted = param_df[param_df["sw_vary"] == True]
            if fitted.empty:
                continue
            for _, row in fitted.iterrows():
                sw_real = row["sw"] * row.get("sw_scale_factor", 1.0)
                records.append({
                    "pressure": p_dir.name,
                    "sw": sw_real,
                    "sw_raw": row["sw"],
                    "gamma0_self": row["gamma0_self"],
                })

        if not records:
            print(f"\n  [{transition}] 未找到 Step 3 拟合结果，跳过")
            continue

        sw_df = pd.DataFrame(records)
        sw_values = sw_df["sw"].values
        median_sw = float(np.median(sw_values))
        mad_sw = float(np.median(np.abs(sw_values - median_sw)))
        if mad_sw < 1e-35:
            mad_sw = float(np.std(sw_values))

        # ---- 2. 基于线强的 MAD 筛选 ----
        sw_df["deviation"] = (
            np.abs(sw_df["sw"] - median_sw) / (mad_sw if mad_sw > 0 else 1)
        )
        sw_df["keep"] = sw_df["deviation"] <= sw_sigma

        print(f"\n  [{transition}] 线强筛选 (中位数={median_sw:.4e}, "
              f"MAD={mad_sw:.4e}, 阈值={sw_sigma}σ)")
        print(f"  {'压力':<10s} {'S (cm/molec)':<14s} {'偏差/MAD':<10s} {'状态':<8s}")
        print(f"  {'─' * 50}")
        for _, r in sw_df.iterrows():
            status = "✓ 保留" if r["keep"] else "✗ 剔除"
            print(f"  {r['pressure']:<10s} {r['sw']:.4e}   "
                  f"{r['deviation']:>6.2f}σ     {status}")

        kept = sw_df[sw_df["keep"]]
        removed = sw_df[~sw_df["keep"]]

        if len(removed) > 0:
            print(f"\n  剔除 {len(removed)} 个离群点: "
                  f"{', '.join(removed['pressure'].tolist())}")
        else:
            print(f"\n  无离群点，全部 {len(kept)} 个压力点保留")

        if len(kept) < 2:
            print(f"  [WARN] 保留点数 < 2，无法联合拟合，跳过")
            continue

        # ---- 3. 收集保留的 etalon CSV 路径 ----
        etalon_csvs: list[Path] = []
        labels: list[str] = []
        for _, r in kept.iterrows():
            csv_path = (etalon_root / transition / r["pressure"]
                        / "tau_etalon_corrected.csv")
            if csv_path.exists():
                etalon_csvs.append(csv_path)
                labels.append(r["pressure"])
            else:
                print(f"  [WARN] 找不到: {csv_path}")

        if len(etalon_csvs) < 2:
            print(f"  [WARN] 有效 etalon CSV < 2，跳过联合拟合")
            continue

        # ---- 4. 多光谱联合拟合 ----
        print(f"\n  开始多光谱联合拟合 ({len(etalon_csvs)} 条光谱)...")
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

        multi_out = mats_multi_root / transition
        multi_out.mkdir(parents=True, exist_ok=True)

        try:
            result = fitter.fit_multi(
                etalon_csvs=etalon_csvs,
                labels=labels,
                output_dir=multi_out,
                dataset_name=f"{transition}_multi",
            )
            if (result and result.summary_df is not None
                    and not result.summary_df.empty):
                fitter.plot_result(
                    result, multi_out,
                    title=f"Multi-spectrum Fit — {transition} "
                          f"({len(etalon_csvs)} spectra)",
                )

            sw_df.to_csv(multi_out / "sw_screening.csv", index=False)
            _save_multi_fit_summary(result, multi_out, transition, labels)

        except Exception as e:
            print(f"  [ERROR] 多光谱联合拟合失败: {e}")
            import traceback
            traceback.print_exc()


# ==================================================================
# 完整四步流水线入口
# ==================================================================
def run_pipeline(
    raw_root: Path | None = None,
    ringdown_root: Path | None = None,
    etalon_root: Path | None = None,
    mats_root: Path | None = None,
    mats_multi_root: Path | None = None,
    lineprofile: str = "SDVP",
    sw_sigma: float = 2.0,
) -> None:
    """执行完整的 CRDS 四步处理流水线

    Parameters
    ----------
    raw_root : Path, optional
        原始数据目录
    ringdown_root : Path, optional
        Step 1 输出目录
    etalon_root : Path, optional
        Step 2 输出目录
    mats_root : Path, optional
        Step 3 输出目录
    mats_multi_root : Path, optional
        Step 4 输出目录
    lineprofile : str
        MATS 线形 (默认 SDVP)
    sw_sigma : float
        Step 4 线强筛选阈值 (默认 2.0)
    """
    raw_root = raw_root or DEFAULT_RAW_ROOT
    ringdown_root = ringdown_root or DEFAULT_RINGDOWN_ROOT
    etalon_root = etalon_root or DEFAULT_ETALON_ROOT
    mats_root = mats_root or DEFAULT_MATS_ROOT
    mats_multi_root = mats_multi_root or DEFAULT_MATS_MULTI_ROOT

    t0 = time.time()

    print("=" * 60)
    print("  CRDS 完整处理流水线")
    print("  Step 1: 衰荡时间处理")
    print("  Step 2: 去除标准具效应")
    print("  Step 3: MATS 单光谱拟合 (各压力独立)")
    print("  Step 4: 筛选 + 多光谱联合拟合 (最终结果)")
    print("=" * 60)
    print(f"  线形:     {lineprofile}")
    print(f"  线强筛选: {sw_sigma}σ (Step 4)")
    print(f"  原始数据: {raw_root}")
    print("=" * 60)

    step1_ringdown(raw_root=raw_root, result_root=ringdown_root)
    step2_etalon(ringdown_root=ringdown_root, etalon_root=etalon_root)
    step3_mats(etalon_root=etalon_root, mats_root=mats_root,
               lineprofile=lineprofile)
    step4_multi_fit(etalon_root=etalon_root, mats_root=mats_root,
                    mats_multi_root=mats_multi_root,
                    lineprofile=lineprofile, sw_sigma=sw_sigma)

    elapsed = time.time() - t0
    print(f"\n{'#' * 60}")
    print(f"  全部完成! 耗时 {elapsed:.1f} s")
    print(f"{'#' * 60}")


# ==================================================================
# 内部辅助函数
# ==================================================================
def _save_multi_fit_summary(result, output_dir: Path,
                            transition: str, labels: list[str]) -> None:
    """保存多光谱联合拟合的最终统计 CSV"""
    if result is None or result.param_linelist.empty:
        return

    param_df = result.param_linelist
    fitted = param_df[param_df["sw_vary"] == True]
    if fitted.empty:
        return

    rows: list[dict] = []
    for _, row in fitted.iterrows():
        scale = row.get("sw_scale_factor", 1.0)
        rows.append({
            "transition": transition,
            "n_spectra": len(labels),
            "pressures": "+".join(labels),
            "nu_HITRAN": row["nu"],
            "sw": row["sw"] * scale,
            "sw_err": row.get("sw_err", 0) * scale,
            "gamma0_self": row["gamma0_self"],
            "gamma0_self_err": row.get("gamma0_self_err", 0),
            "n_gamma0_self": row.get("n_gamma0_self", 0),
            "n_gamma0_self_err": row.get("n_gamma0_self_err", 0),
            "SD_gamma_self": row.get("SD_gamma_self", 0),
            "SD_gamma_self_err": row.get("SD_gamma_self_err", 0),
            "delta0_self": row.get("delta0_self", 0),
            "delta0_self_err": row.get("delta0_self_err", 0),
            "SD_delta_self": row.get("SD_delta_self", 0),
            "SD_delta_self_err": row.get("SD_delta_self_err", 0),
            "residual_std": result.residual_std,
            "QF": result.qf,
        })

    final_df = pd.DataFrame(rows)
    out_path = output_dir / "multi_fit_result.csv"
    final_df.to_csv(out_path, index=False)

    final_dir = DEFAULT_FINAL_ROOT / transition
    final_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_dir / "multi_fit_result.csv", index=False)

    print(f"\n  {'#' * 60}")
    print(f"  多光谱联合拟合最终结果 ({transition})")
    print(f"  光谱数: {len(labels)}, 压力: {', '.join(labels)}")
    print(f"  {'#' * 60}")
    for _, r in final_df.iterrows():
        print(f"  ν = {r['nu_HITRAN']:.6f} cm⁻¹")
        print(f"    S          = {r['sw']:.6e} ± {r['sw_err']:.2e} cm⁻¹/(molec·cm⁻²)")
        print(f"    γ₀_self    = {r['gamma0_self']:.6f} ± {r['gamma0_self_err']:.6f} cm⁻¹/atm")
        print(f"    n_γ₀_self  = {r['n_gamma0_self']:.4f} ± {r['n_gamma0_self_err']:.4f}")
        print(f"    SD_γ_self  = {r['SD_gamma_self']:.6f} ± {r['SD_gamma_self_err']:.6f}")
        print(f"    δ₀_self    = {r['delta0_self']:.6f} ± {r['delta0_self_err']:.6f} cm⁻¹/atm")
        print(f"    SD_δ_self  = {r['SD_delta_self']:.6f} ± {r['SD_delta_self_err']:.6f}")
        print(f"    Res. σ     = {r['residual_std']:.4e}")
        print(f"    QF         = {r['QF']:.1f}")
    print(f"\n  结果已保存: {out_path}")
    print(f"  汇总目录:   {final_dir / 'multi_fit_result.csv'}")


def _collect_final_summary(mats_root: Path) -> None:
    """从 MATS 单光谱拟合结果中提取关键参数并保存汇总表"""
    DEFAULT_FINAL_ROOT.mkdir(parents=True, exist_ok=True)

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

            param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
            if not param_files:
                continue

            param_df = pd.read_csv(param_files[0], index_col=0)

            x_shift = 0.0
            baseline_files = list(p_dir.glob("*baseline_paramlist*.csv"))
            if baseline_files:
                try:
                    bl_df = pd.read_csv(baseline_files[0])
                    if "x_shift" in bl_df.columns:
                        x_shift = float(bl_df["x_shift"].iloc[0])
                except Exception:
                    pass

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
                    res_col = next(
                        (c for c in sdf.columns if "Residual" in c), None)
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

    summary_path = DEFAULT_FINAL_ROOT / "spectral_parameters_summary.csv"
    df.to_csv(summary_path, index=False)

    for transition in df["transition"].unique():
        sub = df[df["transition"] == transition]
        trans_dir = DEFAULT_FINAL_ROOT / str(transition)
        trans_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(trans_dir / "fitted_parameters.csv", index=False)

    print(f"\n{'#' * 60}")
    print(f"  单光谱拟合结果汇总 (SDVP 线形)")
    print(f"{'#' * 60}")

    x_shifts = df[["transition", "pressure_label", "x_shift"]].drop_duplicates()
    for _, xs in x_shifts.iterrows():
        print(f"\n  [{xs['transition']}/{xs['pressure_label']}] "
              f"x_shift = {xs['x_shift']:.6f} cm⁻¹ (波数计系统偏差)")

    fitted = df[df["sw_vary"] == True]
    if fitted.empty:
        fitted = df

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
    print(f"  详细目录: {DEFAULT_FINAL_ROOT}")

    _generate_fit_statistics(mats_root)


def _generate_fit_statistics(mats_root: Path) -> None:
    """从 MATS 拟合结果中提取被拟合目标线的完整参数，生成统计 CSV

    只保留 sw_vary=True 的谱线，按压力排序。

    输出: output/results/final/{跃迁}/fit_summary_statistics.csv
    """
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

            param_files = list(p_dir.glob("*Parameter_LineList*.csv"))
            if not param_files:
                continue
            param_df = pd.read_csv(param_files[0], index_col=0)

            x_shift = 0.0
            bl_files = list(p_dir.glob("*baseline_paramlist*.csv"))
            if bl_files:
                try:
                    bl = pd.read_csv(bl_files[0])
                    if "x_shift" in bl.columns:
                        x_shift = float(bl["x_shift"].iloc[0])
                except Exception:
                    pass

            residual_std = 0.0
            for csv_file in sorted(p_dir.glob("*.csv")):
                if any(kw in csv_file.name for kw in [
                    "Parameter_LineList", "baseline_paramlist",
                    "_linelist", "_spectrum",
                ]):
                    continue
                try:
                    sdf = pd.read_csv(csv_file)
                    res_col = next(
                        (c for c in sdf.columns if "Residual" in c), None)
                    if res_col:
                        residual_std = float(np.std(sdf[res_col].values))
                    break
                except Exception:
                    pass

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
        stat_df["_p"] = stat_df["pressure"].str.extract(r"(\d+)").astype(float)
        stat_df = (stat_df.sort_values("_p")
                   .drop(columns="_p").reset_index(drop=True))

        out_dir = DEFAULT_FINAL_ROOT / transition
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fit_summary_statistics.csv"
        stat_df.to_csv(out_path, index=False)
        print(f"\n  统计表: {out_path}")
