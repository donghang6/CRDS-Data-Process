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

用法:
    # 默认参数
    pipeline = CRDSPipeline()
    pipeline.run()

    # 自定义参数
    pipeline = CRDSPipeline(
        raw_root="data/raw",
        lineprofile="SDVP",
        sw_sigma=2.0,
    )
    pipeline.run()

    # 单独执行某一步
    pipeline.step1_ringdown()
    pipeline.step2_etalon()
    pipeline.step3_mats()
    pipeline.step4_multi_fit()
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from crds_process.log import logger, setup_logging


# ==================================================================
# 默认路径 (基于项目根目录)
# ==================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_PATHS = {
    "raw": _PROJECT_ROOT / "data" / "raw",
    "ringdown": _PROJECT_ROOT / "output" / "results" / "ringdown",
    "etalon": _PROJECT_ROOT / "output" / "results" / "etalon",
    "mats": _PROJECT_ROOT / "output" / "results" / "mats",
    "mats_multi": _PROJECT_ROOT / "output" / "results" / "mats_multi",
    "final": _PROJECT_ROOT / "output" / "results" / "final",
}


class CRDSPipeline:
    """CRDS 四步处理流水线

    Parameters
    ----------
    raw_root : Path or str, optional
        原始数据目录
    ringdown_root : Path or str, optional
        Step 1 输出目录
    etalon_root : Path or str, optional
        Step 2 输出目录
    mats_root : Path or str, optional
        Step 3 输出目录
    mats_multi_root : Path or str, optional
        Step 4 输出目录
    final_root : Path or str, optional
        最终汇总输出目录
    lineprofile : str
        MATS 线形 (默认 "SDVP")
    sw_sigma : float
        Step 4 线强筛选阈值 (默认 2.0)
    molecule : int
        HITRAN 分子编号 (默认 7 = O₂)
    isotopologue : int
        HITRAN 同位素编号 (默认 1)
    molefraction : dict
        摩尔分数 (默认 {7: 1.0})
    diluent : str
        稀释气体 (默认 "O2")
    baseline_order : int
        基线多项式阶数 (默认 1)
    fit_intensity : float
        拟合线强阈值 (默认 1e-30)
    threshold_intensity : float
        线强筛选阈值 (默认 1e-35)
    """

    def __init__(
        self,
        raw_root: Path | str | None = None,
        ringdown_root: Path | str | None = None,
        etalon_root: Path | str | None = None,
        mats_root: Path | str | None = None,
        mats_multi_root: Path | str | None = None,
        final_root: Path | str | None = None,
        lineprofile: str = "SDVP",
        sw_sigma: float = 2.0,
        molecule: int = 7,
        isotopologue: int = 1,
        molefraction: dict | None = None,
        diluent: str = "O2",
        baseline_order: int = 1,
        fit_intensity: float = 1e-30,
        threshold_intensity: float = 1e-35,
    ):
        # ── 路径 ──
        self.raw_root = Path(raw_root) if raw_root else _DEFAULT_PATHS["raw"]
        self.ringdown_root = Path(ringdown_root) if ringdown_root else _DEFAULT_PATHS["ringdown"]
        self.etalon_root = Path(etalon_root) if etalon_root else _DEFAULT_PATHS["etalon"]
        self.mats_root = Path(mats_root) if mats_root else _DEFAULT_PATHS["mats"]
        self.mats_multi_root = Path(mats_multi_root) if mats_multi_root else _DEFAULT_PATHS["mats_multi"]
        self.final_root = Path(final_root) if final_root else _DEFAULT_PATHS["final"]

        # ── MATS 拟合参数 ──
        self.lineprofile = lineprofile
        self.sw_sigma = sw_sigma
        self.molecule = molecule
        self.isotopologue = isotopologue
        self.molefraction = molefraction or {7: 1.0}
        self.diluent = diluent
        self.baseline_order = baseline_order
        self.fit_intensity = fit_intensity
        self.threshold_intensity = threshold_intensity

    # ==============================================================
    # 创建 MATSFitter 实例 (内部复用)
    # ==============================================================
    def _create_fitter(self):
        """创建 MATSFitter 实例"""
        from crds_process.spectral.mats_wrapper import MATSFitter
        return MATSFitter(
            molecule=self.molecule,
            isotopologue=self.isotopologue,
            molefraction=self.molefraction,
            diluent=self.diluent,
            lineprofile=self.lineprofile,
            baseline_order=self.baseline_order,
            fit_intensity=self.fit_intensity,
            threshold_intensity=self.threshold_intensity,
        )

    # ==============================================================
    # Step 1: 衰荡时间处理
    # ==============================================================
    def step1_ringdown(self) -> None:
        """Step 1: 原始衰荡数据 → ringdown_results.csv"""
        from crds_process.preprocessing import batch_preprocess_ringdown

        logger.info("\n" + "=" * 60)
        logger.info("  Step 1 / 4 — 衰荡时间处理")
        logger.info("=" * 60)

        batch_preprocess_ringdown(
            raw_root=self.raw_root,
            result_root=self.ringdown_root,
        )

    # ==============================================================
    # Step 2: 去除标准具
    # ==============================================================
    def step2_etalon(self) -> None:
        """Step 2: ringdown_results → tau_etalon_corrected.csv"""
        from crds_process.baseline.etalon import batch_etalon_removal

        logger.info("\n" + "=" * 60)
        logger.info("  Step 2 / 4 — 去除标准具效应")
        logger.info("=" * 60)

        batch_etalon_removal(
            ringdown_root=self.ringdown_root,
            etalon_root=self.etalon_root,
        )

    # ==============================================================
    # Step 3: MATS 单光谱拟合 + 汇总
    # ==============================================================
    def step3_mats(self) -> None:
        """Step 3: etalon_corrected → MATS SDVP 拟合 → 线强 & 自展宽"""
        from crds_process.spectral.mats_wrapper import MATSBatchProcessor

        logger.info("\n" + "=" * 60)
        logger.info(f"  Step 3 / 4 — MATS 光谱拟合 (线形: {self.lineprofile})")
        logger.info("=" * 60)

        fitter = self._create_fitter()
        processor = MATSBatchProcessor(
            etalon_root=self.etalon_root,
            mats_root=self.mats_root,
            fitter=fitter,
        )
        processor.run()

        # 汇总: 从 MATS 输出中提取线强 & 自展宽
        self._collect_final_summary()

    # ==============================================================
    # Step 4: 筛选线强离群点 + 多光谱联合拟合
    # ==============================================================
    def step4_multi_fit(self) -> None:
        """Step 4: 根据 Step 3 单光谱结果筛除线强离群点，
        用剩余光谱做多光谱联合拟合。
        """
        if not self.mats_root.exists():
            logger.error(f"   Step 3 结果不存在: {self.mats_root}")
            logger.error(f"         请先运行 Step 3")
            return

        logger.info("\n" + "=" * 60)
        logger.info(f"  Step 4 / 4 — 筛选 + 多光谱联合拟合 (σ={self.sw_sigma})")
        logger.info("=" * 60)

        for t_dir in sorted(self.mats_root.iterdir()):
            if not t_dir.is_dir() or t_dir.name.startswith("."):
                continue
            self._process_transition_multi_fit(t_dir.name)

    def _process_transition_multi_fit(self, transition: str) -> None:
        """对单个跃迁执行线强筛选 + 多光谱联合拟合"""
        t_dir = self.mats_root / transition

        # ---- 1. 收集 Step 3 各压力点的拟合线强 ----
        records = self._collect_sw_records(t_dir)
        if not records:
            logger.info(f"\n  [{transition}] 未找到 Step 3 拟合结果，跳过")
            return

        # ---- 2. MAD 筛选 ----
        sw_df = self._screen_sw(records, transition)
        kept = sw_df[sw_df["keep"]]
        if len(kept) < 2:
            logger.warning(f"   保留点数 < 2，无法联合拟合，跳过")
            return

        # ---- 3. 收集保留的 etalon CSV ----
        etalon_csvs, labels = self._collect_etalon_csvs(transition, kept)
        if len(etalon_csvs) < 2:
            logger.warning(f"   有效 etalon CSV < 2，跳过联合拟合")
            return

        # ---- 4. 多光谱联合拟合 ----
        logger.info(f"\n  开始多光谱联合拟合 ({len(etalon_csvs)} 条光谱)...")
        fitter = self._create_fitter()
        multi_out = self.mats_multi_root / transition
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
            self._save_multi_fit_summary(result, multi_out, transition, labels)

        except Exception as e:
            logger.error(f"  多光谱联合拟合失败: {e}")
            logger.exception("  详细错误信息:")

    # ==============================================================
    # 完整四步流水线
    # ==============================================================
    def run(self) -> None:
        """执行完整的 CRDS 四步处理流水线"""
        log_path = setup_logging()
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("  CRDS 完整处理流水线")
        logger.info("  Step 1: 衰荡时间处理")
        logger.info("  Step 2: 去除标准具效应")
        logger.info("  Step 3: MATS 单光谱拟合 (各压力独立)")
        logger.info("  Step 4: 筛选 + 多光谱联合拟合 (最终结果)")
        logger.info("=" * 60)
        logger.info(f"  线形:     {self.lineprofile}")
        logger.info(f"  线强筛选: {self.sw_sigma}σ (Step 4)")
        logger.info(f"  原始数据: {self.raw_root}")
        logger.info(f"  日志文件: {log_path}")
        logger.info("=" * 60)

        self.step1_ringdown()
        self.step2_etalon()
        self.step3_mats()
        self.step4_multi_fit()

        elapsed = time.time() - t0
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  全部完成! 耗时 {elapsed:.1f} s")
        logger.info(f"{'#' * 60}")

    # ==============================================================
    # Step 4 辅助方法
    # ==============================================================
    @staticmethod
    def _collect_sw_records(t_dir: Path) -> list[dict]:
        """收集某个跃迁下各压力点的拟合线强"""
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
                    "gamma0_O2": row["gamma0_O2"],
                })
        return records

    def _screen_sw(self, records: list[dict], transition: str) -> pd.DataFrame:
        """基于 MAD 筛选线强离群点"""
        sw_df = pd.DataFrame(records)
        sw_values = sw_df["sw"].values
        median_sw = float(np.median(sw_values))
        mad_sw = float(np.median(np.abs(sw_values - median_sw)))
        if mad_sw < 1e-35:
            mad_sw = float(np.std(sw_values))

        sw_df["deviation"] = (
            np.abs(sw_df["sw"] - median_sw) / (mad_sw if mad_sw > 0 else 1)
        )
        sw_df["keep"] = sw_df["deviation"] <= self.sw_sigma

        # 打印筛选结果
        logger.info(f"\n  [{transition}] 线强筛选 (中位数={median_sw:.4e}, "
              f"MAD={mad_sw:.4e}, 阈值={self.sw_sigma}σ)")
        logger.info(f"  {'压力':<10s} {'S (cm/molec)':<14s} {'偏差/MAD':<10s} {'状态':<8s}")
        logger.info(f"  {'─' * 50}")
        for _, r in sw_df.iterrows():
            status = "✓ 保留" if r["keep"] else "✗ 剔除"
            logger.info(f"  {r['pressure']:<10s} {r['sw']:.4e}   "
                  f"{r['deviation']:>6.2f}σ     {status}")

        kept = sw_df[sw_df["keep"]]
        removed = sw_df[~sw_df["keep"]]
        if len(removed) > 0:
            logger.info(f"\n  剔除 {len(removed)} 个离群点: "
                  f"{', '.join(removed['pressure'].tolist())}")
        else:
            logger.info(f"\n  无离群点，全部 {len(kept)} 个压力点保留")

        return sw_df

    def _collect_etalon_csvs(self, transition: str,
                             kept: pd.DataFrame) -> tuple[list[Path], list[str]]:
        """收集保留压力点的 etalon CSV 路径"""
        etalon_csvs: list[Path] = []
        labels: list[str] = []
        for _, r in kept.iterrows():
            csv_path = (self.etalon_root / transition / r["pressure"]
                        / "tau_etalon_corrected.csv")
            if csv_path.exists():
                etalon_csvs.append(csv_path)
                labels.append(r["pressure"])
            else:
                logger.warning(f"   找不到: {csv_path}")
        return etalon_csvs, labels

    # ==============================================================
    # 结果汇总方法
    # ==============================================================
    def _save_multi_fit_summary(self, result, output_dir: Path,
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
                "gamma0_O2": row["gamma0_O2"],
                "gamma0_O2_err": row.get("gamma0_O2_err", 0),
                "n_gamma0_O2": row.get("n_gamma0_O2", 0),
                "n_gamma0_O2_err": row.get("n_gamma0_O2_err", 0),
                "SD_gamma_O2": row.get("SD_gamma_O2", 0),
                "SD_gamma_O2_err": row.get("SD_gamma_O2_err", 0),
                "delta0_O2": row.get("delta0_O2", 0),
                "delta0_O2_err": row.get("delta0_O2_err", 0),
                "SD_delta_O2": row.get("SD_delta_O2", 0),
                "SD_delta_O2_err": row.get("SD_delta_O2_err", 0),
                "residual_std": result.residual_std,
                "QF": result.qf,
            })

        final_df = pd.DataFrame(rows)
        out_path = output_dir / "multi_fit_result.csv"
        final_df.to_csv(out_path, index=False)

        final_dir = self.final_root / transition
        final_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(final_dir / "multi_fit_result.csv", index=False)

        logger.info(f"\n  {'#' * 60}")
        logger.info(f"  多光谱联合拟合最终结果 ({transition})")
        logger.info(f"  光谱数: {len(labels)}, 压力: {', '.join(labels)}")
        logger.info(f"  {'#' * 60}")
        for _, r in final_df.iterrows():
            logger.info(f"  ν = {r['nu_HITRAN']:.6f} cm⁻¹")
            logger.info(f"    S          = {r['sw']:.6e} ± {r['sw_err']:.2e} cm⁻¹/(molec·cm⁻²)")
            logger.info(f"    γ₀_O2     = {r['gamma0_O2']:.6f} ± {r['gamma0_O2_err']:.6f} cm⁻¹/atm")
            logger.info(f"    n_γ₀_O2   = {r['n_gamma0_O2']:.4f} ± {r['n_gamma0_O2_err']:.4f}")
            logger.info(f"    SD_γ_O2   = {r['SD_gamma_O2']:.6f} ± {r['SD_gamma_O2_err']:.6f}")
            logger.info(f"    δ₀_O2     = {r['delta0_O2']:.6f} ± {r['delta0_O2_err']:.6f} cm⁻¹/atm")
            logger.info(f"    SD_δ_O2   = {r['SD_delta_O2']:.6f} ± {r['SD_delta_O2_err']:.6f}")
            logger.info(f"    Res. σ     = {r['residual_std']:.4e}")
            logger.info(f"    QF         = {r['QF']:.1f}")
        logger.info(f"\n  结果已保存: {out_path}")
        logger.info(f"  汇总目录:   {final_dir / 'multi_fit_result.csv'}")

    def _collect_final_summary(self) -> None:
        """从 MATS 单光谱拟合结果中提取关键参数并保存汇总表"""
        self.final_root.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []

        if not self.mats_root.exists():
            return

        for t_dir in sorted(self.mats_root.iterdir()):
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

                x_shift = self._read_x_shift(p_dir)
                residual_std = self._read_residual_std(p_dir)

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
                        "gamma0_O2": row.get("gamma0_O2", 0),
                        "gamma0_O2_err": row.get("gamma0_O2_err", 0),
                        "n_gamma0_O2": row.get("n_gamma0_O2", 0),
                        "delta0_O2": row.get("delta0_O2", 0),
                        "delta0_O2_err": row.get("delta0_O2_err", 0),
                        "SD_gamma_O2": row.get("SD_gamma_O2", 0),
                        "SD_gamma_O2_err": row.get("SD_gamma_O2_err", 0),
                        "SD_delta_O2": row.get("SD_delta_O2", 0),
                        "x_shift": x_shift,
                        "residual_std": residual_std,
                        "sw_vary": row.get("sw_vary", False),
                    })

        if not all_rows:
            logger.warning("\n  未找到任何有效拟合结果，跳过汇总")
            return

        df = pd.DataFrame(all_rows)
        if "nu" in df.columns:
            df = df.sort_values("nu").reset_index(drop=True)

        summary_path = self.final_root / "spectral_parameters_summary.csv"
        df.to_csv(summary_path, index=False)

        for transition in df["transition"].unique():
            sub = df[df["transition"] == transition]
            trans_dir = self.final_root / str(transition)
            trans_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(trans_dir / "fitted_parameters.csv", index=False)

        self._print_summary_table(df, summary_path)
        self._generate_fit_statistics()

    def _print_summary_table(self, df: pd.DataFrame, summary_path: Path) -> None:
        """打印单光谱拟合汇总表"""
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  单光谱拟合结果汇总 (SDVP 线形)")
        logger.info(f"{'#' * 60}")

        x_shifts = df[["transition", "pressure_label", "x_shift"]].drop_duplicates()
        for _, xs in x_shifts.iterrows():
            logger.info(f"\n  [{xs['transition']}/{xs['pressure_label']}] "
                  f"x_shift = {xs['x_shift']:.6f} cm⁻¹ (波数计系统偏差)")

        fitted = df[df["sw_vary"] == True]
        if fitted.empty:
            fitted = df

        logger.info(f"\n  {'─' * 100}")
        logger.info(f"  {'跃迁':<12s} {'压力':<10s} {'ν (cm⁻¹)':<16s} "
              f"{'S (cm/molec)':<14s} "
              f"{'γ₀_O2 (cm⁻¹/atm)':<20s} "
              f"{'δ₀_O2':<12s} "
              f"{'SD_γ':<10s} "
              f"{'Res. σ':<12s}")
        logger.info(f"  {'─' * 100}")

        for _, row in fitted.iterrows():
            gamma_str = f"{row['gamma0_O2']:.6f}"
            if row.get('gamma0_O2_err', 0) > 0:
                gamma_str += f" ± {row['gamma0_O2_err']:.6f}"
            logger.info(
                f"  {row['transition']:<12s} {row['pressure_label']:<10s} "
                f"{row['nu']:>14.6f}  "
                f"{row['sw']:>12.4e}  "
                f"{gamma_str:<20s} "
                f"{row.get('delta0_O2', 0):>10.6f}  "
                f"{row.get('SD_gamma_O2', 0):>8.4f}  "
                f"{row['residual_std']:>10.4e}"
            )

        logger.info(f"  {'─' * 100}")
        logger.info(f"\n  汇总表: {summary_path}")
        logger.info(f"  详细目录: {self.final_root}")

    def _generate_fit_statistics(self) -> None:
        """从 MATS 拟合结果中提取被拟合目标线的完整参数，生成统计 CSV

        只保留 sw_vary=True 的谱线，按压力排序。
        输出: output/results/final/{跃迁}/fit_summary_statistics.csv
        """
        if not self.mats_root.exists():
            return

        for t_dir in sorted(self.mats_root.iterdir()):
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

                x_shift = self._read_x_shift(p_dir)
                residual_std = self._read_residual_std(p_dir)

                fitted = param_df[param_df["sw_vary"] == True]
                for _, row in fitted.iterrows():
                    scale = row.get("sw_scale_factor", 1.0)
                    rows.append({
                        "pressure": pressure_label,
                        "nu_HITRAN": row["nu"],
                        "sw": row["sw"] * scale,
                        "sw_err": row.get("sw_err", 0) * scale,
                        "gamma0_O2": row["gamma0_O2"],
                        "gamma0_O2_err": row.get("gamma0_O2_err", 0),
                        "SD_gamma_O2": row.get("SD_gamma_O2", 0),
                        "SD_gamma_O2_err": row.get("SD_gamma_O2_err", 0),
                        "delta0_O2": row.get("delta0_O2", 0),
                        "delta0_O2_err": row.get("delta0_O2_err", 0),
                        "SD_delta_O2": row.get("SD_delta_O2", 0),
                        "SD_delta_O2_err": row.get("SD_delta_O2_err", 0),
                        "x_shift": x_shift,
                        "residual_std": residual_std,
                    })

            if not rows:
                continue

            stat_df = pd.DataFrame(rows)
            stat_df["_p"] = stat_df["pressure"].str.extract(r"(\d+)").astype(float)
            stat_df = (stat_df.sort_values("_p")
                       .drop(columns="_p").reset_index(drop=True))

            out_dir = self.final_root / transition
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "fit_summary_statistics.csv"
            stat_df.to_csv(out_path, index=False)
            logger.info(f"\n  统计表: {out_path}")

    # ==============================================================
    # 通用 I/O 辅助方法
    # ==============================================================
    @staticmethod
    def _read_x_shift(p_dir: Path) -> float:
        """从 baseline_paramlist CSV 中读取 x_shift"""
        baseline_files = list(p_dir.glob("*baseline_paramlist*.csv"))
        if not baseline_files:
            return 0.0
        try:
            bl_df = pd.read_csv(baseline_files[0])
            if "x_shift" in bl_df.columns:
                return float(bl_df["x_shift"].iloc[0])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _read_residual_std(p_dir: Path) -> float:
        """从 MATS 输出 CSV 中读取残差标准差"""
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
                    return float(np.std(sdf[res_col].values))
            except Exception:
                pass
        return 0.0


# ==================================================================
# 便捷函数 (向后兼容)
# ==================================================================
def run_pipeline(**kwargs) -> None:
    """便捷函数: 创建 CRDSPipeline 并执行完整流水线

    支持的关键字参数同 CRDSPipeline.__init__
    """
    pipeline = CRDSPipeline(**kwargs)
    pipeline.run()

