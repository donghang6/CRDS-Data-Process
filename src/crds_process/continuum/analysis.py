"""Continuum absorption analysis from CRDS Step 1 tau spectra.

This module starts from the same ring-down summary used by line absorption
processing. It converts ring-down times into a cavity-loss spectrum,

    loss = 1e12 / c / tau_us

where ``loss`` is in ppm/cm and ``tau_us`` is in microseconds. If an
empty-cavity/reference tau is supplied, the reference loss is subtracted
to produce the absorption coefficient ``alpha_ppm_per_cm``.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crds_process.log import logger

_CACHE_ROOT = Path(tempfile.gettempdir()) / "crds-data-process-cache"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib  # noqa: E402

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RINGDOWN_ROOT = PROJECT_ROOT / "output" / "results" / "ringdown"
CONTINUUM_ROOT = PROJECT_ROOT / "output" / "results" / "continuum"
HITRAN_DIR = PROJECT_ROOT / "data" / "hitran"
RINGDOWN_CSV_NAME = "ringdown_results.csv"

C_CM_PER_S = 2.99792458e10
TAU_US_TO_PPM_PER_CM = 1e12 / C_CM_PER_S
HITRAN_O2_STEP_CM1 = 0.002
HITRAN_O2_MASK_RATIO = 0.01
HITRAN_O2_MASK_MARGIN_CM1 = 0.05

WAVENUMBER_COLUMNS = ("wavenumber", "Wavenumber", "nu", "Total Frequency /MHz")
TAU_COLUMNS = ("tau_mean", "Mean tau/us")
TAU_STATS_COLUMNS = ("tau_sem", "tau_std", "Tau_stats")
PRESSURE_COLUMNS = ("pressure", "pressure_mean", "Cavity Pressure /Torr")
TEMPERATURE_COLUMNS = ("temperature", "temperature_mean", "Cavity Temperature Side 2 /C")


@dataclass
class ContinuumResult:
    """Result for one gas/transition/pressure dataset."""

    gas_type: str
    transition: str
    pressure: str
    source_csv: Path
    spectrum_csv: Path
    plot_path: Path
    fit_csv: Path
    summary: dict


def discover_continuum_tasks(
    input_root: Path | str | None = None,
    input_csv_name: str = RINGDOWN_CSV_NAME,
) -> list[tuple[str, str, str, Path]]:
    """Discover Step 1 ringdown spectra for continuum analysis."""
    root = Path(input_root) if input_root else RINGDOWN_ROOT
    tasks: list[tuple[str, str, str, Path]] = []
    if not root.exists():
        return tasks

    for gas_dir in sorted(root.iterdir()):
        if not gas_dir.is_dir() or gas_dir.name.startswith("."):
            continue
        for transition_dir in sorted(gas_dir.iterdir()):
            if not transition_dir.is_dir() or transition_dir.name.startswith("."):
                continue
            for pressure_dir in sorted(transition_dir.iterdir()):
                if not pressure_dir.is_dir() or pressure_dir.name.startswith("."):
                    continue
                csv_path = pressure_dir / input_csv_name
                if csv_path.exists():
                    tasks.append((gas_dir.name, transition_dir.name, pressure_dir.name, csv_path))
    return tasks


class ContinuumBatchProcessor:
    """Batch processor for continuum absorption spectra."""

    def __init__(
        self,
        input_root: Path | str | None = None,
        input_csv_name: str = RINGDOWN_CSV_NAME,
        output_root: Path | str | None = None,
        reference_csv: Path | str | None = None,
        tau0_us: float | None = None,
        window: tuple[float, float] | None = None,
        tau_col: str | None = None,
        tau_stats_col: str | None = None,
        wavenumber_col: str | None = None,
        min_points: int = 5,
        fit_window_cm1: float = 20.0,
        fit_step_cm1: float = 5.0,
        fit_order: int = 2,
        fit_sigma: float = 4.0,
        fit_smooth_cm1: float = 0.0,
        fit_mode: str = "auto",
    ):
        if reference_csv and tau0_us is not None:
            raise ValueError("reference_csv and tau0_us are mutually exclusive")

        self.input_root = Path(input_root) if input_root else RINGDOWN_ROOT
        self.input_csv_name = input_csv_name
        self.output_root = Path(output_root) if output_root else CONTINUUM_ROOT
        self.reference_csv = Path(reference_csv) if reference_csv else None
        self.tau0_us = tau0_us
        self.window = _normalize_window(window)
        self.tau_col = tau_col
        self.tau_stats_col = tau_stats_col
        self.wavenumber_col = wavenumber_col
        self.min_points = min_points
        self.fit_window_cm1 = float(fit_window_cm1)
        self.fit_step_cm1 = float(fit_step_cm1)
        self.fit_order = int(fit_order)
        self.fit_sigma = float(fit_sigma)
        self.fit_smooth_cm1 = float(fit_smooth_cm1)
        self.fit_mode = _normalize_fit_mode(fit_mode)
        self._reference = self._load_reference()

    def discover(self) -> list[tuple[str, str, str, Path]]:
        """Discover available input spectra."""
        return discover_continuum_tasks(self.input_root, self.input_csv_name)

    def run(
        self,
        tasks: list[tuple[str, str, str, Path]] | None = None,
    ) -> pd.DataFrame:
        """Run continuum analysis and return the combined summary table."""
        tasks = tasks if tasks is not None else self.discover()
        if not tasks:
            logger.error(f"未在 {self.input_root} 下找到 {self.input_csv_name}")
            return pd.DataFrame()

        self.output_root.mkdir(parents=True, exist_ok=True)
        results: list[ContinuumResult] = []
        logger.info("\n" + "=" * 60)
        logger.info("  Continuum absorption analysis")
        logger.info("=" * 60)
        logger.info(f"  Input: {self.input_root}")
        logger.info(f"  Input CSV: {self.input_csv_name}")
        logger.info(f"  Output: {self.output_root}")
        logger.info(f"  Datasets: {len(tasks)}")
        logger.info(f"  Reference: {self._reference_label()}")
        if self.fit_mode == "o2-hitran":
            logger.info(
                "  Step 2: O2 HITRAN subtraction only; "
                f"HITRAN grid step={HITRAN_O2_STEP_CM1:g} cm-1"
            )
        elif self.fit_mode == "o2":
            logger.info(
                "  Step 2 fit: O2 HITRAN-masked CIA baseline fit; "
                f"window={self.fit_window_cm1:g} cm-1, "
                f"step={self.fit_step_cm1:g} cm-1, "
                f"order={self.fit_order}, sigma={self.fit_sigma:g}"
                f", smooth={self.fit_smooth_cm1:g} cm-1, "
                f"mask_ratio={HITRAN_O2_MASK_RATIO:g}, "
                f"mask_margin={HITRAN_O2_MASK_MARGIN_CM1:g} cm-1"
            )
        else:
            logger.info(
                "  Step 2 fit: "
                f"window={self.fit_window_cm1:g} cm-1, "
                f"step={self.fit_step_cm1:g} cm-1, "
                f"order={self.fit_order}, sigma={self.fit_sigma:g}"
                f", smooth={self.fit_smooth_cm1:g} cm-1, "
                f"mode={self.fit_mode}"
            )

        for gas_type, transition, pressure, csv_path in tasks:
            output_dir = self.output_root / gas_type / transition / pressure
            try:
                result = self.process_file(csv_path, output_dir, gas_type, transition, pressure)
            except Exception as exc:
                logger.error(f"  [{gas_type}/{transition}/{pressure}] failed: {exc}")
                continue
            results.append(result)
            mean_alpha = result.summary.get("alpha_mean_ppm_per_cm")
            if pd.notna(mean_alpha):
                logger.info(f"  [{gas_type}/{transition}/{pressure}] alpha_mean={mean_alpha:.6g} ppm/cm")
            else:
                logger.info(f"  [{gas_type}/{transition}/{pressure}] loss_mean="
                            f"{result.summary['loss_mean_ppm_per_cm']:.6g} ppm/cm")

        summary_df = pd.DataFrame([r.summary for r in results])
        if not summary_df.empty:
            summary_path = self.output_root / "continuum_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            self._write_pressure_fits(summary_df)
            logger.info(f"  Summary: {summary_path}")
        return summary_df

    def process_file(
        self,
        csv_path: Path | str,
        output_dir: Path | str,
        gas_type: str,
        transition: str,
        pressure: str,
    ) -> ContinuumResult:
        """Process one Step 1 ringdown tau CSV."""
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        wn_col = self.wavenumber_col or _find_column(df, WAVENUMBER_COLUMNS)
        tau_col = self.tau_col or _find_column(df, TAU_COLUMNS)
        tau_stats_col = self._resolve_optional_column(df, self.tau_stats_col, TAU_STATS_COLUMNS)
        pressure_col = self._resolve_optional_column(df, None, PRESSURE_COLUMNS)
        temperature_col = self._resolve_optional_column(df, None, TEMPERATURE_COLUMNS)

        wn = pd.to_numeric(df[wn_col], errors="coerce").to_numpy(dtype=float)
        tau_us = pd.to_numeric(df[tau_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(wn) & np.isfinite(tau_us) & (tau_us > 0)
        if self.window:
            lo, hi = self.window
            mask &= (wn >= lo) & (wn <= hi)
        if int(mask.sum()) < self.min_points:
            raise ValueError(f"only {int(mask.sum())} valid points after filtering")

        work = pd.DataFrame({
            "wavenumber": wn[mask],
            "tau_us": tau_us[mask],
        }).sort_values("wavenumber")

        if tau_stats_col:
            tau_stats = pd.to_numeric(df.loc[mask, tau_stats_col], errors="coerce").to_numpy(dtype=float)
            if tau_stats_col == "Tau_stats":
                tau_stats = np.abs(tau_stats) / 100.0 * work["tau_us"].to_numpy(dtype=float)
            work["tau_stats_us"] = tau_stats

        if pressure_col:
            work["pressure_torr"] = pd.to_numeric(df.loc[mask, pressure_col], errors="coerce").to_numpy(dtype=float)
        if temperature_col:
            work["temperature_c"] = pd.to_numeric(
                df.loc[mask, temperature_col], errors="coerce"
            ).to_numpy(dtype=float)

        tau_arr = work["tau_us"].to_numpy(dtype=float)
        work["loss_ppm_per_cm"] = _loss_from_tau_us(tau_arr)
        if "tau_stats_us" in work.columns:
            work["loss_stats_ppm_per_cm"] = TAU_US_TO_PPM_PER_CM * np.abs(
                work["tau_stats_us"].to_numpy(dtype=float)
            ) / np.square(tau_arr)

        ref_loss = self._reference_loss_at(work["wavenumber"].to_numpy(dtype=float))
        if ref_loss is None:
            work["reference_loss_ppm_per_cm"] = np.nan
            work["alpha_ppm_per_cm"] = np.nan
        else:
            work["reference_loss_ppm_per_cm"] = ref_loss
            work["alpha_ppm_per_cm"] = work["loss_ppm_per_cm"] - ref_loss
            if "loss_stats_ppm_per_cm" in work.columns:
                work["alpha_stats_ppm_per_cm"] = work["loss_stats_ppm_per_cm"]

        spectrum_csv = output_dir / "continuum_spectrum supplement.csv"
        work.to_csv(spectrum_csv, index=False)
        active_fit_mode = _resolve_fit_mode(self.fit_mode, pressure)
        if active_fit_mode == "o2-hitran":
            work = _add_o2_hitran_subtraction(work, pressure_label=pressure)
            fit_csv = output_dir / "continuum_step2_hitran_subtracted.csv"
        else:
            work = _add_step2_fit(
                work,
                fit_mode=active_fit_mode,
                pressure_label=pressure,
                window_cm1=self.fit_window_cm1,
                step_cm1=self.fit_step_cm1,
                order=self.fit_order,
                sigma=self.fit_sigma,
                smooth_cm1=self.fit_smooth_cm1,
            )
            fit_csv = output_dir / "continuum_step2_fit.csv"
        work.to_csv(fit_csv, index=False)
        if "cia_baseline_loss_ppm_per_cm" in work.columns:
            _write_o2_baseline_csv(work, output_dir / "continuum_step2_cia_baseline.csv")
        plot_path = output_dir / "continuum_spectrum.png"
        self._plot_spectrum(work, plot_path, f"{gas_type}/{transition}/{pressure}")

        summary = self._summarize(
            work,
            gas_type,
            transition,
            pressure,
            csv_path,
            tau_col,
            tau_stats_col,
            active_fit_mode,
        )
        return ContinuumResult(
            gas_type=gas_type,
            transition=transition,
            pressure=pressure,
            source_csv=csv_path,
            spectrum_csv=spectrum_csv,
            plot_path=plot_path,
            fit_csv=fit_csv,
            summary=summary,
        )

    def _load_reference(self) -> tuple[np.ndarray | None, np.ndarray | None] | None:
        if self.tau0_us is not None:
            if self.tau0_us <= 0:
                raise ValueError("tau0_us must be positive")
            return None, np.array([_loss_from_tau_us(np.array([self.tau0_us], dtype=float))[0]])

        if self.reference_csv is None:
            return None

        ref_df = pd.read_csv(self.reference_csv)
        wn_col = self.wavenumber_col or _find_column(ref_df, WAVENUMBER_COLUMNS)
        tau_col = self.tau_col if self.tau_col in ref_df.columns else _find_column(ref_df, TAU_COLUMNS)
        wn = pd.to_numeric(ref_df[wn_col], errors="coerce").to_numpy(dtype=float)
        tau = pd.to_numeric(ref_df[tau_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(wn) & np.isfinite(tau) & (tau > 0)
        if int(mask.sum()) < 2:
            raise ValueError(f"reference CSV needs at least 2 valid points: {self.reference_csv}")

        ref = pd.DataFrame({"wavenumber": wn[mask], "loss": _loss_from_tau_us(tau[mask])})
        ref = ref.groupby("wavenumber", as_index=False)["loss"].mean().sort_values("wavenumber")
        return ref["wavenumber"].to_numpy(dtype=float), ref["loss"].to_numpy(dtype=float)

    def _reference_label(self) -> str:
        if self.reference_csv:
            return str(self.reference_csv)
        if self.tau0_us is not None:
            return f"tau0_us={self.tau0_us:g}"
        return "none (loss only; alpha is NaN)"

    def _reference_loss_at(self, wn: np.ndarray) -> np.ndarray | None:
        if self._reference is None:
            return None
        ref_wn, ref_loss = self._reference
        if ref_wn is None:
            return np.full_like(wn, float(ref_loss[0]), dtype=float)
        return np.interp(wn, ref_wn, ref_loss, left=np.nan, right=np.nan)

    @staticmethod
    def _resolve_optional_column(
        df: pd.DataFrame,
        explicit: str | None,
        candidates: tuple[str, ...],
    ) -> str | None:
        if explicit:
            return explicit if explicit in df.columns else None
        return next((c for c in candidates if c in df.columns), None)

    def _summarize(
        self,
        work: pd.DataFrame,
        gas_type: str,
        transition: str,
        pressure: str,
        source_csv: Path,
        tau_col: str,
        tau_stats_col: str | None,
        active_fit_mode: str,
    ) -> dict:
        wn = work["wavenumber"].to_numpy(dtype=float)
        alpha = work["alpha_ppm_per_cm"].to_numpy(dtype=float)
        has_alpha = np.isfinite(alpha).any()
        pressure_mean = _nanmean(work["pressure_torr"]) if "pressure_torr" in work.columns else np.nan
        temperature_mean = _nanmean(work["temperature_c"]) if "temperature_c" in work.columns else np.nan
        pressure_median = _nanmedian_positive(work["pressure_torr"]) if "pressure_torr" in work.columns else np.nan
        temperature_median = _nanmedian(work["temperature_c"]) if "temperature_c" in work.columns else np.nan

        summary = {
            "gas_type": gas_type,
            "transition": transition,
            "pressure": pressure,
            "source_csv": str(source_csv),
            "reference": self._reference_label(),
            "tau_column": tau_col,
            "tau_stats_column": tau_stats_col or "",
            "n_points": int(len(work)),
            "wn_min": float(np.nanmin(wn)),
            "wn_max": float(np.nanmax(wn)),
            "window_min": self.window[0] if self.window else np.nan,
            "window_max": self.window[1] if self.window else np.nan,
            "pressure_mean_torr": pressure_mean,
            "temperature_mean_c": temperature_mean,
            "pressure_median_torr": pressure_median,
            "temperature_median_c": temperature_median,
            "tau_mean_us": _nanmean(work["tau_us"]),
            "tau_std_us": _nanstd(work["tau_us"]),
            "loss_mean_ppm_per_cm": _nanmean(work["loss_ppm_per_cm"]),
            "loss_median_ppm_per_cm": _nanmedian(work["loss_ppm_per_cm"]),
            "loss_std_ppm_per_cm": _nanstd(work["loss_ppm_per_cm"]),
            "loss_min_ppm_per_cm": _nanmin(work["loss_ppm_per_cm"]),
            "loss_max_ppm_per_cm": _nanmax(work["loss_ppm_per_cm"]),
            "fit_window_cm1": self.fit_window_cm1,
            "fit_step_cm1": self.fit_step_cm1,
            "fit_order": self.fit_order,
            "fit_sigma": self.fit_sigma,
            "fit_smooth_cm1": self.fit_smooth_cm1,
            "fit_mode": active_fit_mode,
            "loss_fit_resid_std_ppm_per_cm": _nanstd(
                work["loss_residual_ppm_per_cm"]
            ) if "loss_residual_ppm_per_cm" in work.columns else np.nan,
            "tau_fit_resid_std_us": _nanstd(
                work["tau_residual_us"]
            ) if "tau_residual_us" in work.columns else np.nan,
            "alpha_mean_ppm_per_cm": _nanmean(alpha) if has_alpha else np.nan,
            "alpha_median_ppm_per_cm": _nanmedian(alpha) if has_alpha else np.nan,
            "alpha_std_ppm_per_cm": _nanstd(alpha) if has_alpha else np.nan,
            "alpha_min_ppm_per_cm": _nanmin(alpha) if has_alpha else np.nan,
            "alpha_max_ppm_per_cm": _nanmax(alpha) if has_alpha else np.nan,
            "alpha_trapz_ppm_per_cm2": _trapezoid(alpha, wn) if has_alpha else np.nan,
        }
        if "loss_stats_ppm_per_cm" in work.columns:
            summary["loss_stats_median_ppm_per_cm"] = _nanmedian(work["loss_stats_ppm_per_cm"])
        if "alpha_stats_ppm_per_cm" in work.columns:
            summary["alpha_stats_median_ppm_per_cm"] = _nanmedian(work["alpha_stats_ppm_per_cm"])
        if "hitran_o2_loss_ppm_per_cm" in work.columns:
            summary["hitran_o2_loss_mean_ppm_per_cm"] = _nanmean(work["hitran_o2_loss_ppm_per_cm"])
        if "hitran_temperature_c" in work.columns:
            summary["hitran_temperature_c"] = _nanmedian(work["hitran_temperature_c"])
        if "hitran_pressure_torr" in work.columns:
            summary["hitran_pressure_torr"] = _nanmedian(work["hitran_pressure_torr"])
        if "loss_minus_hitran_ppm_per_cm" in work.columns:
            summary["loss_minus_hitran_mean_ppm_per_cm"] = _nanmean(
                work["loss_minus_hitran_ppm_per_cm"]
            )
            summary["loss_minus_hitran_std_ppm_per_cm"] = _nanstd(
                work["loss_minus_hitran_ppm_per_cm"]
            )
        if "cia_baseline_loss_ppm_per_cm" in work.columns:
            summary["cia_baseline_mean_ppm_per_cm"] = _nanmean(
                work["cia_baseline_loss_ppm_per_cm"]
            )
        if "o2_fit_used" in work.columns:
            fit_used = work["o2_fit_used"].astype(bool).to_numpy()
            absorption_mask = work["o2_absorption_mask"].astype(bool).to_numpy()
            summary["o2_fit_used_points"] = int(fit_used.sum())
            summary["o2_absorption_masked_points"] = int(absorption_mask.sum())
            summary["o2_absorption_masked_fraction"] = (
                float(absorption_mask.sum() / len(work)) if len(work) else np.nan
            )
            if "loss_residual_ppm_per_cm" in work.columns:
                residual = work["loss_residual_ppm_per_cm"].to_numpy(dtype=float)
                summary["o2_fit_used_loss_resid_std_ppm_per_cm"] = _nanstd(
                    residual[fit_used]
                )
        if "hitran_subtracted_residual_ppm_per_cm" in work.columns:
            summary["hitran_subtracted_fit_resid_std_ppm_per_cm"] = _nanstd(
                work["hitran_subtracted_residual_ppm_per_cm"]
            )
        return summary

    @staticmethod
    def _plot_spectrum(work: pd.DataFrame, plot_path: Path, title: str) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        wn = work["wavenumber"].to_numpy(dtype=float)

        axes[0].plot(
            wn, work["tau_us"], ".", ms=2, color="steelblue", alpha=0.65,
            label="Step 1 tau"
        )
        if "tau_fit_us" in work.columns:
            axes[0].plot(
                wn, work["tau_fit_us"], "-", lw=1.8, color="black",
                label="Step 2 fit"
            )
        axes[0].set_ylabel("Ring-down time (us)")
        axes[0].set_title(title)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        loss = work["loss_ppm_per_cm"].to_numpy(dtype=float)
        if "o2_absorption_mask" in work.columns:
            fit_used = work["o2_fit_used"].astype(bool).to_numpy()
            absorption_mask = work["o2_absorption_mask"].astype(bool).to_numpy()
            axes[1].plot(
                wn[fit_used],
                loss[fit_used],
                ".",
                ms=2,
                color="slategray",
                alpha=0.55,
                label="Baseline fit points",
            )
            axes[1].plot(
                wn[absorption_mask],
                loss[absorption_mask],
                ".",
                ms=2,
                color="tomato",
                alpha=0.75,
                label="Masked O2 absorption",
            )
        else:
            axes[1].plot(
                wn, work["loss_ppm_per_cm"], ".", ms=2, color="tomato", alpha=0.65,
                label="Loss"
            )
        if "hitran_o2_loss_ppm_per_cm" in work.columns:
            axes[1].plot(
                wn,
                work["hitran_o2_loss_ppm_per_cm"],
                "-",
                lw=1.1,
                color="royalblue",
                alpha=0.9,
                label="HITRAN O2",
            )
        if "loss_minus_hitran_ppm_per_cm" in work.columns and "o2_absorption_mask" not in work.columns:
            axes[1].plot(
                wn,
                work["loss_minus_hitran_ppm_per_cm"],
                ".",
                ms=2,
                color="black",
                alpha=0.65,
                label="Loss - HITRAN",
            )
        if "hitran_subtracted_fit_ppm_per_cm" in work.columns:
            axes[1].plot(
                wn,
                work["hitran_subtracted_fit_ppm_per_cm"],
                "-",
                lw=1.8,
                color="darkgreen",
                label="Fit(loss - HITRAN)",
            )
        if "loss_fit_ppm_per_cm" in work.columns:
            label = "CIA baseline" if "cia_baseline_loss_ppm_per_cm" in work.columns else "Step 2 fit"
            axes[1].plot(
                wn, work["loss_fit_ppm_per_cm"], "-", lw=1.8, color="black",
                alpha=0.9, label=label
            )
        axes[1].set_xlabel("Wavenumber (cm-1)")
        axes[1].set_ylabel("Loss (ppm/cm)")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def _write_pressure_fits(self, summary_df: pd.DataFrame) -> None:
        records: list[dict] = []
        for (gas_type, transition), grp in summary_df.groupby(["gas_type", "transition"], dropna=False):
            x = pd.to_numeric(grp["pressure_mean_torr"], errors="coerce").to_numpy(dtype=float)
            for y_col in ("alpha_mean_ppm_per_cm", "loss_mean_ppm_per_cm"):
                y = pd.to_numeric(grp[y_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < 2:
                    continue
                fit = _fit_pressure_dependence(x[mask], y[mask])
                records.append({
                    "gas_type": gas_type,
                    "transition": transition,
                    "metric": y_col,
                    **fit,
                })
        if records:
            fit_df = pd.DataFrame(records)
            fit_df.to_csv(self.output_root / "continuum_pressure_fits.csv", index=False)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"missing required column; tried {', '.join(candidates)}")
    return col


def _write_o2_baseline_csv(work: pd.DataFrame, csv_path: Path) -> None:
    columns = [
        "wavenumber",
        "cia_baseline_loss_ppm_per_cm",
        "tau_fit_us",
        "o2_absorption_mask",
        "o2_fit_used",
        "hitran_o2_loss_ppm_per_cm",
        "hitran_mask_threshold_ppm_per_cm",
        "hitran_mask_ratio",
        "hitran_mask_margin_cm1",
        "hitran_temperature_c",
        "hitran_pressure_torr",
        "hitran_step_cm1",
    ]
    existing = [col for col in columns if col in work.columns]
    work[existing].to_csv(csv_path, index=False)


def _normalize_window(window: tuple[float, float] | None) -> tuple[float, float] | None:
    if window is None:
        return None
    lo, hi = float(window[0]), float(window[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _normalize_fit_mode(fit_mode: str) -> str:
    mode = str(fit_mode or "auto").strip().lower().replace("_", "-")
    aliases = {
        "argon": "ar",
        "loss": "ar",
        "loss-domain": "ar",
        "oxygen": "o2",
        "tau-envelope": "o2",
        "envelope": "o2",
        "hitran": "o2-hitran",
        "o2hitran": "o2-hitran",
        "o2-subtract": "o2-hitran",
        "hitran-subtract": "o2-hitran",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "ar", "o2", "o2-hitran"}:
        raise ValueError("fit_mode must be one of: auto, ar, o2, o2-hitran")
    return mode


def _resolve_fit_mode(fit_mode: str, pressure: str) -> str:
    mode = _normalize_fit_mode(fit_mode)
    if mode != "auto":
        return mode

    label = str(pressure).upper()
    if "O2" in label or "O₂" in label:
        return "o2"
    return "ar"


def _loss_from_tau_us(tau_us: np.ndarray) -> np.ndarray:
    return TAU_US_TO_PPM_PER_CM / tau_us


def _add_step2_fit(
    work: pd.DataFrame,
    fit_mode: str,
    pressure_label: str,
    window_cm1: float,
    step_cm1: float,
    order: int,
    sigma: float,
    smooth_cm1: float,
) -> pd.DataFrame:
    if _normalize_fit_mode(fit_mode) == "o2":
        return _add_o2_hitran_masked_baseline_fit(
            work=work,
            pressure_label=pressure_label,
            window_cm1=window_cm1,
            step_cm1=step_cm1,
            order=order,
            sigma=sigma,
            smooth_cm1=smooth_cm1,
        )
    return _add_sliding_loss_fit(
        work=work,
        window_cm1=window_cm1,
        step_cm1=step_cm1,
        order=order,
        sigma=sigma,
        smooth_cm1=smooth_cm1,
    )


def _add_o2_hitran_subtraction(
    work: pd.DataFrame,
    pressure_label: str,
) -> pd.DataFrame:
    """Subtract the HITRAN O2 simulation from measured total cavity loss."""
    out = work.copy()
    wn = out["wavenumber"].to_numpy(dtype=float)
    loss = out["loss_ppm_per_cm"].to_numpy(dtype=float)
    mask = np.isfinite(wn) & np.isfinite(loss)
    if int(mask.sum()) < 2:
        raise ValueError("not enough finite points for O2 HITRAN subtraction")

    temperature_c = (
        _nanmedian(out["temperature_c"])
        if "temperature_c" in out.columns else np.nan
    )
    pressure_torr = (
        _nanmedian_positive(out["pressure_torr"])
        if "pressure_torr" in out.columns else np.nan
    )
    if not np.isfinite(pressure_torr):
        pressure_torr = _pressure_torr_from_label(pressure_label)
    if not np.isfinite(temperature_c):
        raise ValueError("O2 HITRAN subtraction requires temperature_c in the Step 1 CSV")
    if not np.isfinite(pressure_torr) or pressure_torr <= 0:
        raise ValueError("O2 HITRAN subtraction requires pressure_torr in the Step 1 CSV or pressure label")

    hitran_loss = _simulate_o2_hitran_loss_ppm_per_cm(
        wavenumber=wn,
        temperature_c=float(temperature_c),
        pressure_torr=float(pressure_torr),
    )
    residual = loss - hitran_loss
    tau_equiv = np.full_like(residual, np.nan, dtype=float)
    positive = np.isfinite(residual) & (residual > 0)
    tau_equiv[positive] = TAU_US_TO_PPM_PER_CM / residual[positive]

    out["hitran_o2_loss_ppm_per_cm"] = hitran_loss
    out["hitran_o2_absorption_cm_inv"] = hitran_loss / 1e6
    out["loss_minus_hitran_ppm_per_cm"] = residual
    out["tau_equiv_after_hitran_us"] = tau_equiv
    out["hitran_temperature_c"] = float(temperature_c)
    out["hitran_temperature_k"] = float(temperature_c) + 273.15
    out["hitran_pressure_torr"] = float(pressure_torr)
    out["hitran_pressure_atm"] = float(pressure_torr) / 760.0
    out["hitran_step_cm1"] = HITRAN_O2_STEP_CM1
    out["step2_fit_mode"] = "o2-hitran"
    return out


def _add_o2_hitran_masked_baseline_fit(
    work: pd.DataFrame,
    pressure_label: str,
    window_cm1: float,
    step_cm1: float,
    order: int,
    sigma: float,
    smooth_cm1: float,
) -> pd.DataFrame:
    """Mask HITRAN O2 absorption points, then fit the slow CIA baseline."""
    out = _add_o2_hitran_subtraction(work, pressure_label=pressure_label)
    wn = out["wavenumber"].to_numpy(dtype=float)
    loss = out["loss_ppm_per_cm"].to_numpy(dtype=float)
    hitran_loss = out["hitran_o2_loss_ppm_per_cm"].to_numpy(dtype=float)
    absorption_mask, mask_threshold = _hitran_absorption_mask(
        wavenumber=wn,
        hitran_loss=hitran_loss,
    )
    finite = np.isfinite(wn) & np.isfinite(loss)
    fit_used = finite & ~absorption_mask

    baseline = _fit_loss_values(
        wn=wn,
        loss=loss,
        fit_mask=fit_used,
        window_cm1=window_cm1,
        step_cm1=step_cm1,
        order=order,
        sigma=sigma,
        smooth_cm1=smooth_cm1,
        error_label="O2 HITRAN-masked CIA baseline fit",
    )
    tau_fit = np.full_like(baseline, np.nan, dtype=float)
    positive = np.isfinite(baseline) & (baseline > 0)
    tau_fit[positive] = TAU_US_TO_PPM_PER_CM / baseline[positive]

    out["o2_absorption_mask"] = absorption_mask
    out["o2_fit_used"] = fit_used
    out["hitran_mask_threshold_ppm_per_cm"] = mask_threshold
    out["hitran_mask_ratio"] = HITRAN_O2_MASK_RATIO
    out["hitran_mask_margin_cm1"] = HITRAN_O2_MASK_MARGIN_CM1
    out["cia_baseline_loss_ppm_per_cm"] = baseline
    out["cia_baseline_residual_ppm_per_cm"] = out["loss_ppm_per_cm"] - baseline
    out["loss_fit_ppm_per_cm"] = baseline
    out["loss_residual_ppm_per_cm"] = out["loss_ppm_per_cm"] - baseline
    out["tau_fit_us"] = tau_fit
    out["tau_residual_us"] = out["tau_us"] - tau_fit
    out["step2_fit_mode"] = "o2"
    return out


def _add_sliding_loss_fit(
    work: pd.DataFrame,
    window_cm1: float,
    step_cm1: float,
    order: int,
    sigma: float,
    smooth_cm1: float,
) -> pd.DataFrame:
    """Fit the continuum trend in loss domain using overlapping windows."""
    if window_cm1 <= 0:
        raise ValueError("fit_window_cm1 must be positive")
    if step_cm1 <= 0:
        raise ValueError("fit_step_cm1 must be positive")
    if order < 0:
        raise ValueError("fit_order must be >= 0")

    out = work.copy()
    wn = out["wavenumber"].to_numpy(dtype=float)
    loss = out["loss_ppm_per_cm"].to_numpy(dtype=float)
    fitted = _fit_loss_values(
        wn=wn,
        loss=loss,
        fit_mask=None,
        window_cm1=window_cm1,
        step_cm1=step_cm1,
        order=order,
        sigma=sigma,
        smooth_cm1=smooth_cm1,
        error_label="CIA Step 2 fit",
    )
    tau_fit = TAU_US_TO_PPM_PER_CM / fitted
    out["loss_fit_ppm_per_cm"] = fitted
    out["loss_residual_ppm_per_cm"] = out["loss_ppm_per_cm"] - fitted
    out["tau_fit_us"] = tau_fit
    out["tau_residual_us"] = out["tau_us"] - tau_fit
    out["step2_fit_mode"] = "ar"
    return out


def _fit_loss_values(
    wn: np.ndarray,
    loss: np.ndarray,
    fit_mask: np.ndarray | None,
    window_cm1: float,
    step_cm1: float,
    order: int,
    sigma: float,
    smooth_cm1: float,
    error_label: str,
) -> np.ndarray:
    if window_cm1 <= 0:
        raise ValueError("fit_window_cm1 must be positive")
    if step_cm1 <= 0:
        raise ValueError("fit_step_cm1 must be positive")
    if order < 0:
        raise ValueError("fit_order must be >= 0")

    mask = np.isfinite(wn) & np.isfinite(loss)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != mask.shape:
            raise ValueError("fit_mask must have the same shape as wn")
        mask &= fit_mask
    if int(mask.sum()) < max(order + 1, 2):
        raise ValueError(f"not enough finite points for {error_label}")

    x_min = float(np.nanmin(wn[mask]))
    x_max = float(np.nanmax(wn[mask]))
    half_window = window_cm1 / 2.0

    fitted = _continuous_anchor_loss_fit(
        x=wn,
        y=loss,
        mask=mask,
        x_min=x_min,
        x_max=x_max,
        half_window=half_window,
        step_cm1=step_cm1,
        order=order,
        sigma=sigma,
    )
    if smooth_cm1 > 0:
        fitted = _smooth_fit_by_width(
            x=wn,
            y=fitted,
            smooth_cm1=smooth_cm1,
            order=min(max(order, 1), 3),
        )
    return fitted


def _add_o2_tau_envelope_fit(
    work: pd.DataFrame,
    window_cm1: float,
    step_cm1: float,
    order: int,
    sigma: float,
    smooth_cm1: float,
) -> pd.DataFrame:
    """Fit an absorption-aware O2 baseline in tau domain.

    O2 narrow absorption increases loss and therefore pushes tau downward.
    The background tau should therefore be estimated from the upper envelope,
    not from all points as in Ar.
    """
    if window_cm1 <= 0:
        raise ValueError("fit_window_cm1 must be positive")
    if step_cm1 <= 0:
        raise ValueError("fit_step_cm1 must be positive")
    if order < 0:
        raise ValueError("fit_order must be >= 0")

    out = work.copy()
    wn = out["wavenumber"].to_numpy(dtype=float)
    tau = out["tau_us"].to_numpy(dtype=float)
    mask = np.isfinite(wn) & np.isfinite(tau) & (tau > 0)
    if int(mask.sum()) < max(order + 1, 2):
        raise ValueError("not enough finite points for O2 Step 2 fit")

    x_min = float(np.nanmin(wn[mask]))
    x_max = float(np.nanmax(wn[mask]))
    half_window = window_cm1 / 2.0
    tau_fit = _continuous_anchor_tau_envelope_fit(
        x=wn,
        y=tau,
        mask=mask,
        x_min=x_min,
        x_max=x_max,
        half_window=half_window,
        step_cm1=step_cm1,
        order=order,
        sigma=sigma,
    )
    if smooth_cm1 > 0:
        tau_fit = _smooth_fit_by_width(
            x=wn,
            y=tau_fit,
            smooth_cm1=smooth_cm1,
            order=min(max(order, 1), 3),
        )

    loss_fit = TAU_US_TO_PPM_PER_CM / tau_fit
    out["tau_fit_us"] = tau_fit
    out["tau_residual_us"] = out["tau_us"] - tau_fit
    out["loss_fit_ppm_per_cm"] = loss_fit
    out["loss_residual_ppm_per_cm"] = out["loss_ppm_per_cm"] - loss_fit
    out["step2_fit_mode"] = "o2"
    return out


def _continuous_anchor_loss_fit(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    x_min: float,
    x_max: float,
    half_window: float,
    step_cm1: float,
    order: int,
    sigma: float,
) -> np.ndarray:
    centers = _fit_centers(x_min=x_min, x_max=x_max, step_cm1=step_cm1)
    min_points = max(order + 2, 8)
    anchor_x: list[float] = []
    anchor_y: list[float] = []

    for center in centers:
        local = mask & (np.abs(x - center) <= half_window)
        if int(local.sum()) < min_points:
            continue
        fit_order = min(order, int(local.sum()) - 1)
        y_center = _robust_polyfit_eval(
            x_fit=x[local],
            y_fit=y[local],
            x_eval=np.array([center], dtype=float),
            order=fit_order,
            sigma=sigma,
        )[0]
        if np.isfinite(y_center):
            anchor_x.append(center)
            anchor_y.append(float(y_center))

    if len(anchor_x) < 2:
        return _robust_polyfit_eval(
            x_fit=x[mask],
            y_fit=y[mask],
            x_eval=x,
            order=min(order, int(mask.sum()) - 1),
            sigma=sigma,
        )

    anchors = (
        pd.DataFrame({"x": anchor_x, "y": anchor_y})
        .groupby("x", as_index=False)["y"].mean()
        .sort_values("x")
    )
    x_anchor = anchors["x"].to_numpy(dtype=float)
    y_anchor = anchors["y"].to_numpy(dtype=float)

    if x_anchor[0] > x_min:
        x_anchor = np.insert(x_anchor, 0, x_min)
        y_anchor = np.insert(y_anchor, 0, _nearest_finite_y(x, y, x_min, mask))
    if x_anchor[-1] < x_max:
        x_anchor = np.append(x_anchor, x_max)
        y_anchor = np.append(y_anchor, _nearest_finite_y(x, y, x_max, mask))

    from scipy.interpolate import PchipInterpolator

    interpolator = PchipInterpolator(x_anchor, y_anchor, extrapolate=True)
    return interpolator(x)


def _hitran_absorption_mask(
    wavenumber: np.ndarray,
    hitran_loss: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return points excluded from O2 CIA baseline fitting."""
    wn = np.asarray(wavenumber, dtype=float)
    loss = np.asarray(hitran_loss, dtype=float)
    valid = np.isfinite(wn) & np.isfinite(loss)
    mask = np.zeros_like(wn, dtype=bool)
    if int(valid.sum()) < 2:
        return mask, np.nan

    peak = float(np.nanmax(loss[valid]))
    if not np.isfinite(peak) or peak <= 0:
        return mask, np.nan

    threshold = peak * HITRAN_O2_MASK_RATIO
    mask = valid & (loss >= threshold)
    if HITRAN_O2_MASK_MARGIN_CM1 > 0 and mask.any():
        mask = _expand_mask_by_wavenumber_margin(
            wavenumber=wn,
            mask=mask,
            margin_cm1=HITRAN_O2_MASK_MARGIN_CM1,
        )
    return mask, threshold


def _expand_mask_by_wavenumber_margin(
    wavenumber: np.ndarray,
    mask: np.ndarray,
    margin_cm1: float,
) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if margin_cm1 <= 0 or not out.any():
        return out

    wn = np.asarray(wavenumber, dtype=float)
    finite = np.isfinite(wn)
    idx = np.where(out & finite)[0]
    if len(idx) == 0:
        return out

    for center in wn[idx]:
        lo = center - margin_cm1
        hi = center + margin_cm1
        out |= finite & (wn >= lo) & (wn <= hi)
    return out


def _continuous_anchor_tau_envelope_fit(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    x_min: float,
    x_max: float,
    half_window: float,
    step_cm1: float,
    order: int,
    sigma: float,
) -> np.ndarray:
    centers = _fit_centers(x_min=x_min, x_max=x_max, step_cm1=step_cm1)
    min_points = max(order + 2, 8)
    anchor_x: list[float] = []
    anchor_y: list[float] = []

    for center in centers:
        local = mask & (np.abs(x - center) <= half_window)
        if int(local.sum()) < min_points:
            continue
        fit_order = min(order, int(local.sum()) - 1)
        y_center = _upper_envelope_polyfit_eval(
            x_fit=x[local],
            y_fit=y[local],
            x_eval=np.array([center], dtype=float),
            order=fit_order,
            sigma=sigma,
        )[0]
        if np.isfinite(y_center) and y_center > 0:
            anchor_x.append(center)
            anchor_y.append(float(y_center))

    if len(anchor_x) < 2:
        return _upper_envelope_polyfit_eval(
            x_fit=x[mask],
            y_fit=y[mask],
            x_eval=x,
            order=min(order, int(mask.sum()) - 1),
            sigma=sigma,
        )

    anchors = (
        pd.DataFrame({"x": anchor_x, "y": anchor_y})
        .groupby("x", as_index=False)["y"].mean()
        .sort_values("x")
    )
    x_anchor = anchors["x"].to_numpy(dtype=float)
    y_anchor = anchors["y"].to_numpy(dtype=float)

    if x_anchor[0] > x_min:
        x_anchor = np.insert(x_anchor, 0, x_min)
        y_anchor = np.insert(y_anchor, 0, _nearest_finite_y(x, y, x_min, mask))
    if x_anchor[-1] < x_max:
        x_anchor = np.append(x_anchor, x_max)
        y_anchor = np.append(y_anchor, _nearest_finite_y(x, y, x_max, mask))

    from scipy.interpolate import PchipInterpolator

    interpolator = PchipInterpolator(x_anchor, y_anchor, extrapolate=True)
    return interpolator(x)


def _fit_centers(x_min: float, x_max: float, step_cm1: float) -> list[float]:
    centers = list(np.arange(x_min, x_max + step_cm1, step_cm1, dtype=float))
    centers.extend([x_min, x_max])
    return sorted({round(c, 10) for c in centers if x_min <= c <= x_max})


def _nearest_finite_y(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
    mask: np.ndarray,
) -> float:
    valid_idx = np.where(mask)[0]
    idx = valid_idx[int(np.argmin(np.abs(x[valid_idx] - target)))]
    return float(y[idx])


def _smooth_fit_by_width(
    x: np.ndarray,
    y: np.ndarray,
    smooth_cm1: float,
    order: int,
) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    if smooth_cm1 <= 0 or int(mask.sum()) < 5:
        return y

    spacing = np.diff(np.sort(x[mask]))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    if len(spacing) == 0:
        return y

    median_spacing = float(np.median(spacing))
    window_points = max(int(round(smooth_cm1 / median_spacing)), 5)
    if window_points % 2 == 0:
        window_points += 1
    if window_points >= int(mask.sum()):
        window_points = int(mask.sum()) - 1
        if window_points % 2 == 0:
            window_points -= 1
    if window_points <= order + 2:
        window_points = order + 3
        if window_points % 2 == 0:
            window_points += 1
    if window_points >= int(mask.sum()) or window_points < 5:
        return y

    from scipy.signal import savgol_filter

    out = y.copy()
    out[mask] = savgol_filter(
        y[mask],
        window_length=window_points,
        polyorder=min(order, window_points - 2),
        mode="interp",
    )
    return out


def _robust_polyfit_eval(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    order: int,
    sigma: float,
) -> np.ndarray:
    x0 = float(np.nanmean(x_fit))
    scale = float(np.nanmax(np.abs(x_fit - x0)))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    x_scaled = (x_fit - x0) / scale
    eval_scaled = (x_eval - x0) / scale
    active = np.isfinite(x_scaled) & np.isfinite(y_fit)
    fit_order = min(order, max(int(active.sum()) - 1, 0))

    for _ in range(4):
        if int(active.sum()) <= fit_order:
            break
        coeff = np.polyfit(x_scaled[active], y_fit[active], deg=fit_order)
        residual = y_fit - np.polyval(coeff, x_scaled)
        resid_active = residual[active]
        scale_resid = _robust_scale(resid_active)
        if not np.isfinite(scale_resid) or scale_resid <= 0 or sigma <= 0:
            break
        next_active = active & (np.abs(residual) <= sigma * scale_resid)
        if np.array_equal(next_active, active) or int(next_active.sum()) <= fit_order:
            break
        active = next_active

    coeff = np.polyfit(x_scaled[active], y_fit[active], deg=fit_order)
    return np.polyval(coeff, eval_scaled)


def _upper_envelope_polyfit_eval(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    order: int,
    sigma: float,
) -> np.ndarray:
    """Robust local polynomial fit biased toward the upper tau envelope."""
    x0 = float(np.nanmean(x_fit))
    scale = float(np.nanmax(np.abs(x_fit - x0)))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    x_scaled = (x_fit - x0) / scale
    eval_scaled = (x_eval - x0) / scale
    finite = np.isfinite(x_scaled) & np.isfinite(y_fit)
    fit_order = min(order, max(int(finite.sum()) - 1, 0))
    if int(finite.sum()) <= fit_order:
        coeff = np.polyfit(x_scaled[finite], y_fit[finite], deg=max(int(finite.sum()) - 1, 0))
        return np.polyval(coeff, eval_scaled)

    preliminary = _robust_polyfit_eval(
        x_fit=x_fit[finite],
        y_fit=y_fit[finite],
        x_eval=x_fit[finite],
        order=fit_order,
        sigma=sigma,
    )
    residual = y_fit[finite] - preliminary
    residual_cut = np.nanquantile(residual, 0.55)
    active = finite.copy()
    active_indices = np.where(finite)[0]
    active[active_indices] = residual >= residual_cut
    if int(active.sum()) <= fit_order:
        residual_cut = np.nanquantile(residual, 0.40)
        active = finite.copy()
        active[active_indices] = residual >= residual_cut
    if int(active.sum()) <= fit_order:
        active = finite.copy()

    active = _iterative_lower_clip_active(
        x_scaled=x_scaled,
        y_fit=y_fit,
        active=active,
        fit_order=fit_order,
        sigma=sigma,
    )
    coeff = np.polyfit(x_scaled[active], y_fit[active], deg=fit_order)
    return np.polyval(coeff, eval_scaled)


def _iterative_lower_clip_active(
    x_scaled: np.ndarray,
    y_fit: np.ndarray,
    active: np.ndarray,
    fit_order: int,
    sigma: float,
) -> np.ndarray:
    finite = np.isfinite(x_scaled) & np.isfinite(y_fit)
    sigma = max(float(sigma), 0.0)
    for _ in range(6):
        if int(active.sum()) <= fit_order:
            break
        coeff = np.polyfit(x_scaled[active], y_fit[active], deg=fit_order)
        residual = y_fit - np.polyval(coeff, x_scaled)
        active_resid = residual[active]
        negative_resid = active_resid[active_resid < 0]
        scale_resid = _robust_scale(negative_resid)
        if not np.isfinite(scale_resid) or scale_resid <= 0:
            scale_resid = _robust_scale(active_resid)
        if not np.isfinite(scale_resid) or scale_resid <= 0 or sigma <= 0:
            break
        next_active = finite & (residual >= -sigma * scale_resid)
        if np.array_equal(next_active, active) or int(next_active.sum()) <= fit_order:
            break
        active = next_active
    return active


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return 1.4826 * float(np.nanmedian(np.abs(values - np.nanmedian(values))))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return np.nan
    x_valid = x[mask]
    y_valid = y[mask]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_valid, x_valid))
    dx = np.diff(x_valid)
    return float(np.sum(dx * (y_valid[:-1] + y_valid[1:]) * 0.5))


def _simulate_o2_hitran_loss_ppm_per_cm(
    wavenumber: np.ndarray,
    temperature_c: float,
    pressure_torr: float,
) -> np.ndarray:
    """Simulate O2 absorption loss with the local HITRAN/HAPI table."""
    wn = np.asarray(wavenumber, dtype=float)
    valid = np.isfinite(wn)
    if int(valid.sum()) < 2:
        return np.full_like(wn, np.nan, dtype=float)

    hitran_dir = HITRAN_DIR
    if not hitran_dir.exists():
        raise FileNotFoundError(f"HITRAN data directory not found: {hitran_dir}")

    with contextlib.redirect_stdout(io.StringIO()):
        import hapi  # type: ignore

    saved_dir = os.getcwd()
    os.chdir(hitran_dir)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hapi.db_begin(str(hitran_dir))
        table = _resolve_o2_hitran_table(hapi)

        wn_min = float(np.nanmin(wn[valid])) - 0.5
        wn_max = float(np.nanmax(wn[valid])) + 0.5
        temperature_k = float(temperature_c) + 273.15
        pressure_atm = float(pressure_torr) / 760.0

        with contextlib.redirect_stdout(io.StringIO()):
            nu_sim, alpha_cm_inv = hapi.absorptionCoefficient_Voigt(
                SourceTables=table,
                Environment={"T": temperature_k, "p": pressure_atm},
                WavenumberRange=[wn_min, wn_max],
                WavenumberStep=HITRAN_O2_STEP_CM1,
                HITRAN_units=False,
            )
    finally:
        os.chdir(saved_dir)

    nu_sim = np.asarray(nu_sim, dtype=float)
    alpha_cm_inv = np.asarray(alpha_cm_inv, dtype=float).reshape(-1)
    sim_mask = np.isfinite(nu_sim) & np.isfinite(alpha_cm_inv)
    if int(sim_mask.sum()) < 2:
        return np.full_like(wn, np.nan, dtype=float)

    hitran_loss = np.interp(
        wn,
        nu_sim[sim_mask],
        alpha_cm_inv[sim_mask],
        left=np.nan,
        right=np.nan,
    )
    return hitran_loss * 1e6


def _resolve_o2_hitran_table(hapi) -> str:
    for tname in hapi.LOCAL_TABLE_CACHE:
        if str(tname).startswith("O2"):
            return str(tname)
    raise FileNotFoundError(
        f"No local O2 HITRAN table found in {HITRAN_DIR}. "
        "Run the HITRAN download step first."
    )


def _pressure_torr_from_label(label: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Torr", str(label), flags=re.IGNORECASE)
    if not match:
        return np.nan
    return float(match.group(1))


def _fit_pressure_dependence(x: np.ndarray, y: np.ndarray) -> dict:
    linear = np.polyfit(x, y, deg=1)
    y_lin = np.polyval(linear, x)
    out = {
        "n_points": int(len(x)),
        "linear_intercept": float(linear[1]),
        "linear_slope_per_torr": float(linear[0]),
        "linear_r2": _r_squared(y, y_lin),
        "quadratic_intercept": np.nan,
        "quadratic_linear_per_torr": np.nan,
        "quadratic_term_per_torr2": np.nan,
        "quadratic_r2": np.nan,
    }
    if len(x) >= 3:
        quad = np.polyfit(x, y, deg=2)
        y_quad = np.polyval(quad, x)
        out.update({
            "quadratic_intercept": float(quad[2]),
            "quadratic_linear_per_torr": float(quad[1]),
            "quadratic_term_per_torr2": float(quad[0]),
            "quadratic_r2": _r_squared(y, y_quad),
        })
    return out


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum(np.square(y - y_hat)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def _nanmedian(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan


def _nanmedian_positive(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.nanmedian(arr)) if len(arr) else np.nan


def _nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else np.nan


def _nanmin(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan


def _nanmax(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan
