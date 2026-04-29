"""Continuum absorption analysis from CRDS Step 1 tau spectra.

This module starts from the same ring-down summary used by line absorption
processing. It converts ring-down times into a cavity-loss spectrum,

    loss = 1e12 / c / tau_us

where ``loss`` is in ppm/cm and ``tau_us`` is in microseconds. If an
empty-cavity/reference tau is supplied, the reference loss is subtracted
to produce the absorption coefficient ``alpha_ppm_per_cm``.
"""

from __future__ import annotations

import os
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
RINGDOWN_CSV_NAME = "ringdown_results.csv"

C_CM_PER_S = 2.99792458e10
TAU_US_TO_PPM_PER_CM = 1e12 / C_CM_PER_S

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
        plot_path = output_dir / "continuum_spectrum.png"
        self._plot_spectrum(work, plot_path, f"{gas_type}/{transition}/{pressure}")

        summary = self._summarize(work, gas_type, transition, pressure, csv_path, tau_col, tau_stats_col)
        return ContinuumResult(
            gas_type=gas_type,
            transition=transition,
            pressure=pressure,
            source_csv=csv_path,
            spectrum_csv=spectrum_csv,
            plot_path=plot_path,
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
    ) -> dict:
        wn = work["wavenumber"].to_numpy(dtype=float)
        alpha = work["alpha_ppm_per_cm"].to_numpy(dtype=float)
        has_alpha = np.isfinite(alpha).any()
        pressure_mean = _nanmean(work["pressure_torr"]) if "pressure_torr" in work.columns else np.nan
        temperature_mean = _nanmean(work["temperature_c"]) if "temperature_c" in work.columns else np.nan

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
            "tau_mean_us": _nanmean(work["tau_us"]),
            "tau_std_us": _nanstd(work["tau_us"]),
            "loss_mean_ppm_per_cm": _nanmean(work["loss_ppm_per_cm"]),
            "loss_median_ppm_per_cm": _nanmedian(work["loss_ppm_per_cm"]),
            "loss_std_ppm_per_cm": _nanstd(work["loss_ppm_per_cm"]),
            "loss_min_ppm_per_cm": _nanmin(work["loss_ppm_per_cm"]),
            "loss_max_ppm_per_cm": _nanmax(work["loss_ppm_per_cm"]),
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
        return summary

    @staticmethod
    def _plot_spectrum(work: pd.DataFrame, plot_path: Path, title: str) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        wn = work["wavenumber"].to_numpy(dtype=float)

        axes[0].plot(wn, work["loss_ppm_per_cm"], ".", ms=2, color="steelblue")
        axes[0].set_ylabel("Loss (ppm/cm)")
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.3)

        alpha = work["alpha_ppm_per_cm"].to_numpy(dtype=float)
        if np.isfinite(alpha).any():
            axes[1].plot(wn, alpha, ".", ms=2, color="tomato")
        else:
            axes[1].text(0.5, 0.5, "No reference tau: alpha not computed",
                         transform=axes[1].transAxes, ha="center", va="center")
        axes[1].set_xlabel("Wavenumber (cm-1)")
        axes[1].set_ylabel("Alpha (ppm/cm)")
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


def _normalize_window(window: tuple[float, float] | None) -> tuple[float, float] | None:
    if window is None:
        return None
    lo, hi = float(window[0]), float(window[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _loss_from_tau_us(tau_us: np.ndarray) -> np.ndarray:
    return TAU_US_TO_PPM_PER_CM / tau_us


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


def _nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else np.nan


def _nanmin(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan


def _nanmax(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan
