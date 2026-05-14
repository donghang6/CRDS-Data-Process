#!/usr/bin/env python3
"""Process the 296 K raw O2/Ar CIA data and calculate final B spectra.

This script is a dedicated bridge for the 296 K raw-data folder whose raw files
are named like

    1 9120.00000 0.00 Pa 273.15℃ 20250414145405.txt

and whose file contents contain ring-down times in the first column.  The
temperature and pressure written in these raw filenames are not used; pressure is
taken from the parent folder name (300Torr/500Torr/700Torr), and temperature is
set by --temperature-k.

Processing steps:
  1. Read each raw file, keep the first column as tau, and remove tau outliers
     within each file.
  2. Save one Step-1-like CSV per pressure and gas.
  3. Fit the Ar slow cavity-loss baseline with the same sliding loss-domain fit
     used for Ar continuum data.
  4. Fit the O2 CIA baseline with the same HITRAN-masked O2 Step 2 method.
  5. Interpolate Ar and O2 fitted baselines onto a 0.01 cm-1 grid and calculate
     alpha = (loss_O2_baseline - loss_Ar_baseline) * 1e-6 cm-1.
  6. Calculate B = alpha / rho^2, where rho is the O2 density in amagat.

Example:
    conda run -n CRDS-Data-Process env MPLCONFIGDIR=/private/tmp/mplconfig \\
      PYTHONPATH=src python scripts/process_296k_raw_cia_b.py \\
      --raw-root '/Users/donghang/科研/实验数据/氧气连续吸收/原始数据' \\
      --temperature-k 296 \\
      --pressures 300 500 700 \\
      --grid-min 9120 --grid-max 9820 --grid-step 0.01
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crds_process.continuum.analysis import TAU_US_TO_PPM_PER_CM, _add_step2_fit
from crds_process.ringdown.filtering import filter_ringdown_times


RAW_NAME_RE = re.compile(
    r"^\s*(?:(?P<index>\d+)\s+)?"
    r"(?P<wavenumber>-?\d+(?:\.\d+)?)"
    r"(?:\s+.*?)?"
    r"(?:\s+(?P<timestamp>\d{14}))?"
    r"\.txt$",
)


@dataclass(frozen=True)
class FitParams:
    window: float
    step: float
    order: int
    sigma: float
    smooth: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process 296 K raw O2/Ar CIA data and calculate B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/Users/donghang/科研/实验数据/氧气连续吸收/原始数据"),
        help="Raw data root containing 300Torr/500Torr/700Torr folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/296K_raw_final_B"),
        help="Output directory inside the project.",
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=296.0,
        help="Temperature used for all 296 K raw datasets.",
    )
    parser.add_argument(
        "--pressures",
        type=float,
        nargs="+",
        default=[300.0, 500.0, 700.0],
        help="Pressure folders to process, in Torr.",
    )
    parser.add_argument(
        "--o2-dir-name",
        default="氧气",
        help="Subfolder name for O2 raw files.",
    )
    parser.add_argument(
        "--ar-dir-name",
        default="氩气",
        help="Subfolder name for Ar raw files.",
    )
    parser.add_argument(
        "--grid-min",
        type=float,
        default=9120.0,
        help="Minimum output wavenumber.",
    )
    parser.add_argument(
        "--grid-max",
        type=float,
        default=9820.0,
        help="Maximum output wavenumber.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.01,
        help="Output grid spacing in cm-1.",
    )
    parser.add_argument(
        "--filter-method",
        choices=("sigma_clip", "iqr"),
        default="sigma_clip",
        help="Within-file tau outlier removal method.",
    )
    parser.add_argument(
        "--filter-sigma",
        type=float,
        default=3.0,
        help="Sigma threshold for within-file tau outlier removal.",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR factor when --filter-method iqr is used.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=5,
        help="Minimum kept ring-down events required for one raw file.",
    )
    parser.add_argument(
        "--ar-fit-window",
        type=float,
        default=40.0,
        help="Ar Step 2 sliding-fit window in cm-1.",
    )
    parser.add_argument(
        "--ar-fit-step",
        type=float,
        default=5.0,
        help="Ar Step 2 sliding-fit step in cm-1.",
    )
    parser.add_argument(
        "--ar-fit-order",
        type=int,
        default=2,
        help="Ar Step 2 local polynomial order.",
    )
    parser.add_argument(
        "--ar-fit-sigma",
        type=float,
        default=4.0,
        help="Ar Step 2 robust-fit sigma threshold.",
    )
    parser.add_argument(
        "--ar-fit-smooth",
        type=float,
        default=20.0,
        help="Ar Step 2 smoothing width in cm-1.",
    )
    parser.add_argument(
        "--o2-fit-window",
        type=float,
        default=8.0,
        help="O2 Step 2 HITRAN-masked sliding-fit window in cm-1.",
    )
    parser.add_argument(
        "--o2-fit-step",
        type=float,
        default=1.0,
        help="O2 Step 2 HITRAN-masked sliding-fit step in cm-1.",
    )
    parser.add_argument(
        "--o2-fit-order",
        type=int,
        default=2,
        help="O2 Step 2 local polynomial order.",
    )
    parser.add_argument(
        "--o2-fit-sigma",
        type=float,
        default=2.0,
        help="O2 Step 2 robust-fit sigma threshold.",
    )
    parser.add_argument(
        "--o2-fit-smooth",
        type=float,
        default=2.0,
        help="O2 Step 2 smoothing width in cm-1.",
    )
    parser.add_argument(
        "--force-step1",
        action="store_true",
        help="Rebuild Step-1-like CSV files even if they already exist.",
    )
    parser.add_argument(
        "--force-fit",
        action="store_true",
        help="Rebuild Step 2 fit CSV files even if they already exist.",
    )
    return parser.parse_args()


def pressure_label(pressure_torr: float) -> str:
    if float(pressure_torr).is_integer():
        return f"{int(pressure_torr)}Torr"
    return f"{pressure_torr:g}Torr"


def parse_wavenumber(path: Path) -> float | None:
    match = RAW_NAME_RE.match(path.name)
    if not match:
        return None
    try:
        return float(match.group("wavenumber"))
    except (TypeError, ValueError):
        return None


def read_tau_first_column(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                value = float(parts[0])
            except ValueError:
                continue
            if np.isfinite(value) and value > 0:
                values.append(value)
    return np.asarray(values, dtype=float)


def summarize_raw_dir(
    source_dir: Path,
    pressure_torr: float,
    temperature_k: float,
    output_csv: Path,
    filter_method: str,
    filter_sigma: float,
    iqr_factor: float,
    min_events: int,
    force: bool,
) -> pd.DataFrame:
    if output_csv.exists() and not force:
        return pd.read_csv(output_csv)

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    rows: list[dict] = []
    skipped: list[dict] = []
    paths = sorted(source_dir.glob("*.txt"), key=lambda p: (parse_wavenumber(p) or np.inf, p.name))
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        if index % 5000 == 0 or index == total:
            print(f"  reading {source_dir.name}: {index}/{total}")
        wn = parse_wavenumber(path)
        if wn is None:
            skipped.append({"file": str(path), "reason": "filename_parse_failed"})
            continue

        tau_raw = read_tau_first_column(path)
        if len(tau_raw) < min_events:
            skipped.append({"file": str(path), "reason": "too_few_raw_events"})
            continue

        tau_kept = filter_ringdown_times(
            tau_raw,
            method=filter_method,
            sigma=filter_sigma,
            iqr_factor=iqr_factor,
        )
        tau_kept = tau_kept[np.isfinite(tau_kept) & (tau_kept > 0)]
        if len(tau_kept) < min_events:
            skipped.append({"file": str(path), "reason": "too_few_kept_events"})
            continue

        tau_mean = float(np.mean(tau_kept))
        tau_std = float(np.std(tau_kept, ddof=1)) if len(tau_kept) > 1 else 0.0
        rows.append(
            {
                "wavenumber": wn,
                "tau_us": tau_mean,
                "tau_stats_us": tau_std / np.sqrt(len(tau_kept)) if len(tau_kept) else np.nan,
                "tau_std_us": tau_std,
                "n_raw": int(len(tau_raw)),
                "n_kept": int(len(tau_kept)),
                "n_removed": int(len(tau_raw) - len(tau_kept)),
                "removed_fraction": float((len(tau_raw) - len(tau_kept)) / len(tau_raw)),
                "pressure_torr": float(pressure_torr),
                "temperature_c": float(temperature_k - 273.15),
                "source_file": str(path),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid raw files in {source_dir}")

    df = pd.DataFrame(rows).sort_values("wavenumber")
    numeric_cols = [
        "tau_us",
        "tau_stats_us",
        "tau_std_us",
        "n_raw",
        "n_kept",
        "n_removed",
        "removed_fraction",
        "pressure_torr",
        "temperature_c",
    ]
    grouped = (
        df.groupby("wavenumber", as_index=False)
        .agg(
            {
                "tau_us": "mean",
                "tau_stats_us": "mean",
                "tau_std_us": "mean",
                "n_raw": "sum",
                "n_kept": "sum",
                "n_removed": "sum",
                "removed_fraction": "mean",
                "pressure_torr": "mean",
                "temperature_c": "mean",
                "source_file": "first",
            }
        )
        .sort_values("wavenumber")
    )
    for col in numeric_cols:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
    grouped["loss_ppm_per_cm"] = TAU_US_TO_PPM_PER_CM / grouped["tau_us"].to_numpy(dtype=float)
    grouped["loss_stats_ppm_per_cm"] = (
        TAU_US_TO_PPM_PER_CM
        * np.abs(grouped["tau_stats_us"].to_numpy(dtype=float))
        / np.square(grouped["tau_us"].to_numpy(dtype=float))
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False, float_format="%.15g")
    if skipped:
        pd.DataFrame(skipped).to_csv(
            output_csv.with_name(output_csv.stem + "_skipped_files.csv"),
            index=False,
        )
    return grouped


def run_step2_fit(
    step1: pd.DataFrame,
    output_csv: Path,
    fit_mode: str,
    pressure: float,
    params: FitParams,
    force: bool,
) -> pd.DataFrame:
    if output_csv.exists() and not force:
        return pd.read_csv(output_csv)

    work = step1.copy()
    work = work[np.isfinite(work["wavenumber"]) & np.isfinite(work["tau_us"]) & (work["tau_us"] > 0)]
    work = work.sort_values("wavenumber")
    fitted = _add_step2_fit(
        work=work,
        fit_mode=fit_mode,
        pressure_label=f"{'O2' if fit_mode == 'o2' else 'Ar'} {pressure_label(pressure)}",
        window_cm1=params.window,
        step_cm1=params.step,
        order=params.order,
        sigma=params.sigma,
        smooth_cm1=params.smooth,
        hitran_align=False,
        hitran_align_max_shift_cm1=0.05,
        hitran_align_step_cm1=0.001,
        hitran_align_threshold_ratio=0.03,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(output_csv, index=False, float_format="%.15g")
    return fitted


def grid_values(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    valid_min: float,
    valid_max: float,
) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    out = np.full_like(grid, np.nan, dtype=float)
    inside = (grid >= valid_min) & (grid <= valid_max)
    if int(mask.sum()) < 2 or not inside.any():
        return out
    order = np.argsort(x[mask])
    out[inside] = np.interp(grid[inside], x[mask][order], y[mask][order])
    return out


def calculate_pressure_b(
    pressure: float,
    ar_fit: pd.DataFrame,
    o2_fit: pd.DataFrame,
    grid: np.ndarray,
    temperature_k: float,
    output_csv: Path,
) -> pd.DataFrame:
    ar_wn = ar_fit["wavenumber"].to_numpy(dtype=float)
    o2_wn = o2_fit["wavenumber"].to_numpy(dtype=float)
    valid_min = max(float(np.nanmin(ar_wn)), float(np.nanmin(o2_wn)), float(grid[0]))
    valid_max = min(float(np.nanmax(ar_wn)), float(np.nanmax(o2_wn)), float(grid[-1]))

    ar_loss = grid_values(
        ar_wn,
        ar_fit["loss_fit_ppm_per_cm"].to_numpy(dtype=float),
        grid,
        valid_min,
        valid_max,
    )
    o2_loss_col = (
        "cia_baseline_loss_ppm_per_cm"
        if "cia_baseline_loss_ppm_per_cm" in o2_fit.columns
        else "loss_fit_ppm_per_cm"
    )
    o2_loss = grid_values(
        o2_wn,
        o2_fit[o2_loss_col].to_numpy(dtype=float),
        grid,
        valid_min,
        valid_max,
    )
    ar_tau = grid_values(
        ar_wn,
        ar_fit["tau_fit_us"].to_numpy(dtype=float),
        grid,
        valid_min,
        valid_max,
    )
    o2_tau = grid_values(
        o2_wn,
        o2_fit["tau_fit_us"].to_numpy(dtype=float),
        grid,
        valid_min,
        valid_max,
    )
    alpha = (o2_loss - ar_loss) * 1e-6
    rho_amagat = (pressure / 760.0) * (273.15 / temperature_k)
    b_coeff = alpha / rho_amagat**2

    out = pd.DataFrame(
        {
            "wavenumber": grid,
            "pressure_torr": float(pressure),
            "temperature_k": float(temperature_k),
            "rho_amagat": float(rho_amagat),
            "ar_tau_fit_us": ar_tau,
            "o2_tau_fit_us": o2_tau,
            "ar_loss_fit_ppm_per_cm": ar_loss,
            "o2_cia_baseline_loss_ppm_per_cm": o2_loss,
            "alpha_cm_inv": alpha,
            "B_cm_inv_amagat_neg2": b_coeff,
        }
    )
    out = out[np.isfinite(out["B_cm_inv_amagat_neg2"])].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, float_format="%.15g")
    return out


def plot_fit_pair(ar_fit: pd.DataFrame, o2_fit: pd.DataFrame, pressure: float, plot_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for df, color, label in ((ar_fit, "tab:blue", "Ar"), (o2_fit, "tab:red", "O2")):
        axes[0].plot(df["wavenumber"], df["tau_us"], ".", ms=1.2, alpha=0.28, color=color)
        axes[0].plot(df["wavenumber"], df["tau_fit_us"], "-", lw=1.5, color=color, label=label)
        loss_col = "cia_baseline_loss_ppm_per_cm" if label == "O2" and "cia_baseline_loss_ppm_per_cm" in df.columns else "loss_fit_ppm_per_cm"
        axes[1].plot(df["wavenumber"], df[loss_col], "-", lw=1.5, color=color, label=label)
    axes[0].set_ylabel("tau (us)")
    axes[0].set_title(f"296 K, {pressure_label(pressure)} Step 2 fit")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Loss baseline (ppm/cm)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def plot_b_spectra(results: dict[str, pd.DataFrame], plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for label, df in results.items():
        ax.plot(df["wavenumber"], df["B_cm_inv_amagat_neg2"], lw=1.4, label=label)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("B (cm$^{-1}$ amagat$^{-2}$)")
    ax.set_title("296 K O$_2$-O$_2$ binary CIA coefficient")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ar_params = FitParams(
        args.ar_fit_window,
        args.ar_fit_step,
        args.ar_fit_order,
        args.ar_fit_sigma,
        args.ar_fit_smooth,
    )
    o2_params = FitParams(
        args.o2_fit_window,
        args.o2_fit_step,
        args.o2_fit_order,
        args.o2_fit_sigma,
        args.o2_fit_smooth,
    )
    grid = np.round(
        np.arange(args.grid_min, args.grid_max + args.grid_step / 2.0, args.grid_step),
        8,
    )

    b_results: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []
    for pressure in args.pressures:
        label = pressure_label(pressure)
        pressure_dir = raw_root / label
        print(f"\n=== {label} ===")
        ar_step1 = summarize_raw_dir(
            source_dir=pressure_dir / args.ar_dir_name,
            pressure_torr=pressure,
            temperature_k=args.temperature_k,
            output_csv=output_dir / label / "Ar_step1_ringdown_summary.csv",
            filter_method=args.filter_method,
            filter_sigma=args.filter_sigma,
            iqr_factor=args.iqr_factor,
            min_events=args.min_events,
            force=args.force_step1,
        )
        o2_step1 = summarize_raw_dir(
            source_dir=pressure_dir / args.o2_dir_name,
            pressure_torr=pressure,
            temperature_k=args.temperature_k,
            output_csv=output_dir / label / "O2_step1_ringdown_summary.csv",
            filter_method=args.filter_method,
            filter_sigma=args.filter_sigma,
            iqr_factor=args.iqr_factor,
            min_events=args.min_events,
            force=args.force_step1,
        )
        print(f"  Step1 points: Ar={len(ar_step1)}, O2={len(o2_step1)}")

        ar_fit = run_step2_fit(
            step1=ar_step1,
            output_csv=output_dir / label / "Ar_step2_fit.csv",
            fit_mode="ar",
            pressure=pressure,
            params=ar_params,
            force=args.force_fit,
        )
        print(f"  Ar Step2 fit: {output_dir / label / 'Ar_step2_fit.csv'}")
        o2_fit = run_step2_fit(
            step1=o2_step1,
            output_csv=output_dir / label / "O2_step2_fit.csv",
            fit_mode="o2",
            pressure=pressure,
            params=o2_params,
            force=args.force_fit,
        )
        print(f"  O2 Step2 fit: {output_dir / label / 'O2_step2_fit.csv'}")

        b_df = calculate_pressure_b(
            pressure=pressure,
            ar_fit=ar_fit,
            o2_fit=o2_fit,
            grid=grid,
            temperature_k=args.temperature_k,
            output_csv=output_dir / label / f"296K_{label}_B.csv",
        )
        b_results[label] = b_df
        plot_fit_pair(ar_fit, o2_fit, pressure, output_dir / label / f"296K_{label}_step2_fit.png")
        summaries.append(
            {
                "pressure_torr": pressure,
                "temperature_k": args.temperature_k,
                "n_ar_step1": len(ar_step1),
                "n_o2_step1": len(o2_step1),
                "n_b_points": len(b_df),
                "wavenumber_min": float(b_df["wavenumber"].min()),
                "wavenumber_max": float(b_df["wavenumber"].max()),
                "B_mean": float(b_df["B_cm_inv_amagat_neg2"].mean()),
                "B_min": float(b_df["B_cm_inv_amagat_neg2"].min()),
                "B_max": float(b_df["B_cm_inv_amagat_neg2"].max()),
                "rho_amagat": float(b_df["rho_amagat"].iloc[0]),
            }
        )
        print(f"  B output: {output_dir / label / f'296K_{label}_B.csv'}")

    wide = pd.DataFrame({"wavenumber": grid})
    for label, df in b_results.items():
        wide[f"296K_{label}_B"] = np.interp(
            grid,
            df["wavenumber"].to_numpy(dtype=float),
            df["B_cm_inv_amagat_neg2"].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
    wide_path = output_dir / "296K_B_three_pressures_wide.csv"
    wide.to_csv(wide_path, index=False, float_format="%.15g")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "296K_B_processing_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.15g")
    plot_b_spectra(b_results, output_dir / "296K_B_three_pressures.png")

    print("\nDone.")
    print(f"Wide B table: {wide_path}")
    print(f"Summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
