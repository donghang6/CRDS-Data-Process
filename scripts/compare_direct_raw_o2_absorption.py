#!/usr/bin/env python3
"""Compare direct raw O2 absorption coefficient with the processed result.

This script intentionally bypasses the normal Step 1/Step 2 processing path.
It reads the original ring-down files directly, averages the raw ring-down
events at each file's wavenumber without outlier rejection, interpolates the raw
tau spectra onto a regular 0.01 cm^-1 grid, converts tau to total cavity loss,
subtracts the HITRAN2024 O2 line absorption from the O2 data, and subtracts the
matching Ar background:

    alpha_direct = loss_raw(O2) - loss_HITRAN(O2 lines) - loss_raw(Ar)

The processed comparison is built at the same level from the existing Step 2
outputs:

    alpha_processed = baseline_step2(O2) - baseline_step2(Ar)

Typical usage:

    python scripts/compare_direct_raw_o2_absorption.py \
        --dataset 'CIA/303K/O2 500Torr'

Scan all O2 datasets under data/raw/CIA and rank them:

    python scripts/compare_direct_raw_o2_absorption.py --scan-all

When measured and HITRAN line positions have a small mismatch, add
``--hitran-align``.  The script then searches a small wavenumber shift before
subtracting HITRAN:

    python scripts/compare_direct_raw_o2_absorption.py \
        --dataset 'CIA/303K/O2 500Torr' --hitran-align

Outputs for each dataset:

    direct_raw_vs_processed_o2_absorption.csv
        Regular-grid direct raw absorption coefficient and processed
        absorption coefficient, in ppm/cm and cm^-1.

    direct_raw_vs_processed_o2_absorption_summary.csv
        One-row summary with coverage and residual metrics.

    direct_raw_vs_processed_o2_absorption.png/pdf/tif
        Upper panel: direct raw-HITRAN-Ar and processed O2-Ar absorption.
        Lower panel: direct - processed difference.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crds_process.continuum.analysis import (
    TAU_US_TO_PPM_PER_CM,
    _align_hitran_to_measured_loss,
    _simulate_o2_hitran_loss_ppm_per_cm,
)
from crds_process.io.readers import parse_filename


DEFAULT_DATASET = "CIA/303K/O2 500Torr"
DEFAULT_OUTPUT_ROOT = Path("output/results/analysis/direct_raw_o2_absorption_comparison")


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    ar_dataset: str
    raw_dir: Path
    ar_raw_dir: Path
    processed_csv: Path
    ar_processed_csv: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly interpolate raw O2 tau and compare line-subtracted absorption with Step 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset label, e.g. 'CIA/303K/O2 500Torr'. Ignored when --scan-all is used.",
    )
    parser.add_argument("--scan-all", action="store_true", help="Process all O2 datasets under raw root.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Raw data root.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("output/results/continuum"),
        help="Processed continuum result root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root.")
    parser.add_argument("--grid-start", type=float, default=9120.0, help="Regular grid start wavenumber.")
    parser.add_argument("--grid-end", type=float, default=9820.0, help="Regular grid end wavenumber.")
    parser.add_argument("--grid-step", type=float, default=0.01, help="Regular grid spacing.")
    parser.add_argument(
        "--processed-column",
        default="auto",
        help=(
            "Processed absorption/loss column. 'auto' uses cia_baseline_loss_ppm_per_cm "
            "then loss_fit_ppm_per_cm."
        ),
    )
    parser.add_argument("--dpi", type=int, default=500, help="Figure DPI.")
    parser.add_argument("--figure-width-mm", type=float, default=190.0, help="Figure width.")
    parser.add_argument("--figure-height-mm", type=float, default=120.0, help="Figure height.")
    parser.add_argument(
        "--hitran-align",
        action="store_true",
        help="Shift HITRAN O2 line positions to match measured O2 loss before subtraction.",
    )
    parser.add_argument(
        "--hitran-align-max-shift",
        type=float,
        default=0.05,
        help="Maximum absolute HITRAN alignment shift in cm^-1.",
    )
    parser.add_argument(
        "--hitran-align-step",
        type=float,
        default=0.001,
        help="HITRAN alignment search step in cm^-1.",
    )
    parser.add_argument(
        "--hitran-align-threshold",
        type=float,
        default=0.03,
        help="Relative HITRAN peak threshold used for alignment.",
    )
    return parser.parse_args()


def normalize_dataset_label(label: str) -> str:
    label = str(label).strip().strip("/")
    if not label.startswith("CIA/"):
        label = f"CIA/{label}"
    return label


def discover_o2_datasets(raw_root: Path, processed_root: Path, output_root: Path) -> list[DatasetPaths]:
    cia_root = raw_root / "CIA"
    datasets: list[DatasetPaths] = []
    for raw_dir in sorted(cia_root.glob("*K/O2 *Torr")):
        label = f"CIA/{raw_dir.relative_to(cia_root)}"
        processed_csv = processed_root / label / "continuum_step2_fit.csv"
        paths = make_dataset_paths(label, raw_root, processed_root, output_root)
        if processed_csv.exists() and paths.ar_raw_dir.exists() and paths.ar_processed_csv.exists():
            datasets.append(paths)
    return datasets


def matching_ar_dataset_label(dataset: str) -> str:
    dataset = normalize_dataset_label(dataset)
    parts = Path(dataset).parts
    if len(parts) < 3:
        raise ValueError(f"Dataset label is too short: {dataset}")
    pressure_part = parts[-1]
    ar_pressure_part = re.sub(r"^O2\b", "Ar", pressure_part, flags=re.IGNORECASE)
    if ar_pressure_part == pressure_part:
        ar_pressure_part = pressure_part.replace("O₂", "Ar")
    return str(Path(*parts[:-1], ar_pressure_part))


def make_dataset_paths(
    dataset_label: str,
    raw_root: Path,
    processed_root: Path,
    output_root: Path,
) -> DatasetPaths:
    dataset = normalize_dataset_label(dataset_label)
    ar_dataset = matching_ar_dataset_label(dataset)
    raw_dir = raw_root / dataset
    ar_raw_dir = raw_root / ar_dataset
    processed_csv = processed_root / dataset / "continuum_step2_fit.csv"
    ar_processed_csv = processed_root / ar_dataset / "continuum_step2_fit.csv"
    output_dir = output_root / Path(dataset)
    return DatasetPaths(
        dataset=dataset,
        ar_dataset=ar_dataset,
        raw_dir=raw_dir,
        ar_raw_dir=ar_raw_dir,
        processed_csv=processed_csv,
        ar_processed_csv=ar_processed_csv,
        output_dir=output_dir,
    )


def read_raw_directory(raw_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    skipped = 0
    for path in sorted(raw_dir.glob("*.txt")):
        try:
            meta = parse_filename(path)
            data = np.loadtxt(path, usecols=(0, 2, 3))
            if data.ndim == 1:
                data = data.reshape(1, -1)
            tau = data[:, 0].astype(float)
            temperature_c = data[:, 1].astype(float)
            pressure_torr = data[:, 2].astype(float)
            good_tau = np.isfinite(tau) & (tau > 0)
            if int(good_tau.sum()) < 1:
                skipped += 1
                continue
            event_loss = TAU_US_TO_PPM_PER_CM / tau[good_tau]
            records.append(
                {
                    "wavenumber": meta.wavenumber,
                    "raw_file": str(path),
                    "n_raw_events": int(len(tau)),
                    "n_valid_tau": int(good_tau.sum()),
                    "tau_raw_mean_us": float(np.mean(tau[good_tau])),
                    "tau_raw_median_us": float(np.median(tau[good_tau])),
                    "loss_event_mean_ppm_per_cm": float(np.mean(event_loss)),
                    "temperature_c_mean": float(np.nanmean(temperature_c)),
                    "pressure_torr_mean": float(np.nanmean(pressure_torr)),
                }
            )
        except Exception:
            skipped += 1

    if not records:
        raise ValueError(f"No readable raw files found in {raw_dir}")

    df = pd.DataFrame(records).sort_values("wavenumber")
    grouped = (
        df.groupby("wavenumber", as_index=False)
        .agg(
            {
                "n_raw_events": "sum",
                "n_valid_tau": "sum",
                "tau_raw_mean_us": "mean",
                "tau_raw_median_us": "mean",
                "loss_event_mean_ppm_per_cm": "mean",
                "temperature_c_mean": "mean",
                "pressure_torr_mean": "mean",
            }
        )
        .sort_values("wavenumber")
    )
    grouped.attrs["n_files_read"] = int(len(df))
    grouped.attrs["n_files_skipped"] = int(skipped)
    return grouped


def regular_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("grid-step must be positive")
    if end < start:
        start, end = end, start
    n = int(np.floor((end - start) / step + 0.5)) + 1
    return np.round(start + np.arange(n, dtype=float) * step, 8)


def interpolate_column(source_x: np.ndarray, source_y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(source_x) & np.isfinite(source_y)
    if int(mask.sum()) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(source_x[mask])
    x = source_x[mask][order]
    y = source_y[mask][order]
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def processed_column_name(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Processed CSV is missing requested column: {requested}")
        return requested
    for candidate in ("cia_baseline_loss_ppm_per_cm", "loss_fit_ppm_per_cm"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Processed CSV must contain cia_baseline_loss_ppm_per_cm or loss_fit_ppm_per_cm")


def pressure_from_label(dataset: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Torr", dataset, flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def process_dataset(
    paths: DatasetPaths,
    grid_start: float,
    grid_end: float,
    grid_step: float,
    processed_column: str,
    dpi: int,
    figure_width_mm: float,
    figure_height_mm: float,
    hitran_align: bool,
    hitran_align_max_shift: float,
    hitran_align_step: float,
    hitran_align_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    if not paths.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {paths.raw_dir}")
    if not paths.ar_raw_dir.exists():
        raise FileNotFoundError(f"Matching Ar raw directory not found: {paths.ar_raw_dir}")
    if not paths.processed_csv.exists():
        raise FileNotFoundError(f"Processed CSV not found: {paths.processed_csv}")
    if not paths.ar_processed_csv.exists():
        raise FileNotFoundError(f"Matching Ar processed CSV not found: {paths.ar_processed_csv}")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    raw_o2 = read_raw_directory(paths.raw_dir)
    raw_ar = read_raw_directory(paths.ar_raw_dir)
    processed_o2 = pd.read_csv(paths.processed_csv)
    processed_ar = pd.read_csv(paths.ar_processed_csv)
    o2_proc_col = processed_column_name(processed_o2, processed_column)
    ar_proc_col = processed_column_name(processed_ar, processed_column)

    raw_o2_x = raw_o2["wavenumber"].to_numpy(dtype=float)
    raw_ar_x = raw_ar["wavenumber"].to_numpy(dtype=float)
    grid = regular_grid(grid_start, grid_end, grid_step)
    processed_o2_x = processed_o2["wavenumber"].to_numpy(dtype=float)
    processed_ar_x = processed_ar["wavenumber"].to_numpy(dtype=float)
    overlap = (
        (grid >= np.nanmin(raw_o2_x))
        & (grid <= np.nanmax(raw_o2_x))
        & (grid >= np.nanmin(raw_ar_x))
        & (grid <= np.nanmax(raw_ar_x))
        & (grid >= np.nanmin(processed_o2_x))
        & (grid <= np.nanmax(processed_o2_x))
        & (grid >= np.nanmin(processed_ar_x))
        & (grid <= np.nanmax(processed_ar_x))
    )
    grid = grid[overlap]

    o2_tau_grid = interpolate_column(raw_o2_x, raw_o2["tau_raw_mean_us"].to_numpy(dtype=float), grid)
    ar_tau_grid = interpolate_column(raw_ar_x, raw_ar["tau_raw_mean_us"].to_numpy(dtype=float), grid)
    o2_raw_total_loss = TAU_US_TO_PPM_PER_CM / o2_tau_grid
    ar_raw_total_loss = TAU_US_TO_PPM_PER_CM / ar_tau_grid
    o2_raw_event_loss = interpolate_column(
        raw_o2_x, raw_o2["loss_event_mean_ppm_per_cm"].to_numpy(dtype=float), grid
    )
    ar_raw_event_loss = interpolate_column(
        raw_ar_x, raw_ar["loss_event_mean_ppm_per_cm"].to_numpy(dtype=float), grid
    )
    temperature_grid = interpolate_column(raw_o2_x, raw_o2["temperature_c_mean"].to_numpy(dtype=float), grid)
    pressure_grid = interpolate_column(raw_o2_x, raw_o2["pressure_torr_mean"].to_numpy(dtype=float), grid)
    ar_pressure_grid = interpolate_column(raw_ar_x, raw_ar["pressure_torr_mean"].to_numpy(dtype=float), grid)
    temperature_c = float(np.nanmedian(temperature_grid))
    pressure_torr = float(np.nanmedian(pressure_grid))
    ar_pressure_torr = float(np.nanmedian(ar_pressure_grid))
    if not np.isfinite(pressure_torr) or pressure_torr <= 0:
        pressure_torr = pressure_from_label(paths.dataset)

    hitran_loss_unaligned = _simulate_o2_hitran_loss_ppm_per_cm(
        wavenumber=grid,
        temperature_c=temperature_c,
        pressure_torr=pressure_torr,
    )
    if hitran_align:
        hitran_loss, hitran_alignment = _align_hitran_to_measured_loss(
            wavenumber=grid,
            measured_loss=o2_raw_total_loss,
            hitran_loss=hitran_loss_unaligned,
            max_shift_cm1=hitran_align_max_shift,
            step_cm1=hitran_align_step,
            threshold_ratio=hitran_align_threshold,
        )
    else:
        hitran_loss = hitran_loss_unaligned
        hitran_alignment = {
            "enabled": False,
            "shift_cm1": 0.0,
            "score_ppm_per_cm": np.nan,
            "scale": np.nan,
            "used_points": 0,
        }
    direct_o2_minus_hitran = o2_raw_total_loss - hitran_loss
    direct_event_o2_minus_hitran = o2_raw_event_loss - hitran_loss
    direct_absorption = direct_o2_minus_hitran - ar_raw_total_loss
    direct_event_mean_absorption = direct_event_o2_minus_hitran - ar_raw_event_loss

    o2_proc_x = pd.to_numeric(processed_o2["wavenumber"], errors="coerce").to_numpy(dtype=float)
    ar_proc_x = pd.to_numeric(processed_ar["wavenumber"], errors="coerce").to_numpy(dtype=float)
    processed_o2_loss = interpolate_column(
        o2_proc_x,
        pd.to_numeric(processed_o2[o2_proc_col], errors="coerce").to_numpy(dtype=float),
        grid,
    )
    processed_ar_loss = interpolate_column(
        ar_proc_x,
        pd.to_numeric(processed_ar[ar_proc_col], errors="coerce").to_numpy(dtype=float),
        grid,
    )
    processed_absorption = processed_o2_loss - processed_ar_loss
    if "hitran_o2_loss_ppm_per_cm" in processed_o2.columns:
        processed_hitran_loss = interpolate_column(
            o2_proc_x,
            pd.to_numeric(processed_o2["hitran_o2_loss_ppm_per_cm"], errors="coerce").to_numpy(dtype=float),
            grid,
        )
    else:
        processed_hitran_loss = np.full_like(grid, np.nan, dtype=float)

    diff = direct_absorption - processed_absorption
    relative = diff / processed_absorption * 100.0
    out = pd.DataFrame(
        {
            "wavenumber": grid,
            "o2_direct_tau_raw_mean_us": o2_tau_grid,
            "ar_direct_tau_raw_mean_us": ar_tau_grid,
            "o2_direct_total_loss_from_tau_mean_ppm_per_cm": o2_raw_total_loss,
            "ar_direct_total_loss_from_tau_mean_ppm_per_cm": ar_raw_total_loss,
            "o2_direct_total_loss_from_event_mean_ppm_per_cm": o2_raw_event_loss,
            "ar_direct_total_loss_from_event_mean_ppm_per_cm": ar_raw_event_loss,
            "hitran_o2_line_loss_unaligned_ppm_per_cm": hitran_loss_unaligned,
            "hitran_o2_line_loss_ppm_per_cm": hitran_loss,
            "hitran_o2_shift_cm1": float(hitran_alignment["shift_cm1"]),
            "hitran_o2_alignment_enabled": bool(hitran_alignment["enabled"]),
            "hitran_o2_alignment_score_ppm_per_cm": float(hitran_alignment["score_ppm_per_cm"]),
            "hitran_o2_alignment_scale": float(hitran_alignment["scale"]),
            "hitran_o2_alignment_used_points": int(hitran_alignment["used_points"]),
            "o2_direct_line_subtracted_loss_ppm_per_cm": direct_o2_minus_hitran,
            "direct_absorption_after_ar_ppm_per_cm": direct_absorption,
            "direct_event_mean_absorption_after_ar_ppm_per_cm": direct_event_mean_absorption,
            "o2_processed_column": o2_proc_col,
            "ar_processed_column": ar_proc_col,
            "o2_processed_line_removed_loss_ppm_per_cm": processed_o2_loss,
            "ar_processed_loss_ppm_per_cm": processed_ar_loss,
            "processed_absorption_after_ar_ppm_per_cm": processed_absorption,
            "processed_hitran_o2_line_loss_ppm_per_cm": processed_hitran_loss,
            "difference_direct_minus_processed_ppm_per_cm": diff,
            "relative_difference_percent": relative,
            "direct_absorption_after_ar_alpha_cm_inv": direct_absorption / 1e6,
            "processed_absorption_after_ar_alpha_cm_inv": processed_absorption / 1e6,
            "difference_alpha_cm_inv": diff / 1e6,
            "temperature_c_for_hitran": temperature_c,
            "pressure_torr_for_hitran": pressure_torr,
            "ar_pressure_torr_median": ar_pressure_torr,
        }
    )

    valid = np.isfinite(diff) & np.isfinite(processed_absorption)
    summary = {
        "dataset": paths.dataset,
        "ar_dataset": paths.ar_dataset,
        "raw_dir": str(paths.raw_dir),
        "ar_raw_dir": str(paths.ar_raw_dir),
        "processed_csv": str(paths.processed_csv),
        "ar_processed_csv": str(paths.ar_processed_csv),
        "o2_processed_column": o2_proc_col,
        "ar_processed_column": ar_proc_col,
        "o2_n_files_read": raw_o2.attrs.get("n_files_read", len(raw_o2)),
        "o2_n_files_skipped": raw_o2.attrs.get("n_files_skipped", 0),
        "ar_n_files_read": raw_ar.attrs.get("n_files_read", len(raw_ar)),
        "ar_n_files_skipped": raw_ar.attrs.get("n_files_skipped", 0),
        "o2_raw_unique_wavenumbers": int(len(raw_o2)),
        "ar_raw_unique_wavenumbers": int(len(raw_ar)),
        "grid_step_cm1": grid_step,
        "grid_points": int(len(out)),
        "wavenumber_min": float(np.nanmin(grid)),
        "wavenumber_max": float(np.nanmax(grid)),
        "temperature_c_for_hitran": temperature_c,
        "pressure_torr_for_hitran": pressure_torr,
        "ar_pressure_torr_median": ar_pressure_torr,
        "hitran_o2_alignment_enabled": bool(hitran_alignment["enabled"]),
        "hitran_o2_shift_cm1": float(hitran_alignment["shift_cm1"]),
        "hitran_o2_alignment_score_ppm_per_cm": float(hitran_alignment["score_ppm_per_cm"]),
        "hitran_o2_alignment_scale": float(hitran_alignment["scale"]),
        "hitran_o2_alignment_used_points": int(hitran_alignment["used_points"]),
        "direct_absorption_mean_ppm_per_cm": float(np.nanmean(direct_absorption)),
        "processed_absorption_mean_ppm_per_cm": float(np.nanmean(processed_absorption)),
        "difference_mean_ppm_per_cm": float(np.nanmean(diff)),
        "difference_median_ppm_per_cm": float(np.nanmedian(diff)),
        "difference_std_ppm_per_cm": float(np.nanstd(diff[valid], ddof=1)) if int(valid.sum()) > 1 else np.nan,
        "difference_rmse_ppm_per_cm": float(np.sqrt(np.nanmean(np.square(diff)))),
        "abs_relative_difference_median_percent": float(np.nanmedian(np.abs(relative))),
        "abs_relative_difference_mean_percent": float(np.nanmean(np.abs(relative))),
        "relative_difference_std_percent": float(np.nanstd(relative[np.isfinite(relative)], ddof=1)),
        "max_abs_relative_difference_percent": float(np.nanmax(np.abs(relative))),
    }

    out_csv = paths.output_dir / "direct_raw_vs_processed_o2_absorption.csv"
    summary_csv = paths.output_dir / "direct_raw_vs_processed_o2_absorption_summary.csv"
    out.to_csv(out_csv, index=False, float_format="%.15g")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, float_format="%.15g")
    plot_comparison(out, paths.output_dir / "direct_raw_vs_processed_o2_absorption.png", dpi, figure_width_mm, figure_height_mm)
    return out, summary


def plot_comparison(
    df: pd.DataFrame,
    output_png: Path,
    dpi: int,
    figure_width_mm: float,
    figure_height_mm: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width_mm / 25.4, figure_height_mm / 25.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.975, hspace=0.08)
    x = df["wavenumber"].to_numpy(dtype=float)
    direct = df["direct_absorption_after_ar_alpha_cm_inv"].to_numpy(dtype=float) / 1e-8
    processed = df["processed_absorption_after_ar_alpha_cm_inv"].to_numpy(dtype=float) / 1e-8
    diff = df["difference_alpha_cm_inv"].to_numpy(dtype=float) / 1e-9

    axes[0].plot(x, direct, color="#666666", lw=0.45, alpha=0.85, label="Direct raw - HITRAN - Ar")
    axes[0].plot(x, processed, color="#0072B2", lw=0.85, label="Processed O$_2$ - Ar")
    axes[0].set_ylabel(r"Absorption ($10^{-8}$ cm$^{-1}$)")
    axes[0].legend(frameon=False, loc="best")
    axes[0].minorticks_on()

    axes[1].axhline(0, color="black", lw=0.55)
    axes[1].plot(x, diff, color="black", lw=0.55)
    axes[1].set_ylabel(r"Difference ($10^{-9}$ cm$^{-1}$)")
    axes[1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[1].minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    fig.savefig(output_png.with_suffix(".pdf"))
    fig.savefig(output_png.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.scan_all:
        datasets = discover_o2_datasets(raw_root, processed_root, output_root)
    else:
        datasets = [make_dataset_paths(args.dataset, raw_root, processed_root, output_root)]
    if not datasets:
        raise SystemExit("No O2 datasets found.")

    summaries: list[dict] = []
    for paths in datasets:
        print(f"Processing {paths.dataset}")
        _, summary = process_dataset(
            paths=paths,
            grid_start=args.grid_start,
            grid_end=args.grid_end,
            grid_step=args.grid_step,
            processed_column=args.processed_column,
            dpi=args.dpi,
            figure_width_mm=args.figure_width_mm,
            figure_height_mm=args.figure_height_mm,
            hitran_align=args.hitran_align,
            hitran_align_max_shift=max(args.hitran_align_max_shift, 0.0),
            hitran_align_step=max(args.hitran_align_step, 0.0),
            hitran_align_threshold=max(args.hitran_align_threshold, 0.0),
        )
        summary["output_dir"] = str(paths.output_dir)
        summaries.append(summary)
        print(
            f"  RMSE={summary['difference_rmse_ppm_per_cm']:.6g} ppm/cm, "
            f"median |rel|={summary['abs_relative_difference_median_percent']:.4g}%"
        )

    summary_df = pd.DataFrame(summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    ranking_csv = output_root / "direct_raw_o2_absorption_comparison_summary.csv"
    summary_df.sort_values(
        ["difference_rmse_ppm_per_cm", "abs_relative_difference_median_percent"],
        ascending=True,
    ).to_csv(ranking_csv, index=False, float_format="%.15g")
    print(f"Summary ranking: {ranking_csv}")
    print(
        summary_df.sort_values(
            ["difference_rmse_ppm_per_cm", "abs_relative_difference_median_percent"],
            ascending=True,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
