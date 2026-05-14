#!/usr/bin/env python3
"""Plot existing LBLRTM and Karman temperature-dependence coefficients.

This script does not recalculate the coefficients from raw simulations.  It
uses the previously generated CSV files:

    output/results/analysis/LBLRTM_continuum_temperature/lblrtm_temperature_dependence_fit.csv
    output/results/analysis/Karman_temperature_linearity/karman_temperature_linearity_fit.csv

The two panels are plotted separately because the coefficient ranges are very
different for LBLRTM and Karman.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LBLRTM_CSV = Path(
    "output/results/analysis/LBLRTM_continuum_temperature/"
    "lblrtm_temperature_dependence_fit.csv"
)
DEFAULT_KARMAN_CSV = Path(
    "output/results/analysis/Karman_temperature_linearity/"
    "karman_temperature_linearity_fit.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/results/analysis/LBLRTM_Karman_temperature_dependence_comparison"
)

LBLRTM_SLOPE_COL = "dB_dT_cm_inv_amagat_neg2_K_neg1"
KARMAN_SLOPE_COL = "linear_dB_dT_cm_inv_amagat_neg2_K_neg1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot side-by-side LBLRTM and Karman dB/dT from existing CSV outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lblrtm-csv", type=Path, default=DEFAULT_LBLRTM_CSV)
    parser.add_argument("--karman-csv", type=Path, default=DEFAULT_KARMAN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-width-mm", type=float, default=190.0)
    parser.add_argument("--figure-height-mm", type=float, default=78.0)
    parser.add_argument("--dpi", type=int, default=500)
    return parser.parse_args()


def read_curve(path: Path, slope_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"wavenumber", slope_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    out = df[["wavenumber", slope_col]].copy()
    out.columns = ["wavenumber", "dB_dT_cm_inv_amagat_neg2_K_neg1"]
    out["wavenumber"] = pd.to_numeric(out["wavenumber"], errors="coerce")
    out["dB_dT_cm_inv_amagat_neg2_K_neg1"] = pd.to_numeric(
        out["dB_dT_cm_inv_amagat_neg2_K_neg1"], errors="coerce"
    )
    out = out.dropna().sort_values("wavenumber").reset_index(drop=True)
    return out


def add_padded_limits(ax, x: np.ndarray, y_scaled: np.ndarray, y_pad_fraction: float = 0.08) -> None:
    x = x[np.isfinite(x)]
    y_scaled = y_scaled[np.isfinite(y_scaled)]
    if len(x):
        x_pad = max((float(np.max(x)) - float(np.min(x))) * 0.02, 1.0)
        ax.set_xlim(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
    if len(y_scaled):
        y_min = float(np.min(y_scaled))
        y_max = float(np.max(y_scaled))
        y_pad = max((y_max - y_min) * y_pad_fraction, 0.002)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


def save_plotting_data(lblrtm: pd.DataFrame, karman: pd.DataFrame, output_dir: Path) -> Path:
    n = max(len(lblrtm), len(karman))
    table = pd.DataFrame(
        {
            "LBLRTM_wavenumber": np.nan,
            "LBLRTM_dB_dT_cm_inv_amagat_neg2_K_neg1": np.nan,
            "LBLRTM_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1": np.nan,
            "Karman_wavenumber": np.nan,
            "Karman_dB_dT_cm_inv_amagat_neg2_K_neg1": np.nan,
            "Karman_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1": np.nan,
        },
        index=np.arange(n),
    )
    table.loc[: len(lblrtm) - 1, "LBLRTM_wavenumber"] = lblrtm["wavenumber"].to_numpy(dtype=float)
    table.loc[
        : len(lblrtm) - 1, "LBLRTM_dB_dT_cm_inv_amagat_neg2_K_neg1"
    ] = lblrtm["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)
    table.loc[
        : len(lblrtm) - 1, "LBLRTM_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1"
    ] = lblrtm["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9

    table.loc[: len(karman) - 1, "Karman_wavenumber"] = karman["wavenumber"].to_numpy(dtype=float)
    table.loc[
        : len(karman) - 1, "Karman_dB_dT_cm_inv_amagat_neg2_K_neg1"
    ] = karman["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)
    table.loc[
        : len(karman) - 1, "Karman_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1"
    ] = karman["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9

    out_csv = output_dir / "lblrtm_karman_temperature_dependence_plot_data.csv"
    table.to_csv(out_csv, index=False, float_format="%.12g")
    return out_csv


def plot_side_by_side(
    lblrtm: pd.DataFrame,
    karman: pd.DataFrame,
    output_dir: Path,
    width_mm: float,
    height_mm: float,
    dpi: int,
) -> Path:
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
        1,
        2,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        sharey=False,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.89, wspace=0.24)

    curves = [
        (axes[0], lblrtm, "LBLRTM", "#D55E00"),
        (axes[1], karman, "Karman ab initio", "#0072B2"),
    ]
    for panel_label, (ax, df, title, color) in zip(("a", "b"), curves):
        x = df["wavenumber"].to_numpy(dtype=float)
        y = df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
        ax.plot(x, y, color=color, lw=1.0)
        ax.set_title(title, pad=4)
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.text(
            0.02,
            0.96,
            f"({panel_label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.minorticks_on()
        add_padded_limits(ax, x, y)

    axes[0].set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    axes[1].set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")

    out_png = output_dir / "lblrtm_karman_temperature_dependence_side_by_side.png"
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_png.with_suffix(".pdf"))
    fig.savefig(out_png.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)
    return out_png


def write_summary(lblrtm: pd.DataFrame, karman: pd.DataFrame, output_dir: Path) -> Path:
    rows = []
    for name, df in (("LBLRTM", lblrtm), ("Karman", karman)):
        y = df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)
        rows.append(
            {
                "source": name,
                "n_points": int(len(df)),
                "wavenumber_min": float(df["wavenumber"].min()),
                "wavenumber_max": float(df["wavenumber"].max()),
                "dB_dT_min_cm_inv_amagat_neg2_K_neg1": float(np.nanmin(y)),
                "dB_dT_max_cm_inv_amagat_neg2_K_neg1": float(np.nanmax(y)),
                "dB_dT_mean_cm_inv_amagat_neg2_K_neg1": float(np.nanmean(y)),
                "dB_dT_min_1e_neg9": float(np.nanmin(y) / 1e-9),
                "dB_dT_max_1e_neg9": float(np.nanmax(y) / 1e-9),
                "dB_dT_mean_1e_neg9": float(np.nanmean(y) / 1e-9),
            }
        )
    out_csv = output_dir / "lblrtm_karman_temperature_dependence_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.12g")
    return out_csv


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lblrtm = read_curve(args.lblrtm_csv.expanduser(), LBLRTM_SLOPE_COL)
    karman = read_curve(args.karman_csv.expanduser(), KARMAN_SLOPE_COL)

    data_csv = save_plotting_data(lblrtm, karman, output_dir)
    summary_csv = write_summary(lblrtm, karman, output_dir)
    figure_png = plot_side_by_side(
        lblrtm=lblrtm,
        karman=karman,
        output_dir=output_dir,
        width_mm=args.figure_width_mm,
        height_mm=args.figure_height_mm,
        dpi=args.dpi,
    )

    print(f"Figure: {figure_png}")
    print(f"Plot data: {data_csv}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
