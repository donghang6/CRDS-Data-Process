#!/usr/bin/env python3
"""Compare this work's temperature-dependence coefficient with Karman.

The comparison is restricted to the common wavenumber range of the measured
temperature-dependence result and the Karman ab initio calculation.  With the
current data this range is 9120-9660 cm^-1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = Path(
    "output/results/analysis/B_temperature_dependence_from_summary_303_combined_uB_x2/"
    "temperature_dependence_weighted_fit.csv"
)
DEFAULT_KARMAN = Path(
    "output/results/analysis/Karman_temperature_linearity/karman_temperature_linearity_fit.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/results/analysis/Experiment_vs_Karman_temperature_dependence_common_range"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot this work dB/dT against Karman dB/dT in the common range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--karman", type=Path, default=DEFAULT_KARMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-width-mm", type=float, default=140.0)
    parser.add_argument("--figure-height-mm", type=float, default=82.0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_inputs(experiment_path: Path, karman_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_cols = [
        "wavenumber",
        "dB_dT_cm_inv_amagat_neg2_K_neg1",
        "u_dB_dT_cm_inv_amagat_neg2_K_neg1",
    ]
    karman_cols = ["wavenumber", "linear_dB_dT_cm_inv_amagat_neg2_K_neg1"]
    experiment = pd.read_csv(experiment_path, usecols=exp_cols)
    karman = pd.read_csv(karman_path, usecols=karman_cols)
    return experiment, karman


def build_plot_data(experiment: pd.DataFrame, karman: pd.DataFrame) -> pd.DataFrame:
    exp_x = pd.to_numeric(experiment["wavenumber"], errors="coerce").to_numpy(dtype=float)
    kar_x = pd.to_numeric(karman["wavenumber"], errors="coerce").to_numpy(dtype=float)
    x_min = max(float(np.nanmin(exp_x)), float(np.nanmin(kar_x)))
    x_max = min(float(np.nanmax(exp_x)), float(np.nanmax(kar_x)))

    mask = np.isfinite(exp_x) & (exp_x >= x_min) & (exp_x <= x_max)
    x = exp_x[mask]
    exp = pd.to_numeric(
        experiment.loc[mask, "dB_dT_cm_inv_amagat_neg2_K_neg1"],
        errors="coerce",
    ).to_numpy(dtype=float)
    u_exp = pd.to_numeric(
        experiment.loc[mask, "u_dB_dT_cm_inv_amagat_neg2_K_neg1"],
        errors="coerce",
    ).to_numpy(dtype=float)
    kar = np.interp(
        x,
        kar_x,
        pd.to_numeric(
            karman["linear_dB_dT_cm_inv_amagat_neg2_K_neg1"],
            errors="coerce",
        ).to_numpy(dtype=float),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        difference = exp - kar
        relative_difference = difference / kar * 100.0
        ratio = exp / kar

    return pd.DataFrame(
        {
            "wavenumber": x,
            "this_work_dB_dT_cm_inv_amagat_neg2_K_neg1": exp,
            "this_work_u_dB_dT_cm_inv_amagat_neg2_K_neg1": u_exp,
            "this_work_dB_dT_lower_1sigma_cm_inv_amagat_neg2_K_neg1": exp - u_exp,
            "this_work_dB_dT_upper_1sigma_cm_inv_amagat_neg2_K_neg1": exp + u_exp,
            "karman_dB_dT_cm_inv_amagat_neg2_K_neg1": kar,
            "difference_this_work_minus_karman_cm_inv_amagat_neg2_K_neg1": difference,
            "relative_difference_percent": relative_difference,
            "this_work_over_karman": ratio,
            "this_work_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1": exp / 1e-9,
            "this_work_u_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1": u_exp / 1e-9,
            "this_work_dB_dT_lower_1sigma_1e_neg9_cm_inv_amagat_neg2_K_neg1": (exp - u_exp)
            / 1e-9,
            "this_work_dB_dT_upper_1sigma_1e_neg9_cm_inv_amagat_neg2_K_neg1": (exp + u_exp)
            / 1e-9,
            "karman_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1": kar / 1e-9,
        }
    )


def summarize(plot_data: pd.DataFrame) -> pd.DataFrame:
    x = plot_data["wavenumber"].to_numpy(dtype=float)
    this_work = plot_data["this_work_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)
    karman = plot_data["karman_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)
    rel = plot_data["relative_difference_percent"].replace([np.inf, -np.inf], np.nan).to_numpy(
        dtype=float
    )
    this_idx = int(np.nanargmax(this_work))
    karman_idx = int(np.nanargmax(karman))
    return pd.DataFrame(
        [
            {
                "wavenumber_min": float(np.nanmin(x)),
                "wavenumber_max": float(np.nanmax(x)),
                "n_points": int(len(plot_data)),
                "this_work_peak_wavenumber": float(x[this_idx]),
                "this_work_peak_dB_dT": float(this_work[this_idx]),
                "karman_peak_wavenumber": float(x[karman_idx]),
                "karman_peak_dB_dT": float(karman[karman_idx]),
                "peak_this_work_over_karman": float(this_work[this_idx] / karman[karman_idx]),
                "median_relative_difference_percent": float(np.nanmedian(rel)),
                "mean_relative_difference_percent": float(np.nanmean(rel)),
                "rms_relative_difference_percent": float(np.sqrt(np.nanmean(rel**2))),
            }
        ]
    )


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(plot_data: pd.DataFrame, output_png: Path, width_mm: float, height_mm: float, dpi: int) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.18, top=0.965)

    x = plot_data["wavenumber"].to_numpy(dtype=float)
    this_work = plot_data["this_work_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1"].to_numpy(
        dtype=float
    )
    lower = plot_data[
        "this_work_dB_dT_lower_1sigma_1e_neg9_cm_inv_amagat_neg2_K_neg1"
    ].to_numpy(dtype=float)
    upper = plot_data[
        "this_work_dB_dT_upper_1sigma_1e_neg9_cm_inv_amagat_neg2_K_neg1"
    ].to_numpy(dtype=float)
    karman = plot_data["karman_dB_dT_1e_neg9_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)

    this_color = "#0072B2"
    karman_color = "#D55E00"
    ax.fill_between(x, lower, upper, color=this_color, alpha=0.18, lw=0, label=r"This work $\pm1\sigma$")
    ax.plot(x, this_work, color=this_color, lw=0.85, label="This work")
    ax.plot(x, karman, color=karman_color, lw=0.85, ls="--", label="Karman")
    ax.axhline(0.0, color="black", lw=0.55)

    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    y_max = float(np.nanmax(np.concatenate([upper, karman])))
    ax.set_ylim(0.0, y_max * 1.08)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.savefig(output_png, dpi=dpi)
    fig.savefig(output_png.with_suffix(".pdf"))
    fig.savefig(output_png.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_path = args.experiment.expanduser().resolve()
    karman_path = args.karman.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment, karman = load_inputs(experiment_path, karman_path)
    plot_data = build_plot_data(experiment, karman)
    summary = summarize(plot_data)

    plot_data_csv = output_dir / "this_work_vs_karman_dB_dT_common_range_plot_data.csv"
    summary_csv = output_dir / "this_work_vs_karman_dB_dT_common_range_summary.csv"
    output_png = output_dir / "this_work_vs_karman_dB_dT_common_range.png"
    plot_data.to_csv(plot_data_csv, index=False, float_format="%.15g")
    summary.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_figure(plot_data, output_png, args.figure_width_mm, args.figure_height_mm, args.dpi)

    print(f"Plot data: {plot_data_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
