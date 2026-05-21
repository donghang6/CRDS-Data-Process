#!/usr/bin/env python3
"""Compare measured O2-O2 B spectra with Karman ab initio data.

The measured temperatures are paired with the closest Karman temperatures:

    measured 273 K -> Karman 276 K
    measured 303 K -> Karman 306 K
    measured 333 K -> Karman 336 K

For 303 K, the measured curve uses the uncertainty-weighted combination of
the 500/600/700 Torr pressure groups.  The plot is restricted to the common
wavenumber range of the measured and Karman data.
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
DEFAULT_OUTPUT_DIR = Path("output/results/analysis/Experiment_vs_Karman_B_temperature_comparison")

PAIRS = [
    {
        "exp_temp": 273.0,
        "karman_temp": 276.0,
        "exp_col": "B_273K_500Torr",
        "karman_col": "B_276K",
        "color": "#0072B2",
        "label_exp": "Exp. 273 K",
        "label_karman": "Karman 276 K",
    },
    {
        "exp_temp": 303.0,
        "karman_temp": 306.0,
        "exp_col": "B_303K_weighted",
        "karman_col": "B_306K",
        "color": "#009E73",
        "label_exp": "Exp. 303 K",
        "label_karman": "Karman 306 K",
    },
    {
        "exp_temp": 333.0,
        "karman_temp": 336.0,
        "exp_col": "B_333K_500Torr",
        "karman_col": "B_336K",
        "color": "#D55E00",
        "label_exp": "Exp. 333 K",
        "label_karman": "Karman 336 K",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot measured B spectra against closest-temperature Karman B spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--karman", type=Path, default=DEFAULT_KARMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-width-mm", type=float, default=174.0)
    parser.add_argument("--figure-height-mm", type=float, default=82.0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_inputs(experiment_path: Path, karman_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_cols = ["wavenumber"] + [pair["exp_col"] for pair in PAIRS]
    karman_cols = ["wavenumber"] + [pair["karman_col"] for pair in PAIRS]
    experiment = pd.read_csv(experiment_path, usecols=exp_cols)
    karman = pd.read_csv(karman_path, usecols=karman_cols)
    return experiment, karman


def build_comparison_table(experiment: pd.DataFrame, karman: pd.DataFrame) -> pd.DataFrame:
    exp_x = pd.to_numeric(experiment["wavenumber"], errors="coerce").to_numpy(dtype=float)
    kar_x = pd.to_numeric(karman["wavenumber"], errors="coerce").to_numpy(dtype=float)
    x_min = max(float(np.nanmin(exp_x)), float(np.nanmin(kar_x)))
    x_max = min(float(np.nanmax(exp_x)), float(np.nanmax(kar_x)))
    mask = np.isfinite(exp_x) & (exp_x >= x_min) & (exp_x <= x_max)
    x = exp_x[mask]

    out = pd.DataFrame({"wavenumber": x})
    for pair in PAIRS:
        exp_y = pd.to_numeric(experiment.loc[mask, pair["exp_col"]], errors="coerce").to_numpy(dtype=float)
        kar_y_native_grid = pd.to_numeric(karman[pair["karman_col"]], errors="coerce").to_numpy(dtype=float)
        kar_y = np.interp(x, kar_x, kar_y_native_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = (exp_y - kar_y) / kar_y * 100.0
            ratio = exp_y / kar_y
        stem = f"exp_{int(pair['exp_temp'])}K_vs_karman_{int(pair['karman_temp'])}K"
        out[f"{stem}_B_exp_cm_inv_amagat_neg2"] = exp_y
        out[f"{stem}_B_karman_cm_inv_amagat_neg2"] = kar_y
        out[f"{stem}_difference_cm_inv_amagat_neg2"] = exp_y - kar_y
        out[f"{stem}_relative_difference_percent"] = rel_diff
        out[f"{stem}_experiment_over_karman"] = ratio
    return out


def summarize(plot_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        stem = f"exp_{int(pair['exp_temp'])}K_vs_karman_{int(pair['karman_temp'])}K"
        exp_col = f"{stem}_B_exp_cm_inv_amagat_neg2"
        kar_col = f"{stem}_B_karman_cm_inv_amagat_neg2"
        rel_col = f"{stem}_relative_difference_percent"
        x = plot_data["wavenumber"].to_numpy(dtype=float)
        exp = plot_data[exp_col].to_numpy(dtype=float)
        kar = plot_data[kar_col].to_numpy(dtype=float)
        rel = plot_data[rel_col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        exp_peak_idx = int(np.nanargmax(exp))
        kar_peak_idx = int(np.nanargmax(kar))
        rows.append(
            {
                "experiment_temperature_k": pair["exp_temp"],
                "karman_temperature_k": pair["karman_temp"],
                "wavenumber_min": float(np.nanmin(x)),
                "wavenumber_max": float(np.nanmax(x)),
                "experiment_peak_wavenumber": float(x[exp_peak_idx]),
                "experiment_peak_B": float(exp[exp_peak_idx]),
                "karman_peak_wavenumber": float(x[kar_peak_idx]),
                "karman_peak_B": float(kar[kar_peak_idx]),
                "peak_experiment_over_karman": float(exp[exp_peak_idx] / kar[kar_peak_idx]),
                "median_relative_difference_percent": float(np.nanmedian(rel)),
                "mean_relative_difference_percent": float(np.nanmean(rel)),
                "rms_relative_difference_percent": float(np.sqrt(np.nanmean(rel**2))),
            }
        )
    return pd.DataFrame(rows)


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
            "legend.fontsize": 6.7,
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


def plot_pair_curves(ax: plt.Axes, plot_data: pd.DataFrame, linewidth: float) -> None:
    x = plot_data["wavenumber"].to_numpy(dtype=float)
    for pair in PAIRS:
        stem = f"exp_{int(pair['exp_temp'])}K_vs_karman_{int(pair['karman_temp'])}K"
        exp = plot_data[f"{stem}_B_exp_cm_inv_amagat_neg2"].to_numpy(dtype=float) / 1e-6
        kar = plot_data[f"{stem}_B_karman_cm_inv_amagat_neg2"].to_numpy(dtype=float) / 1e-6
        ax.plot(x, exp, color=pair["color"], lw=linewidth, ls="-", label=pair["label_exp"])
        ax.plot(x, kar, color=pair["color"], lw=linewidth, ls="--", label=pair["label_karman"])


def set_inset_limits(ax: plt.Axes, plot_data: pd.DataFrame, xlim: tuple[float, float], pad: float) -> None:
    x = plot_data["wavenumber"].to_numpy(dtype=float)
    mask = (x >= xlim[0]) & (x <= xlim[1])
    values = []
    for pair in PAIRS:
        stem = f"exp_{int(pair['exp_temp'])}K_vs_karman_{int(pair['karman_temp'])}K"
        values.append(plot_data.loc[mask, f"{stem}_B_exp_cm_inv_amagat_neg2"].to_numpy(dtype=float) / 1e-6)
        values.append(plot_data.loc[mask, f"{stem}_B_karman_cm_inv_amagat_neg2"].to_numpy(dtype=float) / 1e-6)
    y = np.concatenate(values)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_pad = (y_max - y_min) * pad
    ax.set_xlim(*xlim)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_figure(plot_data: pd.DataFrame, output_png: Path, width_mm: float, height_mm: float, dpi: int) -> None:
    set_plot_style()
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        bottom=0.16,
        top=0.965,
        width_ratios=[1.0, 3.25],
        height_ratios=[1.0, 1.0],
        wspace=0.20,
        hspace=0.48,
    )
    peak_inset = fig.add_subplot(grid[0, 0])
    wing_inset = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[:, 1])

    plot_pair_curves(ax, plot_data, linewidth=0.85)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax.minorticks_on()
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.8, labelspacing=0.15)

    x = plot_data["wavenumber"].to_numpy(dtype=float)
    y_cols = [col for col in plot_data.columns if col.endswith("_cm_inv_amagat_neg2") and "_difference_" not in col]
    y_max = float(np.nanmax(plot_data[y_cols].to_numpy(dtype=float))) / 1e-6
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(0.0, y_max * 1.08)

    plot_pair_curves(peak_inset, plot_data, linewidth=0.62)
    set_inset_limits(peak_inset, plot_data, (9280.0, 9365.0), pad=0.08)
    peak_inset.minorticks_on()
    peak_inset.tick_params(labelsize=5.4, pad=1.0)
    peak_inset.legend().remove()

    plot_pair_curves(wing_inset, plot_data, linewidth=0.62)
    set_inset_limits(wing_inset, plot_data, (9130.0, 9210.0), pad=0.12)
    wing_inset.minorticks_on()
    wing_inset.tick_params(labelsize=5.4, pad=1.0)
    wing_inset.legend().remove()

    for one_ax in [ax, peak_inset, wing_inset]:
        for spine in one_ax.spines.values():
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

    experiment, karman = read_inputs(experiment_path, karman_path)
    plot_data = build_comparison_table(experiment, karman)
    summary = summarize(plot_data)

    plot_data_csv = output_dir / "experiment_vs_karman_B_temperature_comparison_plot_data.csv"
    summary_csv = output_dir / "experiment_vs_karman_B_temperature_comparison_summary.csv"
    output_png = output_dir / "experiment_vs_karman_B_temperature_comparison.png"
    plot_data.to_csv(plot_data_csv, index=False, float_format="%.15g")
    summary.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_figure(plot_data, output_png, args.figure_width_mm, args.figure_height_mm, args.dpi)

    print(f"Plot data: {plot_data_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
