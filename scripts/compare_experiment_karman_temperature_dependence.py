#!/usr/bin/env python3
"""Compare measured and Karman dB/dT curves in one figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay measured O2-O2 dB/dT with Karman ab initio dB/dT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(
            "output/results/analysis/B_temperature_dependence_303K_600Torr/"
            "b_temperature_dependence_fit.csv"
        ),
        help="Measured temperature-dependence fit table.",
    )
    parser.add_argument(
        "--karman",
        type=Path,
        default=Path(
            "output/results/analysis/Karman_temperature_linearity_276_336/"
            "karman_temperature_linearity_fit.csv"
        ),
        help="Karman temperature-dependence fit table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/Experiment_vs_Karman_temperature_dependence"),
        help="Output directory.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def find_slope_column(df: pd.DataFrame, preferred: list[str]) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    candidates = [col for col in df.columns if "dB_dT" in col]
    if not candidates:
        raise ValueError("No dB/dT column was found.")
    return candidates[0]


def load_comparison(experiment_csv: Path, karman_csv: Path) -> pd.DataFrame:
    exp = pd.read_csv(experiment_csv)
    kar = pd.read_csv(karman_csv)
    exp_col = find_slope_column(exp, ["dB_dT_cm_inv_amagat_neg2_K_neg1"])
    kar_col = find_slope_column(
        kar,
        [
            "linear_dB_dT_cm_inv_amagat_neg2_K_neg1",
            "dB_dT_cm_inv_amagat_neg2_K_neg1",
        ],
    )

    x_exp = exp["wavenumber"].to_numpy(dtype=float)
    x_kar = kar["wavenumber"].to_numpy(dtype=float)
    overlap = (x_exp >= np.nanmin(x_kar)) & (x_exp <= np.nanmax(x_kar))
    x = x_exp[overlap]
    exp_slope = exp.loc[overlap, exp_col].to_numpy(dtype=float)
    kar_slope = np.interp(x, x_kar, kar[kar_col].to_numpy(dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = exp_slope / kar_slope
        rel_diff_percent = 100.0 * (exp_slope - kar_slope) / kar_slope

    return pd.DataFrame(
        {
            "wavenumber": x,
            "experiment_dB_dT_cm_inv_amagat_neg2_K_neg1": exp_slope,
            "karman_dB_dT_cm_inv_amagat_neg2_K_neg1": kar_slope,
            "difference_experiment_minus_karman": exp_slope - kar_slope,
            "experiment_over_karman": ratio,
            "relative_difference_percent": rel_diff_percent,
        }
    )


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    exp = comparison["experiment_dB_dT_cm_inv_amagat_neg2_K_neg1"]
    kar = comparison["karman_dB_dT_cm_inv_amagat_neg2_K_neg1"]
    diff = comparison["difference_experiment_minus_karman"]
    ratio = comparison["experiment_over_karman"]
    exp_peak = comparison.loc[exp.idxmax()]
    kar_peak = comparison.loc[kar.idxmax()]
    return pd.DataFrame(
        [
            {
                "wavenumber_min": comparison["wavenumber"].min(),
                "wavenumber_max": comparison["wavenumber"].max(),
                "experiment_dB_dT_max": exp_peak["experiment_dB_dT_cm_inv_amagat_neg2_K_neg1"],
                "experiment_dB_dT_max_wavenumber": exp_peak["wavenumber"],
                "karman_dB_dT_max": kar_peak["karman_dB_dT_cm_inv_amagat_neg2_K_neg1"],
                "karman_dB_dT_max_wavenumber": kar_peak["wavenumber"],
                "mean_difference": diff.mean(),
                "median_difference": diff.median(),
                "mean_experiment_over_karman": ratio.replace([np.inf, -np.inf], np.nan).mean(),
                "median_experiment_over_karman": ratio.replace([np.inf, -np.inf], np.nan).median(),
            }
        ]
    )


def plot_comparison(comparison: pd.DataFrame, output_png: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
        }
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.0),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )
    x = comparison["wavenumber"].to_numpy(dtype=float)
    exp = comparison["experiment_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    kar = comparison["karman_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    diff = comparison["difference_experiment_minus_karman"].to_numpy(dtype=float) / 1e-9

    ax_top.plot(x, exp, color="#d62728", lw=1.4, label="Experiment, 303K=600 Torr")
    ax_top.plot(x, kar, color="#1f77b4", lw=1.4, label="Karman ab initio, 276-336 K")
    ax_top.axhline(0, color="black", lw=0.8)
    ax_top.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_top.legend(frameon=False)
    ax_top.minorticks_on()

    ax_bottom.plot(x, diff, color="black", lw=1.0)
    ax_bottom.axhline(0, color="0.4", lw=0.8)
    ax_bottom.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_bottom.set_ylabel(r"Exp. - Karman")
    ax_bottom.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_comparison(args.experiment.expanduser().resolve(), args.karman.expanduser().resolve())
    summary = summarize(comparison)

    comparison_csv = output_dir / "experiment_303K_600Torr_vs_karman_dB_dT.csv"
    summary_csv = output_dir / "experiment_303K_600Torr_vs_karman_summary.csv"
    output_png = output_dir / "experiment_303K_600Torr_vs_karman_dB_dT.png"
    comparison.to_csv(comparison_csv, index=False, float_format="%.15g")
    summary.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_comparison(comparison, output_png, args.dpi)

    print(f"Comparison CSV: {comparison_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
