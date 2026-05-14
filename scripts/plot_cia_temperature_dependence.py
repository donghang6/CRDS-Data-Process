#!/usr/bin/env python3
"""Plot temperature dependence of O2-O2 CIA binary coefficients.

This reproduces the logic of the temperature-dependence figure often used for
CIA data:

1. Left panel: B at selected wavenumbers as a function of temperature.
2. Right panel: fitted temperature coefficient dB/dT as a function of
   wavenumber.

For the current data set, 303 K has three pressure groups. They are first
combined into one 303 K point at each wavenumber using inverse-variance
weighting. Then each wavenumber is fitted independently with the three
temperature points:

    B(nu, T) = intercept(nu) + dB_dT(nu) * T

The default fit is an ordinary three-point linear fit, which matches the
right-panel definition of dB/dT in the reference figure. A weighted fit is
also available for diagnostics.

Usage:
    python scripts/plot_cia_temperature_dependence.py

Optional selected wavenumbers:
    python scripts/plot_cia_temperature_dependence.py \
      --selected 9200 9322.95 9427.60 9700 9800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The middle point is placed at the maximum of dB/dT for the current data.
# The third point is placed where dB/dT is midway between the second and
# fourth selected points, making the left-panel slopes easier to compare.
DEFAULT_SELECTED = (9200.0, 9322.95, 9427.60, 9700.0, 9800.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CIA B temperature dependence and fitted dB/dT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "output/results/uncertainty/CIA/final_binary_coefficient_uncertainty/"
            "binary_coefficient_uncertainty_wide.csv"
        ),
        help="Wide B uncertainty table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/CIA_temperature_dependence"),
        help="Output directory.",
    )
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=list(DEFAULT_SELECTED),
        help="Selected wavenumbers for the left panel.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--fit-mode",
        choices=("unweighted", "weighted"),
        default="unweighted",
        help="Linear fit mode for B vs temperature at each wavenumber.",
    )
    return parser.parse_args()


def weighted_mean(values: np.ndarray, uncertainties: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.where(uncertainties > 0, 1.0 / uncertainties**2, np.nan)
    weights = np.where(np.isfinite(weights), weights, np.nan)
    mean = np.nansum(weights * values, axis=0) / np.nansum(weights, axis=0)
    uncertainty = 1.0 / np.sqrt(np.nansum(weights, axis=0))
    return mean, uncertainty


def combine_temperature_groups(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(df["wavenumber"], errors="coerce").to_numpy(dtype=float)
    b273 = pd.to_numeric(df["273K_500Torr_B"], errors="coerce").to_numpy(dtype=float)
    u273 = pd.to_numeric(df["273K_500Torr_u_B"], errors="coerce").to_numpy(dtype=float)
    b333 = pd.to_numeric(df["333K_500Torr_B"], errors="coerce").to_numpy(dtype=float)
    u333 = pd.to_numeric(df["333K_500Torr_u_B"], errors="coerce").to_numpy(dtype=float)

    b303_values = np.vstack([
        pd.to_numeric(df["303K_500Torr_B"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["303K_600Torr_B"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["303K_700Torr_B"], errors="coerce").to_numpy(dtype=float),
    ])
    u303_values = np.vstack([
        pd.to_numeric(df["303K_500Torr_u_B"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["303K_600Torr_u_B"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["303K_700Torr_u_B"], errors="coerce").to_numpy(dtype=float),
    ])
    b303, u303 = weighted_mean(b303_values, u303_values)

    return pd.DataFrame({
        "wavenumber": x,
        "B_273K": b273,
        "u_B_273K": u273,
        "B_303K_weighted": b303,
        "u_B_303K_weighted": u303,
        "B_333K": b333,
        "u_B_333K": u333,
    })


def linear_fit_by_wavenumber(combined: pd.DataFrame, fit_mode: str) -> pd.DataFrame:
    temperatures = np.asarray([273.0, 303.0, 333.0], dtype=float)
    y = np.vstack([
        combined["B_273K"].to_numpy(dtype=float),
        combined["B_303K_weighted"].to_numpy(dtype=float),
        combined["B_333K"].to_numpy(dtype=float),
    ])
    sigma = np.vstack([
        combined["u_B_273K"].to_numpy(dtype=float),
        combined["u_B_303K_weighted"].to_numpy(dtype=float),
        combined["u_B_333K"].to_numpy(dtype=float),
    ])

    if fit_mode == "weighted":
        weights = np.where(sigma > 0, 1.0 / sigma**2, np.nan)
        weights = np.where(np.isfinite(weights), weights, 0.0)

        s = np.sum(weights, axis=0)
        sx = np.sum(weights * temperatures[:, None], axis=0)
        sy = np.sum(weights * y, axis=0)
        sxx = np.sum(weights * temperatures[:, None] ** 2, axis=0)
        sxy = np.sum(weights * temperatures[:, None] * y, axis=0)
        denom = s * sxx - sx**2

        slope = (s * sxy - sx * sy) / denom
        intercept = (sxx * sy - sx * sxy) / denom
        slope_uncertainty = np.sqrt(s / denom)
        intercept_uncertainty = np.sqrt(sxx / denom)
    else:
        x_mean = float(np.mean(temperatures))
        denom = float(np.sum((temperatures - x_mean) ** 2))
        slope_coeff = (temperatures - x_mean) / denom
        intercept_coeff = np.full_like(temperatures, 1.0 / len(temperatures)) - x_mean * slope_coeff
        slope = np.sum(slope_coeff[:, None] * y, axis=0)
        intercept = np.sum(intercept_coeff[:, None] * y, axis=0)
        slope_uncertainty = np.sqrt(np.sum((slope_coeff[:, None] * sigma) ** 2, axis=0))
        intercept_uncertainty = np.sqrt(np.sum((intercept_coeff[:, None] * sigma) ** 2, axis=0))

    out = combined.copy()
    out["temperature_fit_mode"] = fit_mode
    out["dB_dT_cm_inv_amagat_neg2_K_neg1"] = slope
    out["u_dB_dT_cm_inv_amagat_neg2_K_neg1"] = slope_uncertainty
    out["fit_intercept"] = intercept
    out["u_fit_intercept"] = intercept_uncertainty
    return out


def nearest_rows(fit_df: pd.DataFrame, selected: list[float]) -> pd.DataFrame:
    x = fit_df["wavenumber"].to_numpy(dtype=float)
    rows = []
    for target in selected:
        idx = int(np.nanargmin(np.abs(x - target)))
        row = fit_df.iloc[idx].copy()
        row["selected_target_wavenumber"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def plot_temperature_dependence(fit_df: pd.DataFrame, selected_df: pd.DataFrame, output_png: Path, dpi: int) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    })
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

    temps = np.asarray([273.0, 303.0, 333.0])
    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":"]
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        b_vals = np.asarray([
            row["B_273K"],
            row["B_303K_weighted"],
            row["B_333K"],
        ], dtype=float)
        u_vals = np.asarray([
            row["u_B_273K"],
            row["u_B_303K_weighted"],
            row["u_B_333K"],
        ], dtype=float)
        label = f"{x0:.0f} cm$^{{-1}}$"
        t_line = np.linspace(268.0, 338.0, 100)
        fit_line = row["fit_intercept"] + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        ax_left.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=1.0,
            label=label,
        )
        ax_left.errorbar(
            temps,
            b_vals / 1e-6,
            yerr=u_vals / 1e-6,
            fmt=markers[i % len(markers)],
            color="black",
            ecolor="black",
            ms=4.5,
            capsize=4.0,
            capthick=1.0,
            elinewidth=1.0,
            lw=0.0,
            zorder=4,
        )

    ax_left.set_xlabel("Temperature (K)")
    ax_left.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_left.set_xlim(265, 340)
    ax_left.legend(frameon=False, loc="best")
    ax_left.minorticks_on()

    x = fit_df["wavenumber"].to_numpy(dtype=float)
    slope_scaled = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    ax_right.plot(x, slope_scaled, color="#1f77b4", lw=1.3, label=r"$B_{\mathrm{O_2-O_2}}$")
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        y0 = float(row["dB_dT_cm_inv_amagat_neg2_K_neg1"]) / 1e-9
        ax_right.axvline(x0, color="0.75", lw=0.8, ls="--", zorder=0)
        ax_right.plot(x0, y0, marker="o", ms=3.5, color="black", zorder=3)
        text_offset = 8 if i % 2 == 0 else -14
        va = "bottom" if text_offset > 0 else "top"
        ax_right.annotate(
            f"{x0:.0f}",
            xy=(x0, y0),
            xytext=(0, text_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color="black",
        )
    ax_right.axhline(0, color="black", lw=0.8)
    ax_right.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_right.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_right.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax_right.legend(frameon=False, loc="best")
    ax_right.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    combined = combine_temperature_groups(df)
    fit_df = linear_fit_by_wavenumber(combined, args.fit_mode)
    selected_df = nearest_rows(fit_df, args.selected)

    fit_csv = output_dir / "cia_temperature_dependence_fit.csv"
    selected_csv = output_dir / "cia_temperature_dependence_selected_wavenumbers.csv"
    output_png = output_dir / "cia_temperature_dependence.png"
    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    plot_temperature_dependence(fit_df, selected_df, output_png, args.dpi)

    print(f"Fit CSV: {fit_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Figure: {output_png}")
    print("Selected wavenumbers:")
    cols = [
        "wavenumber",
        "B_273K",
        "B_303K_weighted",
        "B_333K",
        "dB_dT_cm_inv_amagat_neg2_K_neg1",
    ]
    print(selected_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
