#!/usr/bin/env python3
"""Analyze dB/dT using the user's 273 K, 296 K, and 333 K B spectra.

Default inputs:
  - 273 K and 333 K are read from the final uncertainty wide table.
  - 296 K is read from the newly processed 296 K raw-data result.

The 296 K 300 Torr result is ignored by default.  The 296 K representative B
value is the mean of 500 Torr and 700 Torr:

    B_296K = (B_296K_500Torr + B_296K_700Torr) / 2

For each wavenumber, an unweighted linear fit is performed:

    B(nu, T) = a(nu) + (dB/dT)(nu) * T

The fitted slope is the temperature-dependence coefficient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_B_273_333 = Path(
    "output/results/uncertainty/CIA/final_binary_coefficient_uncertainty/"
    "binary_coefficient_uncertainty_wide.csv"
)
DEFAULT_B_296 = Path(
    "output/results/analysis/296K_raw_final_B/296K_B_three_pressures_wide.csv"
)
DEFAULT_SELECTED = (9200.0, 9323.0, 9420.0, 9520.0, 9800.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate dB/dT from 273 K, 296 K, and 333 K B spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--b-273-333",
        type=Path,
        default=DEFAULT_B_273_333,
        help="Wide table containing 273K_500Torr_B and 333K_500Torr_B.",
    )
    parser.add_argument(
        "--b-296",
        type=Path,
        default=DEFAULT_B_296,
        help="Wide table containing 296 K pressure-group B columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/B_temperature_dependence_273_296_333"),
        help="Output directory.",
    )
    parser.add_argument(
        "--use-296",
        choices=("mean-500-700", "500", "700"),
        default="mean-500-700",
        help="Which 296 K pressure result to use in the temperature fit.",
    )
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=list(DEFAULT_SELECTED),
        help="Selected wavenumbers shown in the B-vs-T panel.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def read_inputs(path_273_333: Path, path_296: Path, use_296: str) -> pd.DataFrame:
    b_old = pd.read_csv(path_273_333.expanduser())
    b_296 = pd.read_csv(path_296.expanduser())
    required_old = ["wavenumber", "273K_500Torr_B", "333K_500Torr_B"]
    required_296 = ["wavenumber", "296K_500Torr_B", "296K_700Torr_B"]
    missing_old = [col for col in required_old if col not in b_old.columns]
    missing_296 = [col for col in required_296 if col not in b_296.columns]
    if missing_old:
        raise SystemExit(f"Missing columns in {path_273_333}: {missing_old}")
    if missing_296:
        raise SystemExit(f"Missing columns in {path_296}: {missing_296}")

    left = b_old[["wavenumber", "273K_500Torr_B", "333K_500Torr_B"]].copy()
    left["wavenumber"] = pd.to_numeric(left["wavenumber"], errors="coerce").round(2)
    if "273K_500Torr_u_B" in b_old.columns:
        left["u_B_273K"] = b_old["273K_500Torr_u_B"]
    if "333K_500Torr_u_B" in b_old.columns:
        left["u_B_333K"] = b_old["333K_500Torr_u_B"]

    right = b_296[["wavenumber", "296K_500Torr_B", "296K_700Torr_B"]].copy()
    right["wavenumber"] = pd.to_numeric(right["wavenumber"], errors="coerce").round(2)
    left = left.groupby("wavenumber", as_index=False).mean(numeric_only=True)
    right = right.groupby("wavenumber", as_index=False).mean(numeric_only=True)
    df = left.merge(right, on="wavenumber", how="inner")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["B_296K_mean_500_700"] = df[["296K_500Torr_B", "296K_700Torr_B"]].mean(
        axis=1,
        skipna=False,
    )
    df["B_296K_500_minus_700"] = df["296K_500Torr_B"] - df["296K_700Torr_B"]
    df["B_296K_pressure_half_range"] = 0.5 * np.abs(df["B_296K_500_minus_700"])
    if use_296 == "mean-500-700":
        df["B_296K_used"] = df["B_296K_mean_500_700"]
    elif use_296 == "500":
        df["B_296K_used"] = df["296K_500Torr_B"]
    elif use_296 == "700":
        df["B_296K_used"] = df["296K_700Torr_B"]
    else:
        raise ValueError(use_296)
    df["B_296K_used_mode"] = use_296

    df = df.rename(
        columns={
            "273K_500Torr_B": "B_273K",
            "333K_500Torr_B": "B_333K",
        }
    )
    df = df.dropna(subset=["wavenumber", "B_273K", "B_296K_used", "B_333K"])
    return df.sort_values("wavenumber").reset_index(drop=True)


def fit_temperature_dependence(df: pd.DataFrame) -> pd.DataFrame:
    temperatures = np.asarray([273.0, 296.0, 333.0], dtype=float)
    y = np.vstack(
        [
            df["B_273K"].to_numpy(dtype=float),
            df["B_296K_used"].to_numpy(dtype=float),
            df["B_333K"].to_numpy(dtype=float),
        ]
    )
    t_mean = temperatures.mean()
    denom = float(np.sum((temperatures - t_mean) ** 2))
    slope = np.sum((temperatures - t_mean)[:, None] * y, axis=0) / denom
    intercept = np.mean(y, axis=0) - slope * t_mean
    fitted = intercept[None, :] + slope[None, :] * temperatures[:, None]
    residual = y - fitted
    sse = np.sum(residual**2, axis=0)
    rmse = np.sqrt(sse / len(temperatures))
    residual_std = np.sqrt(sse / max(len(temperatures) - 2, 1))
    slope_stderr = residual_std / np.sqrt(denom)

    out = df.copy()
    out["temperature_fit_mode"] = "unweighted_linear_273_296_333"
    out["fit_temperature_points"] = "273,296,333"
    out["fit_intercept"] = intercept
    out["dB_dT_cm_inv_amagat_neg2_K_neg1"] = slope
    out["u_dB_dT_from_fit_residual"] = slope_stderr
    out["fit_rmse_cm_inv_amagat_neg2"] = rmse
    out["B_fit_273K"] = fitted[0]
    out["B_fit_296K"] = fitted[1]
    out["B_fit_333K"] = fitted[2]
    out["B_residual_273K"] = out["B_273K"] - out["B_fit_273K"]
    out["B_residual_296K"] = out["B_296K_used"] - out["B_fit_296K"]
    out["B_residual_333K"] = out["B_333K"] - out["B_fit_333K"]
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


def summarize(fit_df: pd.DataFrame) -> pd.DataFrame:
    slope = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"]
    max_row = fit_df.loc[slope.idxmax()]
    min_row = fit_df.loc[slope.idxmin()]
    peak_row = fit_df.loc[fit_df["B_296K_used"].idxmax()]
    diff = fit_df["B_296K_500_minus_700"]
    return pd.DataFrame(
        [
            {
                "n_points": int(len(fit_df)),
                "wavenumber_min": float(fit_df["wavenumber"].min()),
                "wavenumber_max": float(fit_df["wavenumber"].max()),
                "dB_dT_min": float(min_row["dB_dT_cm_inv_amagat_neg2_K_neg1"]),
                "dB_dT_min_wavenumber": float(min_row["wavenumber"]),
                "dB_dT_max": float(max_row["dB_dT_cm_inv_amagat_neg2_K_neg1"]),
                "dB_dT_max_wavenumber": float(max_row["wavenumber"]),
                "B_296K_peak": float(peak_row["B_296K_used"]),
                "B_296K_peak_wavenumber": float(peak_row["wavenumber"]),
                "B_296K_500_minus_700_mean": float(diff.mean()),
                "B_296K_500_minus_700_median": float(diff.median()),
                "B_296K_500_greater_fraction": float((diff > 0).mean()),
                "fit_rmse_median": float(fit_df["fit_rmse_cm_inv_amagat_neg2"].median()),
            }
        ]
    )


def plot_temperature_dependence(
    fit_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_png: Path,
    dpi: int,
) -> None:
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
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    temps = np.asarray([273.0, 296.0, 333.0], dtype=float)
    y_cols = ["B_273K", "B_296K_used", "B_333K"]
    markers = ["o", "^", "s", "D", "v", "P"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        b_vals = row[y_cols].to_numpy(dtype=float)
        t_line = np.linspace(268.0, 338.0, 120)
        fit_line = row["fit_intercept"] + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        ax_left.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=1.0,
            label=f"{x0:.0f} cm$^{{-1}}$",
        )
        ax_left.plot(
            temps,
            b_vals / 1e-6,
            marker=markers[i % len(markers)],
            color="black",
            linestyle="none",
            ms=4.5,
            zorder=4,
        )

    ax_left.set_xlabel("Temperature (K)")
    ax_left.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_left.set_xlim(268.0, 338.0)
    ax_left.legend(frameon=False, fontsize=8, loc="best")
    ax_left.minorticks_on()

    x = fit_df["wavenumber"].to_numpy(dtype=float)
    slope_scaled = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    ax_right.plot(x, slope_scaled, color="#1f77b4", lw=1.2, label=r"$B_{\mathrm{O_2-O_2}}$")
    ax_right.axhline(0, color="black", lw=0.8)
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        y0 = float(row["dB_dT_cm_inv_amagat_neg2_K_neg1"]) / 1e-9
        ax_right.axvline(x0, color="0.75", lw=0.8, ls="--", zorder=0)
        ax_right.plot(x0, y0, "o", ms=3.5, color="black", zorder=3)
        offset = 8 if i % 2 == 0 else -14
        ax_right.annotate(
            f"{x0:.0f}",
            xy=(x0, y0),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=8,
        )
    ax_right.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_right.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_right.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax_right.legend(frameon=False, loc="best")
    ax_right.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_inputs(args.b_273_333, args.b_296, args.use_296)
    fit_df = fit_temperature_dependence(df)
    selected_df = nearest_rows(fit_df, args.selected)
    summary_df = summarize(fit_df)

    fit_csv = output_dir / "b_temperature_dependence_273_296_333_fit.csv"
    selected_csv = output_dir / "b_temperature_dependence_273_296_333_selected_wavenumbers.csv"
    summary_csv = output_dir / "b_temperature_dependence_273_296_333_summary.csv"
    output_png = output_dir / "b_temperature_dependence_273_296_333.png"
    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    summary_df.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_temperature_dependence(fit_df, selected_df, output_png, args.dpi)

    print(f"Fit CSV: {fit_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary_df.to_string(index=False))
    print("Selected wavenumbers:")
    cols = [
        "wavenumber",
        "B_273K",
        "B_296K_used",
        "B_333K",
        "dB_dT_cm_inv_amagat_neg2_K_neg1",
        "u_dB_dT_from_fit_residual",
    ]
    print(selected_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
