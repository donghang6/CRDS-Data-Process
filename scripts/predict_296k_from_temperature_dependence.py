#!/usr/bin/env python3
"""Predict 296 K B from temperature-dependence coefficients and compare to data.

The prediction uses the weighted linear temperature model already fitted in
``temperature_dependence_weighted_fit.csv``:

    B(T) = intercept + (dB/dT) * T

The uncertainty of the predicted mean B(T0) is calculated from the weighted
linear-fit covariance matrix:

    u^2[B(T0)] = [1, T0] Cov(intercept, slope) [1, T0]^T

where the covariance matrix is rebuilt from the same three temperature points
and uncertainties stored in the fit table.  The final covariance is multiplied
by ``fit_birge_ratio**2`` to match the reported final fit uncertainties.

If the measured 296 K table contains a third column, it is interpreted as the
absolute uncertainty of the measured B value.  The comparison then also reports
the combined residual uncertainty,

    u_residual = sqrt(u_predicted**2 + u_measured**2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FIT = Path(
    "output/results/analysis/B_temperature_dependence_from_summary_303_combined_uB_x2/"
    "temperature_dependence_weighted_fit.csv"
)
DEFAULT_MEASURED = Path(
    "/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/500Torr 296K.txt"
)
DEFAULT_OUTPUT = Path("output/results/analysis/B_296K_temperature_dependence_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict 296 K B from dB/dT and compare with measured 296 K B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT, help="Temperature-dependence fit CSV.")
    parser.add_argument("--measured", type=Path, default=DEFAULT_MEASURED, help="Measured 296 K B table.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--temperature-k", type=float, default=296.0, help="Prediction temperature.")
    parser.add_argument("--encoding", default="gbk", help="Encoding of the measured text table.")
    parser.add_argument("--dpi", type=int, default=500, help="Figure DPI.")
    parser.add_argument("--figure-width-mm", type=float, default=190.0, help="Figure width in millimeters.")
    parser.add_argument("--figure-height-mm", type=float, default=135.0, help="Figure height in millimeters.")
    return parser.parse_args()


def read_measured(path: Path, encoding: str) -> pd.DataFrame:
    def read_with_skip(skiprows: int) -> pd.DataFrame:
        return pd.read_csv(
            path.expanduser(),
            sep=r"[,\t]+",
            engine="python",
            encoding=encoding,
            header=None,
            skiprows=skiprows,
            comment="#",
        )

    df = read_with_skip(2)
    if df.shape[1] < 2:
        df = read_with_skip(3)
    df = df.iloc[:, :3].copy()
    names = ["wavenumber", "B_296K_measured_500Torr", "u_B_296K_measured_500Torr"]
    df.columns = names[: df.shape[1]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("wavenumber")
    df["wavenumber"] = df["wavenumber"].round(2)
    return df.groupby("wavenumber", as_index=False).mean(numeric_only=True)


def prediction_uncertainty(row: pd.Series, temperature_k: float) -> tuple[float, float, float]:
    temperatures = np.asarray([273.0, 303.0, 333.0], dtype=float)
    uncertainties = np.asarray(
        [
            row["u_B_273K_500Torr"],
            row["u_B_303K_weighted_scaled"],
            row["u_B_333K_500Torr"],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
        return np.nan, np.nan, np.nan
    x = np.column_stack([np.ones_like(temperatures), temperatures])
    weights = 1.0 / uncertainties**2
    cov_unscaled = np.linalg.inv(x.T @ (weights[:, None] * x))
    birge = float(row.get("fit_birge_ratio", 1.0))
    if not np.isfinite(birge) or birge < 1.0:
        birge = 1.0
    cov_scaled = cov_unscaled * birge**2
    v = np.asarray([1.0, temperature_k], dtype=float)
    u_unscaled = float(np.sqrt(v @ cov_unscaled @ v))
    u_scaled = float(np.sqrt(v @ cov_scaled @ v))
    return u_unscaled, u_scaled, birge


def calculate(fit: pd.DataFrame, measured: pd.DataFrame, temperature_k: float) -> pd.DataFrame:
    work = fit.copy()
    work["wavenumber"] = pd.to_numeric(work["wavenumber"], errors="coerce").round(2)
    pred = pd.DataFrame(
        {
            "wavenumber": work["wavenumber"],
            "B_296K_predicted": work["fit_intercept"] + work["dB_dT_cm_inv_amagat_neg2_K_neg1"] * temperature_k,
            "fit_intercept": work["fit_intercept"],
            "dB_dT_cm_inv_amagat_neg2_K_neg1": work["dB_dT_cm_inv_amagat_neg2_K_neg1"],
            "u_dB_dT_cm_inv_amagat_neg2_K_neg1": work["u_dB_dT_cm_inv_amagat_neg2_K_neg1"],
            "fit_birge_ratio": work["fit_birge_ratio"],
            "B_273K_500Torr": work["B_273K_500Torr"],
            "u_B_273K_500Torr": work["u_B_273K_500Torr"],
            "B_303K_weighted": work["B_303K_weighted"],
            "u_B_303K_weighted_scaled": work["u_B_303K_weighted_scaled"],
            "B_333K_500Torr": work["B_333K_500Torr"],
            "u_B_333K_500Torr": work["u_B_333K_500Torr"],
        }
    )
    uncertainties = work.apply(lambda row: prediction_uncertainty(row, temperature_k), axis=1)
    pred["u_B_296K_predicted_unscaled"] = [item[0] for item in uncertainties]
    pred["u_B_296K_predicted"] = [item[1] for item in uncertainties]
    pred["prediction_birge_ratio"] = [item[2] for item in uncertainties]
    pred = pred.groupby("wavenumber", as_index=False).mean(numeric_only=True)

    out = pred.merge(measured, on="wavenumber", how="inner")
    out["B_residual_measured_minus_predicted"] = (
        out["B_296K_measured_500Torr"] - out["B_296K_predicted"]
    )
    out["B_residual_rel_percent"] = (
        out["B_residual_measured_minus_predicted"] / out["B_296K_predicted"] * 100.0
    )
    out["abs_residual_rel_percent"] = out["B_residual_rel_percent"].abs()
    out["residual_over_u_predicted"] = (
        out["B_residual_measured_minus_predicted"] / out["u_B_296K_predicted"]
    )
    out["abs_residual_over_u_predicted"] = out["residual_over_u_predicted"].abs()
    if "u_B_296K_measured_500Torr" in out.columns:
        out["u_B_296K_residual_combined"] = np.sqrt(
            out["u_B_296K_predicted"] ** 2 + out["u_B_296K_measured_500Torr"] ** 2
        )
        out["residual_over_u_combined"] = (
            out["B_residual_measured_minus_predicted"] / out["u_B_296K_residual_combined"]
        )
        out["abs_residual_over_u_combined"] = out["residual_over_u_combined"].abs()
    else:
        out["u_B_296K_measured_500Torr"] = np.nan
        out["u_B_296K_residual_combined"] = out["u_B_296K_predicted"]
        out["residual_over_u_combined"] = out["residual_over_u_predicted"]
        out["abs_residual_over_u_combined"] = out["abs_residual_over_u_predicted"]
    out["u_B_296K_predicted_rel_percent"] = (
        out["u_B_296K_predicted"] / out["B_296K_predicted"] * 100.0
    )
    out["u_B_296K_measured_rel_percent"] = (
        out["u_B_296K_measured_500Torr"] / out["B_296K_measured_500Torr"] * 100.0
    )
    out["u_B_296K_residual_combined_rel_percent"] = (
        out["u_B_296K_residual_combined"] / out["B_296K_predicted"] * 100.0
    )
    return out


def integrate_trapezoid(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return np.nan
    order = np.argsort(x_arr)
    return float(np.trapezoid(y_arr[order], x_arr[order]))


def summarize(out: pd.DataFrame, fit_csv: Path, measured_path: Path) -> pd.DataFrame:
    residual = out["B_residual_measured_minus_predicted"]
    rel = out["B_residual_rel_percent"]
    z = out["abs_residual_over_u_predicted"]
    z_combined = out["abs_residual_over_u_combined"]
    max_abs = out.loc[residual.abs().idxmax()]
    max_rel = out.loc[rel.abs().idxmax()]
    max_z = out.loc[z.idxmax()]
    max_z_combined = out.loc[z_combined.idxmax()]
    wn = out["wavenumber"]
    s_predicted = integrate_trapezoid(wn, out["B_296K_predicted"])
    s_measured = integrate_trapezoid(wn, out["B_296K_measured_500Torr"])
    s_residual = s_measured - s_predicted
    s_residual_rel_percent = s_residual / s_predicted * 100.0
    u_s_predicted_1sigma = integrate_trapezoid(wn, out["u_B_296K_predicted"])
    u_s_measured_1sigma = integrate_trapezoid(wn, out["u_B_296K_measured_500Torr"])
    u_s_combined_1sigma = float(np.sqrt(u_s_predicted_1sigma**2 + u_s_measured_1sigma**2))
    return pd.DataFrame(
        [
            {
                "n_points": int(len(out)),
                "wavenumber_min": float(out["wavenumber"].min()),
                "wavenumber_max": float(out["wavenumber"].max()),
                "B_predicted_mean": float(out["B_296K_predicted"].mean()),
                "B_measured_mean": float(out["B_296K_measured_500Torr"].mean()),
                "S_296K_predicted": s_predicted,
                "u_S_296K_predicted_1sigma": u_s_predicted_1sigma,
                "U_S_296K_predicted_2sigma": 2.0 * u_s_predicted_1sigma,
                "S_296K_predicted_lower_2sigma": s_predicted - 2.0 * u_s_predicted_1sigma,
                "S_296K_predicted_upper_2sigma": s_predicted + 2.0 * u_s_predicted_1sigma,
                "u_S_296K_predicted_rel_percent_1sigma": u_s_predicted_1sigma / s_predicted * 100.0,
                "U_S_296K_predicted_rel_percent_2sigma": 2.0 * u_s_predicted_1sigma / s_predicted * 100.0,
                "S_296K_measured_500Torr": s_measured,
                "u_S_296K_measured_1sigma": u_s_measured_1sigma,
                "U_S_296K_measured_2sigma": 2.0 * u_s_measured_1sigma,
                "u_S_296K_measured_rel_percent_1sigma": u_s_measured_1sigma / s_measured * 100.0,
                "U_S_296K_measured_rel_percent_2sigma": 2.0 * u_s_measured_1sigma / s_measured * 100.0,
                "S_residual_measured_minus_predicted": s_residual,
                "S_residual_rel_percent": s_residual_rel_percent,
                "u_S_residual_combined_1sigma": u_s_combined_1sigma,
                "U_S_residual_combined_2sigma": 2.0 * u_s_combined_1sigma,
                "u_S_residual_combined_rel_percent_1sigma": u_s_combined_1sigma / s_predicted * 100.0,
                "U_S_residual_combined_rel_percent_2sigma": 2.0 * u_s_combined_1sigma / s_predicted * 100.0,
                "residual_mean": float(residual.mean()),
                "residual_median": float(residual.median()),
                "residual_rmse": float(np.sqrt(np.mean(residual**2))),
                "residual_rel_mean_percent": float(rel.mean()),
                "residual_rel_median_percent": float(rel.median()),
                "abs_residual_rel_median_percent": float(rel.abs().median()),
                "abs_residual_rel_mean_percent": float(rel.abs().mean()),
                "max_abs_residual": float(max_abs["B_residual_measured_minus_predicted"]),
                "max_abs_residual_wavenumber": float(max_abs["wavenumber"]),
                "max_abs_residual_rel_percent": float(max_rel["abs_residual_rel_percent"]),
                "max_abs_residual_rel_wavenumber": float(max_rel["wavenumber"]),
                "median_u_B_predicted": float(out["u_B_296K_predicted"].median()),
                "mean_u_B_predicted": float(out["u_B_296K_predicted"].mean()),
                "median_u_B_measured": float(out["u_B_296K_measured_500Torr"].median()),
                "mean_u_B_measured": float(out["u_B_296K_measured_500Torr"].mean()),
                "median_u_B_residual_combined": float(out["u_B_296K_residual_combined"].median()),
                "mean_u_B_residual_combined": float(out["u_B_296K_residual_combined"].mean()),
                "median_u_B_predicted_rel_percent": float(out["u_B_296K_predicted_rel_percent"].median()),
                "mean_u_B_predicted_rel_percent": float(out["u_B_296K_predicted_rel_percent"].mean()),
                "median_u_B_measured_rel_percent": float(out["u_B_296K_measured_rel_percent"].median()),
                "mean_u_B_measured_rel_percent": float(out["u_B_296K_measured_rel_percent"].mean()),
                "median_u_B_residual_combined_rel_percent": float(
                    out["u_B_296K_residual_combined_rel_percent"].median()
                ),
                "mean_u_B_residual_combined_rel_percent": float(
                    out["u_B_296K_residual_combined_rel_percent"].mean()
                ),
                "fraction_within_1u_predicted": float((z <= 1.0).mean()),
                "fraction_within_2u_predicted": float((z <= 2.0).mean()),
                "max_abs_residual_over_u_predicted": float(max_z["abs_residual_over_u_predicted"]),
                "max_abs_residual_over_u_wavenumber": float(max_z["wavenumber"]),
                "fraction_within_1u_combined": float((z_combined <= 1.0).mean()),
                "fraction_within_2u_combined": float((z_combined <= 2.0).mean()),
                "max_abs_residual_over_u_combined": float(max_z_combined["abs_residual_over_u_combined"]),
                "max_abs_residual_over_u_combined_wavenumber": float(max_z_combined["wavenumber"]),
                "fit_source": str(fit_csv),
                "measured_source": str(measured_path),
            }
        ]
    )


def mm_to_inch(value_mm: float) -> float:
    return value_mm / 25.4


def plot_comparison(
    out: pd.DataFrame,
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
        figsize=(mm_to_inch(figure_width_mm), mm_to_inch(figure_height_mm)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.975, hspace=0.08)
    x = out["wavenumber"].to_numpy(dtype=float)
    pred = out["B_296K_predicted"].to_numpy(dtype=float)
    measured = out["B_296K_measured_500Torr"].to_numpy(dtype=float)
    relative_error = out["B_residual_rel_percent"].to_numpy(dtype=float)
    relative_uncertainty = out["u_B_296K_residual_combined_rel_percent"].to_numpy(dtype=float)

    scale_b = 1e-6
    blue = "#0072B2"
    red = "#D55E00"
    axes[0].plot(x, pred / scale_b, color=blue, lw=0.8, label="Predicted 296 K")
    axes[0].plot(x, measured / scale_b, color=red, lw=0.8, alpha=0.9, label="Measured 296 K, 500 Torr")
    axes[0].set_ylabel(r"$B$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    axes[0].legend(frameon=False, loc="best")
    axes[0].minorticks_on()

    axes[1].fill_between(
        x,
        -2.0 * relative_uncertainty,
        2.0 * relative_uncertainty,
        color="#BDBDBD",
        alpha=0.45,
        lw=0,
        zorder=1,
        label=r"$\pm 2\sigma$",
    )
    axes[1].axhline(0, color="black", lw=0.55, zorder=2)
    axes[1].plot(x, relative_error, color="black", lw=0.7, label="Relative error", zorder=3)
    axes[1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Relative error (%)")
    axes[1].legend(frameon=False, loc="best")
    axes[1].minorticks_on()
    fig.savefig(output_png, dpi=dpi)
    fig.savefig(output_png.with_suffix(".pdf"))
    fig.savefig(output_png.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_csv = args.fit_csv.expanduser().resolve()
    measured_path = args.measured.expanduser().resolve()
    fit = pd.read_csv(fit_csv)
    measured = read_measured(measured_path, args.encoding)
    out = calculate(fit, measured, args.temperature_k)
    summary = summarize(out, fit_csv, measured_path)

    output_csv = output_dir / "B_296K_predicted_vs_measured_500Torr.csv"
    summary_csv = output_dir / "B_296K_predicted_vs_measured_500Torr_summary.csv"
    output_png = output_dir / "B_296K_predicted_vs_measured_500Torr.png"
    out.to_csv(output_csv, index=False, float_format="%.15g")
    summary.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_comparison(out, output_png, args.dpi, args.figure_width_mm, args.figure_height_mm)

    print(f"Comparison CSV: {output_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {output_png}")
    print(f"Figure PDF: {output_png.with_suffix('.pdf')}")
    print(f"Figure TIFF: {output_png.with_suffix('.tif')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
