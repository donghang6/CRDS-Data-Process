#!/usr/bin/env python3
"""Analyze temperature dependence coefficient dB/dT from a B coefficient table.

The default input is the user's final B table with columns:

    波数, B 273, B 303 500, B 303 600, B 303 700, B 333 500

For each wavenumber, the 303 K value is either the mean of the three pressures
or one specified pressure group. Then a linear fit is applied:

    B(nu, T) = intercept(nu) + dB_dT(nu) * T

The default mode uses 273, 303, and 333 K. A two-point slope can be calculated
with, for example:

    --temperature-mode 273-303

Outputs:
    b_temperature_dependence_fit.csv
    b_temperature_dependence_selected_wavenumbers.csv
    b_temperature_dependence_summary.csv
    b_temperature_dependence.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/B .txt")
DEFAULT_SELECTED = (9200.0, 9322.95, 9427.60, 9700.0, 9800.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dB/dT from final O2-O2 CIA B table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input B table.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/B_temperature_dependence"),
        help="Output directory.",
    )
    parser.add_argument(
        "--encoding",
        default="gbk",
        help="Text encoding of the input table.",
    )
    parser.add_argument(
        "--use-303",
        choices=("mean", "500", "600", "700"),
        default="mean",
        help="How to use the three 303 K pressure groups.",
    )
    parser.add_argument(
        "--temperature-mode",
        choices=("all", "273-303", "303-333", "273-333"),
        default="all",
        help="Temperature points used to calculate dB/dT.",
    )
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=list(DEFAULT_SELECTED),
        help="Selected wavenumbers for the left-panel B-vs-T plot.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def read_b_table(path: Path, encoding: str) -> pd.DataFrame:
    raw = pd.read_csv(path.expanduser(), sep="\t", encoding=encoding)
    if raw.shape[1] < 6:
        raise ValueError(f"Expected at least 6 tab-separated columns in {path}")

    df = raw.iloc[:, :6].copy()
    df.columns = [
        "wavenumber",
        "B_273K",
        "B_303K_500Torr",
        "B_303K_600Torr",
        "B_303K_700Torr",
        "B_333K",
    ]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=df.columns).sort_values("wavenumber").reset_index(drop=True)
    return df


def prepare_fit_input(df: pd.DataFrame, use_303: str) -> pd.DataFrame:
    out = df.copy()
    b303_cols = ["B_303K_500Torr", "B_303K_600Torr", "B_303K_700Torr"]
    out["B_303K_mean"] = out[b303_cols].mean(axis=1)
    out["B_303K_std"] = out[b303_cols].std(axis=1, ddof=1)
    out["B_303K_sem"] = out["B_303K_std"] / np.sqrt(3.0)
    out["B_303K_pressure_rel_std_percent"] = 100.0 * out["B_303K_std"] / out["B_303K_mean"]

    if use_303 == "mean":
        out["B_303K_used"] = out["B_303K_mean"]
    else:
        out["B_303K_used"] = out[f"B_303K_{use_303}Torr"]
    out["B_303K_used_mode"] = use_303
    return out


def fit_temperature_columns(temperature_mode: str) -> tuple[np.ndarray, list[str], str]:
    if temperature_mode == "all":
        return np.asarray([273.0, 303.0, 333.0], dtype=float), [
            "B_273K",
            "B_303K_used",
            "B_333K",
        ], "unweighted_linear_273_303_333"
    if temperature_mode == "273-303":
        return np.asarray([273.0, 303.0], dtype=float), [
            "B_273K",
            "B_303K_used",
        ], "two_point_slope_273_303"
    if temperature_mode == "303-333":
        return np.asarray([303.0, 333.0], dtype=float), [
            "B_303K_used",
            "B_333K",
        ], "two_point_slope_303_333"
    if temperature_mode == "273-333":
        return np.asarray([273.0, 333.0], dtype=float), [
            "B_273K",
            "B_333K",
        ], "two_point_slope_273_333"
    raise ValueError(f"Unsupported temperature mode: {temperature_mode}")


def linear_fit_by_wavenumber(df: pd.DataFrame, temperature_mode: str) -> pd.DataFrame:
    temperatures, y_cols, fit_mode_label = fit_temperature_columns(temperature_mode)
    y = np.vstack([df[col].to_numpy(dtype=float) for col in y_cols])
    x_mean = float(np.mean(temperatures))
    denom = float(np.sum((temperatures - x_mean) ** 2))
    slope_coeff = (temperatures - x_mean) / denom
    intercept_coeff = np.full_like(temperatures, 1.0 / len(temperatures)) - x_mean * slope_coeff

    slope = np.sum(slope_coeff[:, None] * y, axis=0)
    intercept = np.sum(intercept_coeff[:, None] * y, axis=0)
    fitted = intercept[None, :] + slope[None, :] * temperatures[:, None]
    residual = y - fitted
    rmse = np.sqrt(np.sum(residual**2, axis=0))
    slope_stderr_from_fit_residual = rmse / np.sqrt(denom)

    out = df.copy()
    out["temperature_fit_mode"] = fit_mode_label
    out["fit_temperature_points"] = ",".join(f"{t:.0f}" for t in temperatures)
    out["fit_intercept"] = intercept
    out["dB_dT_cm_inv_amagat_neg2_K_neg1"] = slope
    out["fit_rmse_cm_inv_amagat_neg2"] = rmse
    out["slope_stderr_from_fit_residual"] = slope_stderr_from_fit_residual
    out["B_fit_273K"] = intercept + slope * 273.0
    out["B_fit_303K"] = intercept + slope * 303.0
    out["B_fit_333K"] = intercept + slope * 333.0
    out["B_residual_273K"] = out["B_273K"] - out["B_fit_273K"]
    out["B_residual_303K"] = out["B_303K_used"] - out["B_fit_303K"]
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
    b_peak_row = fit_df.loc[fit_df["B_303K_used"].idxmax()]
    return pd.DataFrame(
        [
            {
                "n_points": len(fit_df),
                "wavenumber_min": fit_df["wavenumber"].min(),
                "wavenumber_max": fit_df["wavenumber"].max(),
                "dB_dT_min": min_row["dB_dT_cm_inv_amagat_neg2_K_neg1"],
                "dB_dT_min_wavenumber": min_row["wavenumber"],
                "dB_dT_max": max_row["dB_dT_cm_inv_amagat_neg2_K_neg1"],
                "dB_dT_max_wavenumber": max_row["wavenumber"],
                "B_303K_peak": b_peak_row["B_303K_used"],
                "B_303K_peak_wavenumber": b_peak_row["wavenumber"],
                "B_303K_pressure_rel_std_percent_mean": fit_df[
                    "B_303K_pressure_rel_std_percent"
                ].mean(),
                "B_303K_pressure_rel_std_percent_max": fit_df[
                    "B_303K_pressure_rel_std_percent"
                ].max(),
            }
        ]
    )


def plot_temperature_dependence(
    fit_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_png: Path,
    dpi: int,
    temperature_mode: str,
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
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

    temps, y_cols, _ = fit_temperature_columns(temperature_mode)
    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2))]
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        b_vals = row[y_cols].to_numpy(dtype=float)
        t_line = np.linspace(float(np.nanmin(temps)) - 5.0, float(np.nanmax(temps)) + 5.0, 100)
        fit_line = row["fit_intercept"] + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        label = f"{x0:.1f} cm$^{{-1}}$"
        ax_left.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=1.0,
            label=label,
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
    ax_left.set_xlim(float(np.nanmin(temps)) - 10.0, float(np.nanmax(temps)) + 10.0)
    ax_left.legend(frameon=False, loc="best", fontsize=8)
    ax_left.minorticks_on()

    x = fit_df["wavenumber"].to_numpy(dtype=float)
    slope_scaled = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    ax_right.plot(x, slope_scaled, color="#1f77b4", lw=1.3, label=r"$B_{\mathrm{O_2-O_2}}$")
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        y0 = float(row["dB_dT_cm_inv_amagat_neg2_K_neg1"]) / 1e-9
        ax_right.axvline(x0, color="0.75", lw=0.8, ls="--", zorder=0)
        ax_right.plot(x0, y0, marker="o", ms=3.5, color="black", zorder=3)
        offset = 8 if i % 2 == 0 else -14
        ax_right.annotate(
            f"{x0:.0f}",
            xy=(x0, y0),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_b_table(args.input, args.encoding)
    prepared = prepare_fit_input(df, args.use_303)
    fit_df = linear_fit_by_wavenumber(prepared, args.temperature_mode)
    selected_df = nearest_rows(fit_df, args.selected)
    summary_df = summarize(fit_df)

    fit_csv = output_dir / "b_temperature_dependence_fit.csv"
    selected_csv = output_dir / "b_temperature_dependence_selected_wavenumbers.csv"
    summary_csv = output_dir / "b_temperature_dependence_summary.csv"
    output_png = output_dir / "b_temperature_dependence.png"
    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    summary_df.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_temperature_dependence(fit_df, selected_df, output_png, args.dpi, args.temperature_mode)

    print(f"Fit CSV: {fit_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary_df.to_string(index=False))
    print("Selected wavenumbers:")
    cols = [
        "wavenumber",
        "B_273K",
        "B_303K_used",
        "dB_dT_cm_inv_amagat_neg2_K_neg1",
    ]
    if args.temperature_mode != "273-303":
        cols.insert(3, "B_333K")
    print(selected_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
