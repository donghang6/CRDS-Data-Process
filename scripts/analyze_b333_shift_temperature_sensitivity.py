#!/usr/bin/env python3
"""Test how shifting measured 333 K B affects fitted dB/dT.

The input B table is expected to contain:

    波数, B 273, B 303 500, B 303 600, B 303 700, B 333 500

By default, the 303 K point is the mean of the three 303 K pressure groups.
For each requested relative shift, the 333 K data are modified as:

    B_333K_shifted = B_333K * (1 + relative_shift)

Then each wavenumber is fitted with the three points:

    B(nu, T) = intercept(nu) + dB_dT(nu) * T

This is a sensitivity test rather than a replacement for the final B table.
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
        description="Sensitivity of dB/dT to relative shifts of measured 333 K B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input measured B table.")
    parser.add_argument("--encoding", default="gbk", help="Input text encoding.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/B333_shift_temperature_sensitivity"),
        help="Output directory.",
    )
    parser.add_argument(
        "--use-303",
        choices=("mean", "500", "600", "700"),
        default="mean",
        help="How to use the three 303 K pressure groups.",
    )
    parser.add_argument(
        "--relative-shifts",
        type=float,
        nargs="+",
        default=[-0.02, -0.01, 0.0, 0.01, 0.02],
        help="Relative shifts applied to B_333K, e.g. 0.01 means +1 percent.",
    )
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=list(DEFAULT_SELECTED),
        help="Selected wavenumbers to report.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_b_table(path: Path, encoding: str) -> pd.DataFrame:
    raw = pd.read_csv(path.expanduser(), sep="\t", encoding=encoding)
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
    return df.dropna(subset=df.columns).sort_values("wavenumber").reset_index(drop=True)


def add_303_used(df: pd.DataFrame, use_303: str) -> pd.DataFrame:
    out = df.copy()
    b303_cols = ["B_303K_500Torr", "B_303K_600Torr", "B_303K_700Torr"]
    out["B_303K_mean"] = out[b303_cols].mean(axis=1)
    if use_303 == "mean":
        out["B_303K_used"] = out["B_303K_mean"]
    else:
        out["B_303K_used"] = out[f"B_303K_{use_303}Torr"]
    out["B_303K_used_mode"] = use_303
    return out


def fit_slope(df: pd.DataFrame, b333_shifted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures = np.asarray([273.0, 303.0, 333.0], dtype=float)
    y = np.vstack(
        [
            df["B_273K"].to_numpy(dtype=float),
            df["B_303K_used"].to_numpy(dtype=float),
            b333_shifted,
        ]
    )
    x_mean = float(np.mean(temperatures))
    denom = float(np.sum((temperatures - x_mean) ** 2))
    slope_coeff = (temperatures - x_mean) / denom
    intercept_coeff = np.full_like(temperatures, 1.0 / len(temperatures)) - x_mean * slope_coeff
    slope = np.sum(slope_coeff[:, None] * y, axis=0)
    intercept = np.sum(intercept_coeff[:, None] * y, axis=0)
    fitted = intercept[None, :] + slope[None, :] * temperatures[:, None]
    rmse = np.sqrt(np.sum((y - fitted) ** 2, axis=0))
    return intercept, slope, rmse


def nearest_selected(wavenumber: np.ndarray, selected: list[float]) -> list[int]:
    return [int(np.nanargmin(np.abs(wavenumber - target))) for target in selected]


def analyze(df: pd.DataFrame, relative_shifts: list[float], selected: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df["wavenumber"].to_numpy(dtype=float)
    base_b333 = df["B_333K"].to_numpy(dtype=float)
    result = pd.DataFrame({"wavenumber": x})
    selected_indices = nearest_selected(x, selected)
    selected_rows = []
    summary_rows = []

    for shift in sorted(relative_shifts):
        b333_shifted = base_b333 * (1.0 + shift)
        intercept, slope, rmse = fit_slope(df, b333_shifted)
        label = f"{shift:+.2%}"
        safe = label.replace("+", "plus_").replace("-", "minus_").replace(".", "p").replace("%", "pct")
        result[f"B333_shift_{safe}"] = b333_shifted
        result[f"dB_dT_{safe}"] = slope
        result[f"fit_intercept_{safe}"] = intercept
        result[f"fit_rmse_{safe}"] = rmse

        peak_idx = int(np.nanargmax(slope))
        summary_rows.append(
            {
                "B333_relative_shift": shift,
                "B333_shift_label": label,
                "dB_dT_max": slope[peak_idx],
                "dB_dT_max_wavenumber": x[peak_idx],
                "dB_dT_at_9200": slope[int(np.nanargmin(np.abs(x - 9200.0)))],
                "dB_dT_at_9322p95": slope[int(np.nanargmin(np.abs(x - 9322.95)))],
                "dB_dT_at_9427p60": slope[int(np.nanargmin(np.abs(x - 9427.60)))],
                "dB_dT_at_9700": slope[int(np.nanargmin(np.abs(x - 9700.0)))],
                "dB_dT_at_9800": slope[int(np.nanargmin(np.abs(x - 9800.0)))],
            }
        )
        for idx in selected_indices:
            selected_rows.append(
                {
                    "wavenumber": x[idx],
                    "selected_target_wavenumber": selected[selected_indices.index(idx)],
                    "B333_relative_shift": shift,
                    "B333_shift_label": label,
                    "B_273K": df.loc[idx, "B_273K"],
                    "B_303K_used": df.loc[idx, "B_303K_used"],
                    "B_333K_shifted": b333_shifted[idx],
                    "dB_dT_cm_inv_amagat_neg2_K_neg1": slope[idx],
                    "fit_intercept": intercept[idx],
                    "fit_rmse_cm_inv_amagat_neg2": rmse[idx],
                }
            )

    return result, pd.DataFrame(selected_rows), pd.DataFrame(summary_rows)


def plot_sensitivity(fit_df: pd.DataFrame, summary_df: pd.DataFrame, output_png: Path, dpi: int) -> None:
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
    fig, (ax_curve, ax_peak) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    x = fit_df["wavenumber"].to_numpy(dtype=float)

    slope_cols = [col for col in fit_df.columns if col.startswith("dB_dT_")]
    shifts = summary_df["B333_relative_shift"].to_numpy(dtype=float)
    cmap = plt.get_cmap("coolwarm")
    max_abs_shift = max(float(np.nanmax(np.abs(shifts))), 1e-12)
    for col, shift in zip(slope_cols, sorted(shifts)):
        color = cmap(0.5 + 0.5 * shift / max_abs_shift)
        label = f"{shift:+.0%}"
        lw = 1.8 if abs(shift) < 1e-12 else 1.0
        ax_curve.plot(x, fit_df[col].to_numpy(dtype=float) / 1e-9, color=color, lw=lw, label=label)

    ax_curve.axhline(0, color="black", lw=0.8)
    ax_curve.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_curve.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_curve.legend(title="333 K shift", frameon=False, fontsize=8)
    ax_curve.minorticks_on()

    ax_peak.plot(
        summary_df["B333_relative_shift"] * 100.0,
        summary_df["dB_dT_max"] / 1e-9,
        marker="o",
        color="black",
        lw=1.0,
    )
    for _, row in summary_df.iterrows():
        ax_peak.annotate(
            f"{row['dB_dT_max_wavenumber']:.0f}",
            xy=(row["B333_relative_shift"] * 100.0, row["dB_dT_max"] / 1e-9),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax_peak.set_xlabel("Relative shift of 333 K B (%)")
    ax_peak.set_ylabel(r"Maximum $dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_peak.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_303_used(read_b_table(args.input, args.encoding), args.use_303)
    fit_df, selected_df, summary_df = analyze(df, args.relative_shifts, args.selected)

    fit_csv = output_dir / "b333_shift_temperature_sensitivity_fit.csv"
    selected_csv = output_dir / "b333_shift_temperature_sensitivity_selected.csv"
    summary_csv = output_dir / "b333_shift_temperature_sensitivity_summary.csv"
    output_png = output_dir / "b333_shift_temperature_sensitivity.png"

    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    summary_df.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_sensitivity(fit_df, summary_df, output_png, args.dpi)

    print(f"Fit CSV: {fit_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
