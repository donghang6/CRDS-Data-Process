#!/usr/bin/env python3
"""Calculate and plot Karman O2-O2 CIA temperature dependence.

The input Karman theory files are expected to contain two columns:

    wavenumber   B_native

By default, B_native is treated as cm^5 molecule^-2 and converted to
cm^-1 amagat^-2 by multiplying by Loschmidt_number^2. Then each wavenumber is
fitted independently with:

    B(nu, T) = intercept(nu) + dB_dT(nu) * T

Outputs:
    karman_temperature_dependence_fit.csv
    karman_temperature_dependence_selected_wavenumbers.csv
    karman_temperature_dependence.png

Usage:
    python scripts/plot_karman_temperature_dependence.py

If the second column is already in cm^-1 amagat^-2:
    python scripts/plot_karman_temperature_dependence.py \
      --input-unit cm_inv_amagat_neg2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUTS = (
    Path("/Users/donghang/科研/实验数据/氧气连续吸收/理论计算/O2-O2 理论计算/9060_9660_276k.txt"),
    Path("/Users/donghang/科研/实验数据/氧气连续吸收/理论计算/O2-O2 理论计算/9060_9660_306k.txt"),
    Path("/Users/donghang/科研/实验数据/氧气连续吸收/理论计算/O2-O2 理论计算/9060_9660_336k.txt"),
)
DEFAULT_TEMPERATURES = (276.0, 306.0, 336.0)
LOSCHMIDT_CM3 = 2.686780111e19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Karman theory dB/dT for O2-O2 CIA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs=3,
        default=list(DEFAULT_INPUTS),
        help="Three theory files ordered by temperature.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs=3,
        default=list(DEFAULT_TEMPERATURES),
        help="Temperatures corresponding to --inputs.",
    )
    parser.add_argument(
        "--input-unit",
        choices=("cm5_molecule_neg2", "cm_inv_amagat_neg2"),
        default="cm5_molecule_neg2",
        help="Unit of the second column in the input files.",
    )
    parser.add_argument(
        "--loschmidt-cm3",
        type=float,
        default=LOSCHMIDT_CM3,
        help="Loschmidt number in molecule cm^-3 for 1 amagat.",
    )
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Selected wavenumbers for the left panel. If omitted, points are "
            "chosen automatically from the available range and dB/dT curve."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/Karman_temperature_dependence"),
        help="Output directory.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def read_theory_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path.expanduser())
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    return data[:, 0].astype(float), data[:, 1].astype(float)


def load_theory(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelengths = []
    b_native = []
    for path in args.inputs:
        x, y = read_theory_file(path)
        wavelengths.append(x)
        b_native.append(y)

    reference_x = wavelengths[0]
    for path, x in zip(args.inputs[1:], wavelengths[1:]):
        if len(x) != len(reference_x) or not np.allclose(x, reference_x, rtol=0, atol=1e-8):
            raise ValueError(f"Wavenumber grid does not match the first file: {path}")

    native = np.vstack(b_native)
    if args.input_unit == "cm5_molecule_neg2":
        converted = native * args.loschmidt_cm3**2
    else:
        converted = native.copy()
    return reference_x, native, converted


def linear_fit_by_wavenumber(
    wavenumber: np.ndarray,
    temperatures: np.ndarray,
    b_values: np.ndarray,
    b_native: np.ndarray,
) -> pd.DataFrame:
    x_mean = float(np.mean(temperatures))
    denom = float(np.sum((temperatures - x_mean) ** 2))
    slope_coeff = (temperatures - x_mean) / denom
    intercept_coeff = np.full_like(temperatures, 1.0 / len(temperatures)) - x_mean * slope_coeff

    slope = np.sum(slope_coeff[:, None] * b_values, axis=0)
    intercept = np.sum(intercept_coeff[:, None] * b_values, axis=0)
    fitted = intercept[None, :] + slope[None, :] * temperatures[:, None]
    residual = b_values - fitted
    dof = max(len(temperatures) - 2, 1)
    rmse = np.sqrt(np.sum(residual**2, axis=0) / dof)
    slope_stderr = rmse / np.sqrt(denom)

    return pd.DataFrame(
        {
            "wavenumber": wavenumber,
            "B_276K_native": b_native[0],
            "B_306K_native": b_native[1],
            "B_336K_native": b_native[2],
            "B_276K": b_values[0],
            "B_306K": b_values[1],
            "B_336K": b_values[2],
            "temperature_fit_mode": "unweighted_linear",
            "dB_dT_cm_inv_amagat_neg2_K_neg1": slope,
            "fit_intercept": intercept,
            "fit_rmse_cm_inv_amagat_neg2": rmse,
            "slope_stderr_from_fit_residual": slope_stderr,
        }
    )


def automatic_selected_wavenumbers(fit_df: pd.DataFrame) -> list[float]:
    x = fit_df["wavenumber"].to_numpy(dtype=float)
    slope = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float)
    b306 = fit_df["B_306K"].to_numpy(dtype=float)

    left = 9200.0 if np.nanmin(x) <= 9200.0 <= np.nanmax(x) else float(x[0])
    b_peak = float(x[int(np.nanargmax(b306))])
    slope_peak = float(x[int(np.nanargmax(slope))])
    right_mid = 9600.0 if np.nanmin(x) <= 9600.0 <= np.nanmax(x) else float(x[-1])
    right_edge = 9650.0 if np.nanmin(x) <= 9650.0 <= np.nanmax(x) else float(x[-1])
    return sorted(set([left, b_peak, slope_peak, right_mid, right_edge]))


def nearest_rows(fit_df: pd.DataFrame, selected: list[float]) -> pd.DataFrame:
    x = fit_df["wavenumber"].to_numpy(dtype=float)
    rows = []
    for target in selected:
        idx = int(np.nanargmin(np.abs(x - target)))
        row = fit_df.iloc[idx].copy()
        row["selected_target_wavenumber"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def plot_temperature_dependence(
    fit_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    temperatures: np.ndarray,
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
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2))]
    b_cols = [f"B_{int(t)}K" for t in temperatures]
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        b_vals = row[b_cols].to_numpy(dtype=float)
        label = f"{x0:.1f} cm$^{{-1}}$"
        t_line = np.linspace(float(np.nanmin(temperatures)) - 5, float(np.nanmax(temperatures)) + 5, 100)
        fit_line = row["fit_intercept"] + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        ax_left.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=1.0,
            label=label,
        )
        ax_left.plot(
            temperatures,
            b_vals / 1e-6,
            marker=markers[i % len(markers)],
            color="black",
            linestyle="none",
            ms=4.5,
            zorder=4,
        )

    ax_left.set_xlabel("Temperature (K)")
    ax_left.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_left.set_xlim(float(np.nanmin(temperatures)) - 10, float(np.nanmax(temperatures)) + 10)
    ax_left.legend(frameon=False, loc="best", fontsize=8)
    ax_left.minorticks_on()

    x = fit_df["wavenumber"].to_numpy(dtype=float)
    slope_scaled = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    ax_right.plot(x, slope_scaled, color="#1f77b4", lw=1.3, label="Karman")
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

    temperatures = np.asarray(args.temperatures, dtype=float)
    wavenumber, b_native, b_values = load_theory(args)
    fit_df = linear_fit_by_wavenumber(wavenumber, temperatures, b_values, b_native)
    selected = args.selected if args.selected is not None else automatic_selected_wavenumbers(fit_df)
    selected_df = nearest_rows(fit_df, selected)

    fit_csv = output_dir / "karman_temperature_dependence_fit.csv"
    selected_csv = output_dir / "karman_temperature_dependence_selected_wavenumbers.csv"
    output_png = output_dir / "karman_temperature_dependence.png"
    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    plot_temperature_dependence(fit_df, selected_df, temperatures, output_png, args.dpi)

    print(f"Fit CSV: {fit_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Figure: {output_png}")
    peak = fit_df.loc[fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].idxmax()]
    print(
        "Max dB/dT: "
        f"{peak['wavenumber']:.2f} cm^-1, "
        f"{peak['dB_dT_cm_inv_amagat_neg2_K_neg1']:.6e} cm^-1 amagat^-2 K^-1"
    )
    print("Selected wavenumbers:")
    cols = [
        "wavenumber",
        "B_276K",
        "B_306K",
        "B_336K",
        "dB_dT_cm_inv_amagat_neg2_K_neg1",
    ]
    print(selected_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
