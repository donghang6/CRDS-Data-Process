#!/usr/bin/env python3
"""Analyze whether Karman ab initio O2-O2 CIA data are linear in temperature.

The script scans files named like:

    9060_9660_206k.txt
    9060_9660_296.15k.txt

Each file is expected to have two columns:

    wavenumber   B

The second column may be either cm^5 molecule^-2 or cm^-1 amagat^-2. In the
default auto mode, values smaller than 1e-20 are treated as cm^5 molecule^-2
and converted with Loschmidt_number^2.

For every common wavenumber, the script fits:

    linear:    B(T) = a + b T
    quadratic: B(T) = a + b T + c T^2

It then reports R^2, residuals, and the improvement obtained by adding the
quadratic term.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("/Users/donghang/科研/实验数据/氧气连续吸收/理论计算/O2-O2 理论计算")
LOSCHMIDT_CM3 = 2.686780111e19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Karman ab initio B(T) linearity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory.")
    parser.add_argument("--pattern", default="9060_9660_*k.txt", help="Input file glob pattern.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/Karman_temperature_linearity"),
        help="Output directory.",
    )
    parser.add_argument(
        "--input-unit",
        choices=("auto", "cm5_molecule_neg2", "cm_inv_amagat_neg2"),
        default="auto",
        help="Unit of second column.",
    )
    parser.add_argument(
        "--auto-native-threshold",
        type=float,
        default=1e-20,
        help="Median absolute value below this is treated as cm^5 molecule^-2 in auto mode.",
    )
    parser.add_argument("--temperature-min", type=float, default=None, help="Minimum temperature to include.")
    parser.add_argument("--temperature-max", type=float, default=None, help="Maximum temperature to include.")
    parser.add_argument("--loschmidt-cm3", type=float, default=LOSCHMIDT_CM3)
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=[9200.0, 9320.0, 9380.0, 9515.0, 9600.0],
        help="Selected wavenumbers for B(T) and residual plots.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_temperature(path: Path) -> float:
    match = re.search(r"_(\d+(?:\.\d+)?)k\.txt$", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse temperature from file name: {path.name}")
    return float(match.group(1))


def read_one_file(path: Path, args: argparse.Namespace) -> tuple[float, pd.DataFrame, str]:
    temperature = parse_temperature(path)
    data = np.loadtxt(path.expanduser())
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected two numeric columns in {path}")
    wavenumber = np.round(data[:, 0].astype(float), 4)
    values = data[:, 1].astype(float)

    if args.input_unit == "auto":
        median_abs = float(np.nanmedian(np.abs(values)))
        unit = "cm5_molecule_neg2" if median_abs < args.auto_native_threshold else "cm_inv_amagat_neg2"
    else:
        unit = args.input_unit

    if unit == "cm5_molecule_neg2":
        converted = values * args.loschmidt_cm3**2
    else:
        converted = values.copy()

    df = pd.DataFrame({"wavenumber": wavenumber, f"B_{temperature:g}K": converted})
    return temperature, df, unit


def load_all(args: argparse.Namespace) -> tuple[pd.DataFrame, list[float], pd.DataFrame]:
    paths = []
    for path in args.input_dir.expanduser().glob(args.pattern):
        temperature = parse_temperature(path)
        if args.temperature_min is not None and temperature < args.temperature_min:
            continue
        if args.temperature_max is not None and temperature > args.temperature_max:
            continue
        paths.append(path)
    paths = sorted(paths, key=parse_temperature)
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}")

    frames = []
    metadata = []
    for path in paths:
        temperature, frame, unit = read_one_file(path, args)
        frames.append(frame)
        metadata.append({"temperature_k": temperature, "file": str(path), "detected_unit": unit})

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="wavenumber", how="inner")
    temperatures = [row["temperature_k"] for row in metadata]
    return merged, temperatures, pd.DataFrame(metadata)


def fit_by_wavenumber(data: pd.DataFrame, temperatures: list[float]) -> pd.DataFrame:
    temps = np.asarray(temperatures, dtype=float)
    y_cols = [f"B_{t:g}K" for t in temperatures]
    y = data[y_cols].to_numpy(dtype=float)

    x_lin = np.vstack([np.ones_like(temps), temps]).T
    beta_lin = np.linalg.lstsq(x_lin, y.T, rcond=None)[0]
    y_lin = (x_lin @ beta_lin).T
    residual_lin = y - y_lin

    x_quad = np.vstack([np.ones_like(temps), temps, temps**2]).T
    beta_quad = np.linalg.lstsq(x_quad, y.T, rcond=None)[0]
    y_quad = (x_quad @ beta_quad).T
    residual_quad = y - y_quad

    y_mean = np.mean(y, axis=1, keepdims=True)
    ss_tot = np.sum((y - y_mean) ** 2, axis=1)
    ss_lin = np.sum(residual_lin**2, axis=1)
    ss_quad = np.sum(residual_quad**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_lin = 1.0 - ss_lin / ss_tot
        r2_quad = 1.0 - ss_quad / ss_tot
        rmse_lin = np.sqrt(ss_lin / len(temps))
        rmse_quad = np.sqrt(ss_quad / len(temps))
        mean_b = np.mean(y, axis=1)
        rmse_lin_rel_percent = 100.0 * rmse_lin / np.abs(mean_b)
        rmse_quad_rel_percent = 100.0 * rmse_quad / np.abs(mean_b)
        max_abs_resid_lin_rel_percent = 100.0 * np.max(np.abs(residual_lin), axis=1) / np.abs(mean_b)
        quadratic_rmse_improvement_percent = 100.0 * (rmse_lin - rmse_quad) / rmse_lin

    out = pd.DataFrame(
        {
            "wavenumber": data["wavenumber"].to_numpy(dtype=float),
            "linear_intercept": beta_lin[0],
            "linear_dB_dT_cm_inv_amagat_neg2_K_neg1": beta_lin[1],
            "quadratic_intercept": beta_quad[0],
            "quadratic_linear_term": beta_quad[1],
            "quadratic_T2_term": beta_quad[2],
            "linear_R2": r2_lin,
            "quadratic_R2": r2_quad,
            "linear_rmse_cm_inv_amagat_neg2": rmse_lin,
            "quadratic_rmse_cm_inv_amagat_neg2": rmse_quad,
            "linear_rmse_rel_percent": rmse_lin_rel_percent,
            "quadratic_rmse_rel_percent": rmse_quad_rel_percent,
            "linear_max_abs_residual_rel_percent": max_abs_resid_lin_rel_percent,
            "quadratic_rmse_improvement_percent": quadratic_rmse_improvement_percent,
            "linear_residual_at_minT": residual_lin[:, 0],
            "linear_residual_at_maxT": residual_lin[:, -1],
        }
    )
    return pd.concat([out, data.drop(columns=["wavenumber"])], axis=1)


def nearest_rows(fit_df: pd.DataFrame, selected: list[float]) -> pd.DataFrame:
    x = fit_df["wavenumber"].to_numpy(dtype=float)
    rows = []
    for target in selected:
        idx = int(np.nanargmin(np.abs(x - target)))
        row = fit_df.iloc[idx].copy()
        row["selected_target_wavenumber"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(fit_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    slope = fit_df["linear_dB_dT_cm_inv_amagat_neg2_K_neg1"]
    r2 = fit_df["linear_R2"]
    rel = fit_df["linear_rmse_rel_percent"]
    max_rel = fit_df["linear_max_abs_residual_rel_percent"]
    improvement = fit_df["quadratic_rmse_improvement_percent"]
    return pd.DataFrame(
        [
            {
                "n_temperature_files": len(metadata),
                "temperature_min_k": metadata["temperature_k"].min(),
                "temperature_max_k": metadata["temperature_k"].max(),
                "n_common_wavenumbers": len(fit_df),
                "wavenumber_min": fit_df["wavenumber"].min(),
                "wavenumber_max": fit_df["wavenumber"].max(),
                "linear_R2_min": r2.min(),
                "linear_R2_median": r2.median(),
                "linear_R2_mean": r2.mean(),
                "linear_rmse_rel_percent_median": rel.median(),
                "linear_rmse_rel_percent_95pct": rel.quantile(0.95),
                "linear_max_abs_residual_rel_percent_median": max_rel.median(),
                "linear_max_abs_residual_rel_percent_95pct": max_rel.quantile(0.95),
                "quadratic_rmse_improvement_percent_median": improvement.median(),
                "quadratic_rmse_improvement_percent_95pct": improvement.quantile(0.95),
                "linear_dB_dT_max": slope.max(),
                "linear_dB_dT_max_wavenumber": fit_df.loc[slope.idxmax(), "wavenumber"],
                "linear_dB_dT_min": slope.min(),
                "linear_dB_dT_min_wavenumber": fit_df.loc[slope.idxmin(), "wavenumber"],
            }
        ]
    )


def plot_linearity(
    fit_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    temperatures: list[float],
    output_png: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    ax_bt, ax_resid, ax_slope, ax_r2 = axes.ravel()

    temps = np.asarray(temperatures, dtype=float)
    b_cols = [f"B_{t:g}K" for t in temperatures]
    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2))]
    for i, (_, row) in enumerate(selected_df.iterrows()):
        b_vals = row[b_cols].to_numpy(dtype=float)
        t_line = np.linspace(float(np.nanmin(temps)), float(np.nanmax(temps)), 200)
        fit_line = row["linear_intercept"] + row["linear_dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        residual = b_vals - (
            row["linear_intercept"] + row["linear_dB_dT_cm_inv_amagat_neg2_K_neg1"] * temps
        )
        label = f"{row['wavenumber']:.1f} cm$^{{-1}}$"
        ax_bt.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=0.9,
            label=label,
        )
        ax_bt.plot(
            temps,
            b_vals / 1e-6,
            marker=markers[i % len(markers)],
            color="black",
            linestyle="none",
            ms=3.5,
        )
        ax_resid.plot(
            temps,
            residual / 1e-6,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color="black",
            lw=0.8,
            ms=3.0,
            label=label,
        )

    ax_bt.set_xlabel("Temperature (K)")
    ax_bt.set_ylabel(r"$B$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_bt.legend(frameon=False, fontsize=7)
    ax_bt.minorticks_on()

    ax_resid.axhline(0, color="0.4", lw=0.8)
    ax_resid.set_xlabel("Temperature (K)")
    ax_resid.set_ylabel(r"Linear residual ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_resid.legend(frameon=False, fontsize=7)
    ax_resid.minorticks_on()

    x = fit_df["wavenumber"].to_numpy(dtype=float)
    ax_slope.plot(
        x,
        fit_df["linear_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9,
        color="#1f77b4",
        lw=1.2,
    )
    ax_slope.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_slope.set_ylabel(r"Linear $dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_slope.minorticks_on()

    ax_r2.plot(x, fit_df["linear_R2"].to_numpy(dtype=float), color="black", lw=1.0)
    ax_r2.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_r2.set_ylabel(r"Linear fit $R^2$")
    ax_r2.set_ylim(max(0.0, float(np.nanmin(fit_df["linear_R2"])) - 0.002), 1.0002)
    ax_r2.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, temperatures, metadata = load_all(args)
    fit_df = fit_by_wavenumber(data, temperatures)
    selected_df = nearest_rows(fit_df, args.selected)
    summary_df = summarize(fit_df, metadata)

    metadata_csv = output_dir / "karman_temperature_linearity_input_files.csv"
    fit_csv = output_dir / "karman_temperature_linearity_fit.csv"
    selected_csv = output_dir / "karman_temperature_linearity_selected_wavenumbers.csv"
    summary_csv = output_dir / "karman_temperature_linearity_summary.csv"
    output_png = output_dir / "karman_temperature_linearity.png"

    metadata.to_csv(metadata_csv, index=False)
    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    summary_df.to_csv(summary_csv, index=False, float_format="%.15g")
    plot_linearity(fit_df, selected_df, temperatures, output_png, args.dpi)

    print(f"Input metadata: {metadata_csv}")
    print(f"Fit CSV: {fit_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {output_png}")
    print(summary_df.to_string(index=False))
    print("Detected units:")
    print(metadata[["temperature_k", "detected_unit", "file"]].to_string(index=False))
    print("Selected wavenumbers:")
    cols = [
        "wavenumber",
        "linear_dB_dT_cm_inv_amagat_neg2_K_neg1",
        "linear_R2",
        "linear_rmse_rel_percent",
        "quadratic_rmse_improvement_percent",
    ]
    print(selected_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
