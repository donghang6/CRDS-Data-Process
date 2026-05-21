#!/usr/bin/env python3
"""Plot integrated band strength S versus temperature with Karman data.

The integrated strength is calculated from B(nu):

    S = integral B(nu) dnu

For a fair comparison with the Karman ab initio data, all integrations use the
common measured/Karman wavenumber range.  With the current files this is
9120-9660 cm^-1.

Default comparison:
    This work: 273 K, 303 K weighted, 333 K
    Karman: closest available temperatures, 276 K, 306 K, 336 K

Use --karman-mode all to include all available Karman temperatures.
"""

from __future__ import annotations

import argparse
import re
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
DEFAULT_OUTPUT_DIR = Path("output/results/analysis/Integrated_strength_temperature_fit_with_Karman")

EXPERIMENT_SERIES = [
    {
        "temperature_k": 273.0,
        "b_col": "B_273K_500Torr",
        "u_col": "u_B_273K_500Torr",
        "label": "This work 273 K",
    },
    {
        "temperature_k": 303.0,
        "b_col": "B_303K_weighted",
        "u_col": "u_B_303K_weighted_scaled",
        "label": "This work 303 K",
    },
    {
        "temperature_k": 333.0,
        "b_col": "B_333K_500Torr",
        "u_col": "u_B_333K_500Torr",
        "label": "This work 333 K",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and plot integrated strength S(T) with Karman data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--karman", type=Path, default=DEFAULT_KARMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Lower integration limit. If omitted, use the common data range.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Upper integration limit. If omitted, use the common data range.",
    )
    parser.add_argument(
        "--karman-mode",
        choices=("nearest", "all"),
        default="nearest",
        help="Use Karman temperatures nearest to this work, or all available Karman temperatures.",
    )
    parser.add_argument("--figure-width-mm", type=float, default=90.0)
    parser.add_argument("--figure-height-mm", type=float, default=72.0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def find_karman_temperature_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    pattern = re.compile(r"^B_(\d+(?:\.\d+)?)K$")
    columns = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            columns.append((float(match.group(1)), col))
    if not columns:
        raise ValueError("No Karman columns like B_306K were found.")
    return sorted(columns)


def trapezoid_integral(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 2:
        raise ValueError("At least two valid points are required for integration.")
    return float(np.trapezoid(y, x))


def filter_range(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    work = df.copy()
    work["wavenumber"] = pd.to_numeric(work["wavenumber"], errors="coerce")
    mask = np.isfinite(work["wavenumber"]) & (work["wavenumber"] >= start) & (work["wavenumber"] <= end)
    return work.loc[mask].sort_values("wavenumber").reset_index(drop=True)


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    dof = max(len(x) - 2, 1)
    rmse = float(np.sqrt(np.sum(residual**2) / dof))
    return {
        "slope_cm_neg2_amagat_neg2_K_neg1": float(slope),
        "intercept_cm_neg2_amagat_neg2": float(intercept),
        "rmse_cm_neg2_amagat_neg2": rmse,
        "n_points": int(len(x)),
    }


def calculate_experiment_strengths(experiment: pd.DataFrame) -> pd.DataFrame:
    x = experiment["wavenumber"].to_numpy(dtype=float)
    rows = []
    for item in EXPERIMENT_SERIES:
        b = pd.to_numeric(experiment[item["b_col"]], errors="coerce").to_numpy(dtype=float)
        u_b = pd.to_numeric(experiment[item["u_col"]], errors="coerce").to_numpy(dtype=float)
        s = trapezoid_integral(x, b)
        s_upper = trapezoid_integral(x, b + u_b)
        s_lower = trapezoid_integral(x, b - u_b)
        u_s = (s_upper - s_lower) / 2.0
        rows.append(
            {
                "source": "This work",
                "temperature_k": item["temperature_k"],
                "temperature_label": item["label"],
                "S_cm_neg2_amagat_neg2": s,
                "u_S_1sigma_cm_neg2_amagat_neg2": u_s,
                "S_lower_1sigma_cm_neg2_amagat_neg2": s - u_s,
                "S_upper_1sigma_cm_neg2_amagat_neg2": s + u_s,
                "S_1e_neg6_cm_neg2_amagat_neg2": s / 1e-6,
                "u_S_1sigma_1e_neg6_cm_neg2_amagat_neg2": u_s / 1e-6,
            }
        )
    return pd.DataFrame(rows)


def select_karman_columns(
    columns: list[tuple[float, str]],
    mode: str,
    targets: list[float],
) -> list[tuple[float, str]]:
    if mode == "all":
        return columns
    selected: list[tuple[float, str]] = []
    used = set()
    for target in targets:
        temp, col = min(columns, key=lambda item: abs(item[0] - target))
        if col not in used:
            selected.append((temp, col))
            used.add(col)
    return selected


def calculate_karman_strengths(karman: pd.DataFrame, mode: str, targets: list[float]) -> pd.DataFrame:
    x = karman["wavenumber"].to_numpy(dtype=float)
    all_columns = find_karman_temperature_columns(karman)
    selected_columns = select_karman_columns(all_columns, mode=mode, targets=targets)
    rows = []
    for temp, col in selected_columns:
        b = pd.to_numeric(karman[col], errors="coerce").to_numpy(dtype=float)
        s = trapezoid_integral(x, b)
        rows.append(
            {
                "source": "Karman",
                "temperature_k": temp,
                "temperature_label": f"Karman {temp:g} K",
                "S_cm_neg2_amagat_neg2": s,
                "u_S_1sigma_cm_neg2_amagat_neg2": np.nan,
                "S_lower_1sigma_cm_neg2_amagat_neg2": np.nan,
                "S_upper_1sigma_cm_neg2_amagat_neg2": np.nan,
                "S_1e_neg6_cm_neg2_amagat_neg2": s / 1e-6,
                "u_S_1sigma_1e_neg6_cm_neg2_amagat_neg2": np.nan,
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


def plot_strengths(
    experiment_rows: pd.DataFrame,
    karman_rows: pd.DataFrame,
    fit_rows: pd.DataFrame,
    output_png: Path,
    width_mm: float,
    height_mm: float,
    dpi: int,
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.17, right=0.965, bottom=0.17, top=0.955)

    exp_color = "#000000"
    karman_color = "#D55E00"
    exp_x = experiment_rows["temperature_k"].to_numpy(dtype=float)
    exp_y = experiment_rows["S_1e_neg6_cm_neg2_amagat_neg2"].to_numpy(dtype=float)
    exp_u = experiment_rows["u_S_1sigma_1e_neg6_cm_neg2_amagat_neg2"].to_numpy(dtype=float)
    kar_x = karman_rows["temperature_k"].to_numpy(dtype=float)
    kar_y = karman_rows["S_1e_neg6_cm_neg2_amagat_neg2"].to_numpy(dtype=float)

    ax.errorbar(
        exp_x,
        exp_y,
        yerr=exp_u,
        fmt="s",
        ms=3.2,
        mfc=exp_color,
        mec=exp_color,
        ecolor=exp_color,
        elinewidth=0.65,
        capsize=2.0,
        capthick=0.65,
        linestyle="none",
        label="This work",
    )
    ax.plot(
        kar_x,
        kar_y,
        "o",
        ms=3.2,
        mfc=karman_color,
        mec=karman_color,
        linestyle="none",
        label="Karman",
    )

    x_min = min(float(np.nanmin(exp_x)), float(np.nanmin(kar_x))) - 8.0
    x_max = max(float(np.nanmax(exp_x)), float(np.nanmax(kar_x))) + 8.0
    x_line = np.linspace(x_min, x_max, 200)
    for _, row in fit_rows.iterrows():
        if row["source"] == "This work":
            color = exp_color
            label = "This work fit"
        else:
            color = karman_color
            label = "Karman fit"
        y_line = (
            row["intercept_cm_neg2_amagat_neg2"]
            + row["slope_cm_neg2_amagat_neg2_K_neg1"] * x_line
        ) / 1e-6
        ax.plot(x_line, y_line, color=color, lw=0.85, label=label)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$S_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-2}$ amagat$^{-2}$)")
    ax.legend(frameon=False, loc="best", handlelength=1.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.savefig(output_png, dpi=dpi)
    fig.savefig(output_png.with_suffix(".pdf"))
    fig.savefig(output_png.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)


def make_fit_curve_points(
    fit_rows: pd.DataFrame,
    x_min: float,
    x_max: float,
    n_points: int = 200,
) -> pd.DataFrame:
    temperature = np.linspace(x_min, x_max, n_points)
    records = []
    for _, row in fit_rows.iterrows():
        s_fit = (
            row["intercept_cm_neg2_amagat_neg2"]
            + row["slope_cm_neg2_amagat_neg2_K_neg1"] * temperature
        )
        for t, s in zip(temperature, s_fit):
            records.append(
                {
                    "source": row["source"],
                    "temperature_k": t,
                    "S_fit_cm_neg2_amagat_neg2": s,
                    "S_fit_1e_neg6_cm_neg2_amagat_neg2": s / 1e-6,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    experiment_path = args.experiment.expanduser().resolve()
    karman_path = args.karman.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_exp_cols = ["wavenumber"]
    for item in EXPERIMENT_SERIES:
        required_exp_cols.extend([item["b_col"], item["u_col"]])
    experiment = pd.read_csv(experiment_path, usecols=required_exp_cols)
    karman = pd.read_csv(karman_path)

    exp_min = float(pd.to_numeric(experiment["wavenumber"], errors="coerce").min())
    exp_max = float(pd.to_numeric(experiment["wavenumber"], errors="coerce").max())
    kar_min = float(pd.to_numeric(karman["wavenumber"], errors="coerce").min())
    kar_max = float(pd.to_numeric(karman["wavenumber"], errors="coerce").max())
    start = max(exp_min, kar_min) if args.start is None else args.start
    end = min(exp_max, kar_max) if args.end is None else args.end
    if start >= end:
        raise SystemExit(f"Invalid integration range: {start} to {end}")

    experiment = filter_range(experiment, start, end)
    karman = filter_range(karman, start, end)
    experiment_rows = calculate_experiment_strengths(experiment)
    karman_rows = calculate_karman_strengths(
        karman,
        mode=args.karman_mode,
        targets=experiment_rows["temperature_k"].tolist(),
    )

    fit_records = []
    for source, rows in [("This work", experiment_rows), ("Karman", karman_rows)]:
        fit = linear_fit(
            rows["temperature_k"].to_numpy(dtype=float),
            rows["S_cm_neg2_amagat_neg2"].to_numpy(dtype=float),
        )
        fit["source"] = source
        fit["integration_start_cm1"] = start
        fit["integration_end_cm1"] = end
        fit_records.append(fit)
    fit_rows = pd.DataFrame(fit_records)

    strength_rows = pd.concat([experiment_rows, karman_rows], ignore_index=True)
    strength_rows.insert(0, "integration_start_cm1", start)
    strength_rows.insert(1, "integration_end_cm1", end)

    strength_csv = output_dir / "integrated_strength_temperature_points.csv"
    fit_csv = output_dir / "integrated_strength_temperature_linear_fit.csv"
    fit_points_csv = output_dir / "integrated_strength_temperature_fit_curve_points.csv"
    output_png = output_dir / "integrated_strength_temperature_fit_with_karman.png"
    strength_rows.to_csv(strength_csv, index=False, float_format="%.15g")
    fit_rows.to_csv(fit_csv, index=False, float_format="%.15g")
    x_min = min(
        float(experiment_rows["temperature_k"].min()),
        float(karman_rows["temperature_k"].min()),
    ) - 8.0
    x_max = max(
        float(experiment_rows["temperature_k"].max()),
        float(karman_rows["temperature_k"].max()),
    ) + 8.0
    fit_curve_points = make_fit_curve_points(fit_rows, x_min, x_max)
    fit_curve_points.to_csv(fit_points_csv, index=False, float_format="%.15g")
    plot_strengths(
        experiment_rows,
        karman_rows,
        fit_rows,
        output_png,
        args.figure_width_mm,
        args.figure_height_mm,
        args.dpi,
    )

    print(f"Strength points: {strength_csv}")
    print(f"Fit summary: {fit_csv}")
    print(f"Fit curve points: {fit_points_csv}")
    print(f"Figure: {output_png}")
    print(strength_rows.to_string(index=False))
    print(fit_rows.to_string(index=False))


if __name__ == "__main__":
    main()
