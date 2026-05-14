#!/usr/bin/env python3
"""Calculate integrated CIA band strength and uncertainty.

The input is the final binary coefficient uncertainty table:

    B(nu) = alpha(nu) / rho^2

The integrated strength over the measured wavenumber range is calculated by
the trapezoidal rule:

    S = integral B(nu) dnu ~= sum_i w_i B_i

with w_i the trapezoidal integration weights.  Since B has units
cm^-1 amagat^-2 and dnu has units cm^-1, S has units cm^-2 amagat^-2.

Uncertainty propagation:

The primary uncertainty is calculated by integrating the upper and lower
boundaries:

    B_upper(nu) = B(nu) + u_B(nu)
    B_lower(nu) = B(nu) - u_B(nu)

    S_upper = integral B_upper(nu) dnu
    S_lower = integral B_lower(nu) dnu
    u_S = (S_upper - S_lower) / 2

For comparison only, the script also keeps diagnostic columns for the alpha
point-to-point quadrature and rho common-scale propagation used previously.

Usage:
    python scripts/calculate_final_integrated_strength_uncertainty.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate integrated CIA strength and uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_binary_coefficient_uncertainty"),
        help="Directory containing *_binary_coefficient_uncertainty.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_integrated_strength"),
        help="Output directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*_binary_coefficient_uncertainty.csv",
        help="Input filename glob pattern.",
    )
    parser.add_argument(
        "--start",
        type=float,
        help="Optional lower integration limit in cm^-1.",
    )
    parser.add_argument(
        "--end",
        type=float,
        help="Optional upper integration limit in cm^-1.",
    )
    return parser.parse_args()


def group_key(path: Path) -> str:
    suffix = "_binary_coefficient_uncertainty"
    if not path.stem.endswith(suffix):
        raise ValueError(f"Unexpected filename: {path.name}")
    return path.stem[: -len(suffix)]


def require_columns(df: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        raise ValueError("At least two points are required for integration.")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("Wavenumber grid must be strictly increasing.")
    weights = np.empty_like(x, dtype=float)
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    if len(x) > 2:
        weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights


def process_file(path: Path, start: float | None, end: float | None) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)
    require_columns(
        df,
        path,
        (
            "group",
            "temperature_group",
            "pressure_group",
            "wavenumber",
            "binary_coeff_recomputed_cm_inv_amagat_neg2",
            "u_binary_coeff_from_alpha",
            "u_binary_coeff_from_rho",
            "u_binary_coeff",
            "rho_amagat",
            "u_rho_rel",
        ),
    )
    work = df.copy()
    for column in (
        "wavenumber",
        "binary_coeff_recomputed_cm_inv_amagat_neg2",
        "u_binary_coeff_from_alpha",
        "u_binary_coeff_from_rho",
        "u_binary_coeff",
        "rho_amagat",
        "u_rho_rel",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    mask = np.isfinite(work["wavenumber"])
    if start is not None:
        mask &= work["wavenumber"] >= start
    if end is not None:
        mask &= work["wavenumber"] <= end
    work = work.loc[mask].sort_values("wavenumber").reset_index(drop=True)
    work = work.dropna(
        subset=[
            "wavenumber",
            "binary_coeff_recomputed_cm_inv_amagat_neg2",
            "u_binary_coeff_from_alpha",
            "u_binary_coeff_from_rho",
            "u_binary_coeff",
        ],
    ).reset_index(drop=True)
    if len(work) < 2:
        raise SystemExit(f"Not enough valid points for integration: {path}")

    x = work["wavenumber"].to_numpy(dtype=float)
    weights = trapezoid_weights(x)
    b = work["binary_coeff_recomputed_cm_inv_amagat_neg2"].to_numpy(dtype=float)
    u_b_alpha = np.abs(work["u_binary_coeff_from_alpha"].to_numpy(dtype=float))
    u_b_rho = np.abs(work["u_binary_coeff_from_rho"].to_numpy(dtype=float))
    u_b = np.abs(work["u_binary_coeff"].to_numpy(dtype=float))

    integrated_strength = float(np.sum(weights * b))
    integrated_upper = float(np.sum(weights * (b + u_b)))
    integrated_lower = float(np.sum(weights * (b - u_b)))
    u_integrated_boundary = float((integrated_upper - integrated_lower) / 2.0)

    u_integrated_alpha_quadrature = float(np.sqrt(np.sum((weights * u_b_alpha) ** 2)))
    u_integrated_alpha_boundary = float(np.sum(weights * u_b_alpha))
    valid_b = np.isfinite(b) & (b != 0) & np.isfinite(u_b_rho)
    if np.any(valid_b):
        rho_relative_factor = float(np.nanmedian(np.abs(u_b_rho[valid_b] / b[valid_b])))
    else:
        rho_relative_factor = float(2.0 * np.nanmedian(np.abs(work["u_rho_rel"].to_numpy(dtype=float))))
    u_integrated_rho = float(abs(integrated_strength) * rho_relative_factor)

    # Diagnostic only: this treats every total u_B point as independent, which
    # is not appropriate for the common rho scale term but is useful for checks.
    u_integrated_all_points_independent = float(np.sqrt(np.sum((weights * u_b) ** 2)))

    rel = u_integrated_boundary / abs(integrated_strength) if integrated_strength != 0 else np.nan
    out_points = pd.DataFrame({
        "wavenumber": x,
        "integration_weight_cm_inv": weights,
        "weighted_B": weights * b,
        "weighted_B_upper": weights * (b + u_b),
        "weighted_B_lower": weights * (b - u_b),
        "weighted_u_B": weights * u_b,
        "weighted_u_B_alpha": weights * u_b_alpha,
        "weighted_u_B_rho_pointwise": weights * u_b_rho,
        "weighted_u_B_total_pointwise": weights * u_b,
    })
    summary = {
        "group": str(work["group"].iloc[0]),
        "temperature_group": str(work["temperature_group"].iloc[0]),
        "pressure_group": str(work["pressure_group"].iloc[0]),
        "n_points": int(len(work)),
        "wavenumber_min": float(x[0]),
        "wavenumber_max": float(x[-1]),
        "integration_method": "trapezoid",
        "uncertainty_method": "integrated_upper_lower_half_width",
        "integrated_strength": integrated_strength,
        "integrated_strength_upper": integrated_upper,
        "integrated_strength_lower": integrated_lower,
        "u_integrated_strength": u_integrated_boundary,
        "u_integrated_strength_rel": rel,
        "u_integrated_strength_rel_percent": rel * 100.0,
        "u_integrated_strength_from_alpha_boundary": u_integrated_alpha_boundary,
        "u_integrated_strength_from_alpha_quadrature_diagnostic": u_integrated_alpha_quadrature,
        "u_integrated_strength_from_rho_common_scale_diagnostic": u_integrated_rho,
        "rho_common_relative_factor": rho_relative_factor,
        "u_integrated_strength_all_points_independent_diagnostic": u_integrated_all_points_independent,
        "input_csv": str(path),
    }
    return summary, out_points


def main() -> None:
    args = parse_args()
    binary_dir = args.binary_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(
        path
        for path in binary_dir.glob(args.pattern)
        if path.name != "binary_coefficient_uncertainty_all.csv"
    )
    if not input_paths:
        raise SystemExit(f"No binary coefficient uncertainty files found under {binary_dir}")

    summaries = []
    for path in input_paths:
        summary, out_points = process_file(path, args.start, args.end)
        key = group_key(path)
        weights_path = output_dir / f"{key}_integration_weights.csv"
        out_points.to_csv(weights_path, index=False, float_format="%.15g")
        summary["integration_weights_csv"] = str(weights_path)
        summaries.append(summary)
        print(f"Wrote: {weights_path}")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "integrated_strength_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.15g")
    print(f"Summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
