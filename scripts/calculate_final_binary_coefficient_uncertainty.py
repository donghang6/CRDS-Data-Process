#!/usr/bin/env python3
"""Calculate final binary CIA coefficient uncertainty.

For O2-O2 collision-induced absorption,

    B = alpha / rho^2

where alpha is the absorption coefficient in cm^-1 and rho is the O2 number
density in amagat.  With alpha and rho treated as independent quantities, the
first-order propagated standard uncertainty is

    u_B = sqrt(
        (u_alpha / rho^2)^2
        + (2 * B * u_rho / rho)^2
    )

or equivalently,

    u_B / B = sqrt((u_alpha / alpha)^2 + (2 * u_rho / rho)^2)

Usage:
    python scripts/calculate_final_binary_coefficient_uncertainty.py

Inputs by default:
    output/results/uncertainty/CIA/final_absorption_uncertainty/
    output/results/uncertainty/CIA/final_rho_uncertainty/

Outputs by default:
    output/results/uncertainty/CIA/final_binary_coefficient_uncertainty/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate final binary CIA coefficient uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--absorption-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_absorption_uncertainty"),
        help="Directory containing *_absorption_uncertainty.csv files.",
    )
    parser.add_argument(
        "--rho-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_rho_uncertainty"),
        help="Directory containing *_rho_uncertainty.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_binary_coefficient_uncertainty"),
        help="Output directory.",
    )
    parser.add_argument(
        "--absorption-pattern",
        default="*_absorption_uncertainty.csv",
        help="Glob pattern for absorption uncertainty files.",
    )
    parser.add_argument(
        "--rho-pattern-suffix",
        default="_rho_uncertainty.csv",
        help="Suffix used to find the matching rho uncertainty file.",
    )
    return parser.parse_args()


def group_key_from_absorption_path(path: Path) -> str:
    suffix = "_absorption_uncertainty"
    if not path.stem.endswith(suffix):
        raise ValueError(f"Unexpected absorption filename: {path.name}")
    return path.stem[: -len(suffix)]


def matching_rho_path(absorption_path: Path, rho_dir: Path, rho_suffix: str) -> Path:
    key = group_key_from_absorption_path(absorption_path)
    return rho_dir / f"{key}{rho_suffix}"


def require_columns(df: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")


def process_pair(absorption_path: Path, rho_path: Path) -> tuple[pd.DataFrame, dict, str]:
    alpha_df = pd.read_csv(absorption_path)
    rho_df = pd.read_csv(rho_path)
    require_columns(
        alpha_df,
        absorption_path,
        (
            "group",
            "temperature_group",
            "pressure_group",
            "wavenumber",
            "alpha_cm_inv",
            "u_alpha_cm_inv",
            "binary_coeff_cm_inv_amagat_neg2",
        ),
    )
    require_columns(
        rho_df,
        rho_path,
        (
            "wavenumber",
            "rho_amagat",
            "u_rho_amagat",
            "u_rho_rel",
            "pressure_torr_summary",
            "temperature_k_summary",
        ),
    )
    if len(alpha_df) != len(rho_df) or not np.allclose(
        alpha_df["wavenumber"].to_numpy(dtype=float),
        rho_df["wavenumber"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    ):
        raise SystemExit(f"Wavenumber grids do not match: {absorption_path} vs {rho_path}")

    alpha = pd.to_numeric(alpha_df["alpha_cm_inv"], errors="coerce").to_numpy(dtype=float)
    u_alpha = pd.to_numeric(alpha_df["u_alpha_cm_inv"], errors="coerce").to_numpy(dtype=float)
    rho = pd.to_numeric(rho_df["rho_amagat"], errors="coerce").to_numpy(dtype=float)
    u_rho = pd.to_numeric(rho_df["u_rho_amagat"], errors="coerce").to_numpy(dtype=float)
    binary_existing = pd.to_numeric(
        alpha_df["binary_coeff_cm_inv_amagat_neg2"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(alpha)
        & np.isfinite(u_alpha)
        & np.isfinite(rho)
        & np.isfinite(u_rho)
        & (rho > 0)
    )
    binary_calc = np.full(len(alpha_df), np.nan, dtype=float)
    u_b_alpha = np.full(len(alpha_df), np.nan, dtype=float)
    u_b_rho = np.full(len(alpha_df), np.nan, dtype=float)
    u_b = np.full(len(alpha_df), np.nan, dtype=float)

    binary_calc[valid] = alpha[valid] / rho[valid] ** 2
    u_b_alpha[valid] = np.abs(u_alpha[valid] / rho[valid] ** 2)
    u_b_rho[valid] = np.abs(2.0 * binary_calc[valid] * u_rho[valid] / rho[valid])
    u_b[valid] = np.sqrt(u_b_alpha[valid] ** 2 + u_b_rho[valid] ** 2)

    u_b_rel = np.full(len(alpha_df), np.nan, dtype=float)
    valid_b = valid & np.isfinite(binary_calc) & (binary_calc != 0)
    u_b_rel[valid_b] = u_b[valid_b] / np.abs(binary_calc[valid_b])

    out = pd.DataFrame({
        "group": alpha_df["group"],
        "temperature_group": alpha_df["temperature_group"],
        "pressure_group": alpha_df["pressure_group"],
        "wavenumber": alpha_df["wavenumber"],
        "alpha_cm_inv": alpha,
        "u_alpha_cm_inv": u_alpha,
        "rho_amagat": rho,
        "u_rho_amagat": u_rho,
        "u_rho_rel": rho_df["u_rho_rel"].to_numpy(dtype=float),
        "binary_coeff_cm_inv_amagat_neg2": binary_existing,
        "binary_coeff_recomputed_cm_inv_amagat_neg2": binary_calc,
        "binary_coeff_recompute_diff": binary_calc - binary_existing,
        "u_binary_coeff_from_alpha": u_b_alpha,
        "u_binary_coeff_from_rho": u_b_rho,
        "u_binary_coeff": u_b,
        "u_binary_coeff_rel": u_b_rel,
        "u_binary_coeff_rel_percent": u_b_rel * 100.0,
        "pressure_torr_summary": rho_df["pressure_torr_summary"],
        "temperature_k_summary": rho_df["temperature_k_summary"],
        "pressure_uncertainty_torr": rho_df.get("pressure_uncertainty_torr", np.nan),
        "temperature_uncertainty_k": rho_df.get("temperature_uncertainty_k", np.nan),
    })
    key = group_key_from_absorption_path(absorption_path)
    summary = {
        "group": str(out["group"].iloc[0]),
        "n_points": int(len(out)),
        "wavenumber_min": float(pd.to_numeric(out["wavenumber"], errors="coerce").min()),
        "wavenumber_max": float(pd.to_numeric(out["wavenumber"], errors="coerce").max()),
        "mean_u_binary_coeff": float(out["u_binary_coeff"].mean()),
        "median_u_binary_coeff": float(out["u_binary_coeff"].median()),
        "max_u_binary_coeff": float(out["u_binary_coeff"].max()),
        "mean_u_binary_coeff_rel_percent": float(out["u_binary_coeff_rel_percent"].mean()),
        "median_u_binary_coeff_rel_percent": float(out["u_binary_coeff_rel_percent"].median()),
        "median_u_binary_coeff_from_alpha": float(out["u_binary_coeff_from_alpha"].median()),
        "median_u_binary_coeff_from_rho": float(out["u_binary_coeff_from_rho"].median()),
        "max_abs_binary_recompute_diff": float(out["binary_coeff_recompute_diff"].abs().max()),
        "absorption_source": str(absorption_path),
        "rho_source": str(rho_path),
    }
    return out, summary, key


def main() -> None:
    args = parse_args()
    absorption_dir = args.absorption_dir.expanduser().resolve()
    rho_dir = args.rho_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    absorption_paths = sorted(
        path
        for path in absorption_dir.glob(args.absorption_pattern)
        if path.name != "absorption_uncertainty_all.csv"
    )
    if not absorption_paths:
        raise SystemExit(f"No absorption uncertainty files found under {absorption_dir}")

    summaries = []
    all_rows = []
    wide = None
    for absorption_path in absorption_paths:
        rho_path = matching_rho_path(absorption_path, rho_dir, args.rho_pattern_suffix)
        if not rho_path.exists():
            raise SystemExit(f"Missing matching rho file for {absorption_path}: {rho_path}")
        out, summary, key = process_pair(absorption_path, rho_path)
        output_csv = output_dir / f"{key}_binary_coefficient_uncertainty.csv"
        out.to_csv(output_csv, index=False, float_format="%.15g")
        summary["output_csv"] = str(output_csv)
        summaries.append(summary)
        all_rows.append(out)
        if wide is None:
            wide = pd.DataFrame({"wavenumber": pd.to_numeric(out["wavenumber"], errors="coerce")})
        wide[f"{key}_B"] = out["binary_coeff_recomputed_cm_inv_amagat_neg2"].to_numpy(dtype=float)
        wide[f"{key}_u_B"] = out["u_binary_coeff"].to_numpy(dtype=float)
        wide[f"{key}_u_B_rel_percent"] = out["u_binary_coeff_rel_percent"].to_numpy(dtype=float)
        print(f"Wrote: {output_csv}")

    summary_df = pd.DataFrame(summaries)
    all_df = pd.concat(all_rows, ignore_index=True)
    summary_path = output_dir / "binary_coefficient_uncertainty_summary.csv"
    all_path = output_dir / "binary_coefficient_uncertainty_all.csv"
    wide_path = output_dir / "binary_coefficient_uncertainty_wide.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.15g")
    all_df.to_csv(all_path, index=False, float_format="%.15g")
    if wide is not None:
        wide.to_csv(wide_path, index=False, float_format="%.15g")

    print(f"Summary: {summary_path}")
    print(f"All groups: {all_path}")
    print(f"Wide table: {wide_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
