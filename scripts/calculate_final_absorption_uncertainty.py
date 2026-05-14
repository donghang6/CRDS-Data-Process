#!/usr/bin/env python3
"""Calculate absorption-coefficient uncertainty from final CIA data tables.

The input files are the final 0.01 cm^-1 grid tables with the following
numeric columns after three header rows:

    wavenumber
    Ar tau (us)
    Ar tau uncertainty (us)
    O2 tau (us)
    O2 tau uncertainty (us)
    absorption coefficient alpha (cm^-1)
    binary CIA coefficient (cm^-1 amagat^-2)

For CRDS,

    alpha = (1 / c) * (1 / tau_O2 - 1 / tau_Ar)

where c is in cm/s and tau is in seconds.  Using tau in microseconds:

    alpha = (1e6 / c) * (1 / tau_O2_us - 1 / tau_Ar_us)

Assuming independent Ar and O2 tau uncertainties, first-order uncertainty
propagation gives:

    u_alpha = (1e6 / c) * sqrt(
        (u_tau_O2 / tau_O2_us^2)^2
        + (u_tau_Ar / tau_Ar_us^2)^2
    )

Usage:
    python scripts/calculate_final_absorption_uncertainty.py \
        "/path/to/final/data"

Common command for the current data set:
    python scripts/calculate_final_absorption_uncertainty.py \
        "/Users/donghang/科研/实验数据/氧气连续吸收温度/最终处理数据"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

C_CM_PER_S = 2.99792458e10
TAU_US_TO_ALPHA_CM_INV = 1e6 / C_CM_PER_S


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate absorption-coefficient uncertainty from final CIA tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing final CIA Calculation *.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_absorption_uncertainty"),
        help="Output directory.",
    )
    parser.add_argument(
        "--encoding",
        default="gbk",
        help="Input text encoding.",
    )
    parser.add_argument(
        "--pattern",
        default="CIA Calculation *.txt",
        help="Input filename glob pattern.",
    )
    return parser.parse_args()


def detect_separator(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace") as handle:
        first_line = handle.readline()
    if first_line.count(",") > first_line.count("\t"):
        return ","
    return "\t"


def parse_group(path: Path) -> tuple[str, str, str, str]:
    match = re.search(r"(\d+K)\s+(\d+Torr)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse temperature/pressure from filename: {path.name}")
    temperature, pressure = match.groups()
    safe_name = f"{temperature}_{pressure}"
    label = f"{temperature}/{pressure}"
    return temperature, pressure, safe_name, label


def read_final_table(path: Path, encoding: str) -> pd.DataFrame:
    sep = detect_separator(path, encoding)
    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        header=None,
        skiprows=3,
        usecols=list(range(7)),
        names=[
            "wavenumber",
            "tau_ar_us",
            "u_tau_ar_us",
            "tau_o2_us",
            "u_tau_o2_us",
            "alpha_cm_inv",
            "binary_coeff_cm_inv_amagat_neg2",
        ],
    )
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["wavenumber", "tau_ar_us", "u_tau_ar_us", "tau_o2_us", "u_tau_o2_us"])
    df = df.sort_values("wavenumber").reset_index(drop=True)
    return df


def calculate_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    tau_ar = df["tau_ar_us"].to_numpy(dtype=float)
    tau_o2 = df["tau_o2_us"].to_numpy(dtype=float)
    u_tau_ar = np.abs(df["u_tau_ar_us"].to_numpy(dtype=float))
    u_tau_o2 = np.abs(df["u_tau_o2_us"].to_numpy(dtype=float))

    valid = (tau_ar > 0) & (tau_o2 > 0) & np.isfinite(tau_ar) & np.isfinite(tau_o2)
    alpha_recomputed = np.full(len(df), np.nan, dtype=float)
    u_alpha = np.full(len(df), np.nan, dtype=float)

    alpha_recomputed[valid] = TAU_US_TO_ALPHA_CM_INV * (1.0 / tau_o2[valid] - 1.0 / tau_ar[valid])
    u_alpha[valid] = TAU_US_TO_ALPHA_CM_INV * np.sqrt(
        (u_tau_o2[valid] / tau_o2[valid] ** 2) ** 2
        + (u_tau_ar[valid] / tau_ar[valid] ** 2) ** 2
    )

    out = df.copy()
    out["alpha_recomputed_cm_inv"] = alpha_recomputed
    out["alpha_recompute_diff_cm_inv"] = out["alpha_recomputed_cm_inv"] - out["alpha_cm_inv"]
    out["u_alpha_cm_inv"] = u_alpha
    out["u_alpha_rel"] = out["u_alpha_cm_inv"] / out["alpha_cm_inv"].abs()
    out["u_alpha_rel_percent"] = out["u_alpha_rel"] * 100.0

    ratio = out["binary_coeff_cm_inv_amagat_neg2"] / out["alpha_cm_inv"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    out["u_binary_coeff_tau_only"] = out["u_alpha_cm_inv"] * ratio.abs()
    return out


def summarize(group: str, path: Path, out: pd.DataFrame, output_csv: Path) -> dict:
    return {
        "group": group,
        "n_points": int(len(out)),
        "wavenumber_min": float(out["wavenumber"].min()),
        "wavenumber_max": float(out["wavenumber"].max()),
        "mean_u_alpha_cm_inv": float(out["u_alpha_cm_inv"].mean()),
        "median_u_alpha_cm_inv": float(out["u_alpha_cm_inv"].median()),
        "max_u_alpha_cm_inv": float(out["u_alpha_cm_inv"].max()),
        "mean_u_alpha_rel_percent": float(out["u_alpha_rel_percent"].mean()),
        "median_u_alpha_rel_percent": float(out["u_alpha_rel_percent"].median()),
        "max_abs_alpha_recompute_diff_cm_inv": float(out["alpha_recompute_diff_cm_inv"].abs().max()),
        "source_file": str(path),
        "output_csv": str(output_csv),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No input files found: {input_dir / args.pattern}")

    all_rows = []
    summaries = []
    wide = None
    for path in paths:
        temperature, pressure, safe_name, label = parse_group(path)
        df = read_final_table(path, args.encoding)
        out = calculate_uncertainty(df)
        out.insert(0, "group", label)
        out.insert(1, "temperature_group", temperature)
        out.insert(2, "pressure_group", pressure)

        output_csv = output_dir / f"{safe_name}_absorption_uncertainty.csv"
        out.to_csv(output_csv, index=False, float_format="%.15g")
        all_rows.append(out)
        summaries.append(summarize(label, path, out, output_csv))

        if wide is None:
            wide = pd.DataFrame({"wavenumber": out["wavenumber"].to_numpy(dtype=float)})
        wide[f"{safe_name}_u_alpha_cm_inv"] = out["u_alpha_cm_inv"].to_numpy(dtype=float)
        print(f"Wrote: {output_csv}")

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    all_path = output_dir / "absorption_uncertainty_all.csv"
    summary_path = output_dir / "summary.csv"
    wide_path = output_dir / "absorption_uncertainty_wide_cm_inv.csv"

    all_df.to_csv(all_path, index=False, float_format="%.15g")
    summary_df.to_csv(summary_path, index=False, float_format="%.15g")
    if wide is not None:
        wide.to_csv(wide_path, index=False, float_format="%.15g")

    print(f"All groups: {all_path}")
    print(f"Wide table: {wide_path}")
    print(f"Summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
