#!/usr/bin/env python3
"""Calculate gas-density rho uncertainty for final CIA data.

The final CIA coefficient is calculated as

    B = alpha / rho^2

where rho is the O2 number density in amagat.  For the ideal-gas density
relative to Loschmidt density,

    rho = (p / 760) * (273.15 / T)

where p is in Torr and T is in K.  Assuming independent pressure and
temperature standard uncertainties, first-order propagation gives

    u_rho / rho = sqrt((u_p / p)^2 + (u_T / T)^2)

This script reads the final CIA Calculation tables, derives the rho value
actually used in those final tables from `rho = sqrt(alpha / B)`, and applies
the pressure/temperature relative uncertainty to that rho.  This keeps the
rho value consistent with the final processed data even if a later pressure
summary table differs slightly.

By default, the CTR100 pressure specification is treated as a rectangular
half-width of 0.20% of reading and converted to a standard uncertainty by
dividing by sqrt(3).  The temperature standard uncertainty defaults to
0.001 K.  If your instrument specification should be interpreted differently,
override the command-line options.

Usage:
    python scripts/calculate_final_rho_uncertainty.py \
        "/Users/donghang/科研/实验数据/氧气连续吸收温度/最终处理数据" \
        --pressure-relative-half-width-percent 0.2 \
        --temperature-uncertainty-k 0.001
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_TEMPERATURE_K = 273.15
STANDARD_PRESSURE_TORR = 760.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate rho and rho uncertainty for final CIA tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing final CIA Calculation *.txt files.",
    )
    parser.add_argument(
        "--pressure-temperature-summary",
        type=Path,
        default=Path("output/results/CIA_pressure_temperature_summary.csv"),
        help="Pressure/temperature summary CSV after outlier rejection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/uncertainty/CIA/final_rho_uncertainty"),
        help="Output directory.",
    )
    parser.add_argument("--encoding", default="gbk", help="Input text encoding.")
    parser.add_argument("--pattern", default="CIA Calculation *.txt", help="Input filename glob pattern.")
    parser.add_argument(
        "--pressure-uncertainty-torr",
        type=float,
        default=None,
        help="Pressure standard uncertainty in Torr.",
    )
    parser.add_argument(
        "--pressure-relative-uncertainty-percent",
        type=float,
        default=None,
        help="Pressure standard uncertainty as percent of reading.",
    )
    parser.add_argument(
        "--pressure-relative-half-width-percent",
        type=float,
        default=0.2,
        help="Pressure rectangular half-width as percent of reading.",
    )
    parser.add_argument(
        "--temperature-uncertainty-k",
        type=float,
        default=0.001,
        help="Temperature standard uncertainty in K.",
    )
    parser.add_argument(
        "--pressure-half-width-torr",
        type=float,
        help="Pressure rectangular half-width in Torr. Overrides --pressure-uncertainty-torr.",
    )
    parser.add_argument(
        "--temperature-half-width-k",
        type=float,
        help="Temperature rectangular half-width in K. Overrides --temperature-uncertainty-k.",
    )
    return parser.parse_args()


def temperature_standard_uncertainty(args: argparse.Namespace) -> tuple[float, str]:
    temperature_u = args.temperature_uncertainty_k
    mode = "temperature_standard_uncertainty"
    if args.temperature_half_width_k is not None:
        temperature_u = args.temperature_half_width_k / math.sqrt(3.0)
        mode = "temperature_rectangular_half_width_converted_to_standard"
    if temperature_u < 0:
        raise SystemExit("Temperature uncertainty must be non-negative.")
    return float(temperature_u), mode


def pressure_standard_uncertainty(args: argparse.Namespace, pressure_torr: float) -> tuple[float, str]:
    if args.pressure_half_width_torr is not None:
        pressure_u = args.pressure_half_width_torr / math.sqrt(3.0)
        mode = "pressure_rectangular_half_width_torr_converted_to_standard"
    elif args.pressure_uncertainty_torr is not None:
        pressure_u = args.pressure_uncertainty_torr
        mode = "pressure_standard_uncertainty_torr"
    elif args.pressure_relative_uncertainty_percent is not None:
        pressure_u = pressure_torr * args.pressure_relative_uncertainty_percent / 100.0
        mode = "pressure_relative_standard_uncertainty_percent"
    else:
        pressure_u = pressure_torr * args.pressure_relative_half_width_percent / 100.0 / math.sqrt(3.0)
        mode = "pressure_relative_rectangular_half_width_percent_converted_to_standard"
    if pressure_u < 0:
        raise SystemExit("Pressure uncertainty must be non-negative.")
    return float(pressure_u), mode


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
    temperature_group, pressure_group = match.groups()
    safe_name = f"{temperature_group}_{pressure_group}"
    label = f"{temperature_group}/{pressure_group}"
    return temperature_group, pressure_group, safe_name, label


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
    df = df.dropna(subset=["wavenumber", "alpha_cm_inv", "binary_coeff_cm_inv_amagat_neg2"])
    return df.sort_values("wavenumber").reset_index(drop=True)


def load_pressure_temperature_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"temperature_group", "gas_pressure", "pressure_torr", "temperature_k"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return df


def find_o2_pressure_temperature(summary: pd.DataFrame, temperature_group: str, pressure_group: str) -> pd.Series:
    gas_pressure = f"O2 {pressure_group}"
    mask = (summary["temperature_group"] == temperature_group) & (summary["gas_pressure"] == gas_pressure)
    matched = summary.loc[mask]
    if matched.empty:
        raise SystemExit(
            "Cannot find pressure/temperature summary row for "
            f"{temperature_group}/{gas_pressure}"
        )
    return matched.iloc[0]


def rho_from_pressure_temperature(pressure_torr: float, temperature_k: float) -> float:
    return (pressure_torr / STANDARD_PRESSURE_TORR) * (STANDARD_TEMPERATURE_K / temperature_k)


def process_group(
    path: Path,
    pt_summary: pd.DataFrame,
    args: argparse.Namespace,
    temperature_u_k: float,
    temperature_uncertainty_mode: str,
    encoding: str,
) -> tuple[pd.DataFrame, dict, str]:
    temperature_group, pressure_group, safe_name, label = parse_group(path)
    df = read_final_table(path, encoding)
    pt_row = find_o2_pressure_temperature(pt_summary, temperature_group, pressure_group)

    pressure_torr_summary = float(pt_row["pressure_torr"])
    temperature_k_summary = float(pt_row["temperature_k"])
    rho_pt = rho_from_pressure_temperature(pressure_torr_summary, temperature_k_summary)

    alpha = df["alpha_cm_inv"].to_numpy(dtype=float)
    binary_coeff = df["binary_coeff_cm_inv_amagat_neg2"].to_numpy(dtype=float)
    valid = np.isfinite(alpha) & np.isfinite(binary_coeff) & (alpha > 0) & (binary_coeff > 0)
    rho_from_final = np.full(len(df), np.nan, dtype=float)
    rho_from_final[valid] = np.sqrt(alpha[valid] / binary_coeff[valid])

    rho_reference = float(np.nanmedian(rho_from_final))
    if not np.isfinite(rho_reference):
        rho_reference = rho_pt
        rho_source = "pressure_temperature_summary"
    else:
        rho_source = "sqrt_alpha_over_binary_coeff"

    pressure_torr_effective = rho_reference * STANDARD_PRESSURE_TORR * temperature_k_summary / STANDARD_TEMPERATURE_K
    pressure_u_torr, pressure_uncertainty_mode = pressure_standard_uncertainty(
        args,
        pressure_torr_effective,
    )
    relative_u = math.sqrt(
        (pressure_u_torr / pressure_torr_effective) ** 2
        + (temperature_u_k / temperature_k_summary) ** 2
    )

    out = pd.DataFrame({
        "group": label,
        "temperature_group": temperature_group,
        "pressure_group": pressure_group,
        "wavenumber": df["wavenumber"].to_numpy(dtype=float),
        "rho_amagat": rho_from_final,
        "u_rho_amagat": rho_from_final * relative_u,
        "u_rho_rel": relative_u,
        "u_rho_rel_percent": relative_u * 100.0,
        "rho_from_pressure_temperature_amagat": rho_pt,
        "rho_source": rho_source,
        "pressure_torr_summary": pressure_torr_summary,
        "temperature_k_summary": temperature_k_summary,
        "pressure_torr_effective_from_final_rho": pressure_torr_effective,
        "pressure_uncertainty_torr": pressure_u_torr,
        "temperature_uncertainty_k": temperature_u_k,
        "pressure_uncertainty_mode": pressure_uncertainty_mode,
        "temperature_uncertainty_mode": temperature_uncertainty_mode,
        "n_pressure_temperature_used": int(pt_row.get("used_points", -1)),
        "n_pressure_temperature_removed": int(pt_row.get("removed_points", -1)),
    })
    rho_diff_percent = (rho_reference - rho_pt) / rho_pt * 100.0 if rho_pt else np.nan
    summary = {
        "group": label,
        "temperature_group": temperature_group,
        "pressure_group": pressure_group,
        "n_points": int(len(out)),
        "rho_amagat": rho_reference,
        "u_rho_amagat": float(rho_reference * relative_u),
        "u_rho_rel": relative_u,
        "u_rho_rel_percent": relative_u * 100.0,
        "rho_from_pressure_temperature_amagat": rho_pt,
        "rho_relative_difference_from_pressure_temperature_percent": rho_diff_percent,
        "pressure_torr_summary": pressure_torr_summary,
        "temperature_k_summary": temperature_k_summary,
        "pressure_torr_effective_from_final_rho": pressure_torr_effective,
        "pressure_uncertainty_torr": pressure_u_torr,
        "temperature_uncertainty_k": temperature_u_k,
        "pressure_uncertainty_mode": pressure_uncertainty_mode,
        "temperature_uncertainty_mode": temperature_uncertainty_mode,
        "n_pressure_temperature_used": int(pt_row.get("used_points", -1)),
        "n_pressure_temperature_removed": int(pt_row.get("removed_points", -1)),
        "rho_source": rho_source,
        "source_file": str(path),
    }
    return out, summary, safe_name


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    pt_summary_path = args.pressure_temperature_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature_u_k, temperature_uncertainty_mode = temperature_standard_uncertainty(args)
    pt_summary = load_pressure_temperature_summary(pt_summary_path)
    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No input files found: {input_dir / args.pattern}")

    summaries = []
    all_rows = []
    wide = None
    for path in paths:
        out, summary, safe_name = process_group(
            path,
            pt_summary,
            args,
            temperature_u_k,
            temperature_uncertainty_mode,
            args.encoding,
        )
        output_csv = output_dir / f"{safe_name}_rho_uncertainty.csv"
        out.to_csv(output_csv, index=False, float_format="%.15g")
        summary["output_csv"] = str(output_csv)
        summaries.append(summary)
        all_rows.append(out)
        if wide is None:
            wide = pd.DataFrame({"wavenumber": out["wavenumber"].to_numpy(dtype=float)})
        wide[f"{safe_name}_rho_amagat"] = out["rho_amagat"].to_numpy(dtype=float)
        wide[f"{safe_name}_u_rho_amagat"] = out["u_rho_amagat"].to_numpy(dtype=float)
        print(f"Wrote: {output_csv}")

    summary_df = pd.DataFrame(summaries)
    all_df = pd.concat(all_rows, ignore_index=True)
    summary_path = output_dir / "rho_uncertainty_summary.csv"
    all_path = output_dir / "rho_uncertainty_all.csv"
    wide_path = output_dir / "rho_uncertainty_wide.csv"
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
