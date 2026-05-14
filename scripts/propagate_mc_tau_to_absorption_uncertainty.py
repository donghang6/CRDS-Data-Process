#!/usr/bin/env python3
"""Propagate Monte Carlo tau-fit uncertainty to absorption/loss uncertainty.

使用说明
--------
本脚本读取每组 `continuum_step2_fit_mc_uncertainty.csv`，把 Monte Carlo 得到的
拟合衰荡时间不确定度传播到吸收系数/腔损耗不确定度。

使用的关系式为：

    loss_ppm_per_cm = K / tau_us
    K = 1e12 / c

其中 c = 2.99792458e10 cm/s，tau_us 的单位为微秒，loss 的单位为 ppm/cm。
一阶误差传播为：

    u_loss_ppm_per_cm = K * u_tau_us / tau_us^2

若没有额外参考谱不确定度，吸收系数 alpha = loss - reference_loss 的不确定度
可先取：

    u_alpha_ppm_per_cm = u_loss_ppm_per_cm
    u_alpha_cm_inv = u_alpha_ppm_per_cm * 1e-6

若后续要扣除参考谱且参考谱不确定度不可忽略，则应进一步使用平方和传播：

    u_alpha = sqrt(u_sample_loss^2 + u_reference_loss^2)

常用命令:
    python scripts/propagate_mc_tau_to_absorption_uncertainty.py

输出:
    output/results/continuum/CIA/absorption_uncertainty_interp_9120_9820_step0p01/
        {temperature}__{gas_pressure}.csv
        all_groups_wide_ppm_per_cm.csv
        all_groups_wide_cm_inv.csv
        summary.csv
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d

C_CM_PER_S = 2.99792458e10
TAU_US_TO_PPM_PER_CM = 1e12 / C_CM_PER_S


def parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"Invalid decimal value: {value}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"Invalid decimal value: {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate MC tau uncertainty to absorption/loss uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/results/continuum/CIA"),
        help="Continuum CIA output root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/results/continuum/CIA/"
            "absorption_uncertainty_interp_9120_9820_step0p01"
        ),
        help="Output directory.",
    )
    parser.add_argument("--start", type=parse_decimal, default=Decimal("9120"))
    parser.add_argument("--end", type=parse_decimal, default=Decimal("9820"))
    parser.add_argument("--step", type=parse_decimal, default=Decimal("0.01"))
    parser.add_argument("--x-column", default="wavenumber")
    parser.add_argument("--tau-column", default="tau_fit_us_mc_mean_us")
    parser.add_argument("--tau-uncertainty-column", default="tau_fit_us_mc_uncertainty_us")
    parser.add_argument(
        "--method",
        choices=("pchip", "linear"),
        default="pchip",
        help="Interpolation method for tau and u_tau.",
    )
    parser.add_argument(
        "--edge-fill",
        choices=("nearest", "none"),
        default="nearest",
        help="How to fill grid points just outside measured x range.",
    )
    return parser.parse_args()


def make_grid(start: Decimal, end: Decimal, step: Decimal) -> np.ndarray:
    if step <= 0:
        raise SystemExit("--step must be > 0")
    if end < start:
        raise SystemExit("--end must be >= --start")
    values = []
    current = start
    while current <= end:
        values.append(float(current))
        current += step
    return np.asarray(values, dtype=float)


def interpolate(x: np.ndarray, y: np.ndarray, grid: np.ndarray, method: str, edge_fill: str) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if len(x_valid) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]

    unique_x, inverse = np.unique(x_valid, return_inverse=True)
    if len(unique_x) != len(x_valid):
        y_grouped = np.zeros(len(unique_x), dtype=float)
        for idx in range(len(unique_x)):
            y_grouped[idx] = float(np.nanmean(y_valid[inverse == idx]))
        x_valid = unique_x
        y_valid = y_grouped

    if method == "pchip":
        out = PchipInterpolator(x_valid, y_valid, extrapolate=False)(grid)
    else:
        out = interp1d(
            x_valid,
            y_valid,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )(grid)

    out = np.asarray(out, dtype=float)
    if edge_fill == "nearest":
        out[grid < x_valid[0]] = y_valid[0]
        out[grid > x_valid[-1]] = y_valid[-1]
    return out


def safe_group_name(path: Path, root: Path) -> str:
    group = "/".join(path.relative_to(root).parts[:-1])
    return group.replace("/", "__").replace(" ", "_")


def group_label(path: Path, root: Path) -> str:
    return "/".join(path.relative_to(root).parts[:-1])


def process_file(path: Path, root: Path, grid: np.ndarray, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    for column in (args.x_column, args.tau_column, args.tau_uncertainty_column):
        if column not in df.columns:
            raise SystemExit(f"{path} is missing column: {column}")

    x = pd.to_numeric(df[args.x_column], errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(df[args.tau_column], errors="coerce").to_numpy(dtype=float)
    u_tau = pd.to_numeric(df[args.tau_uncertainty_column], errors="coerce").to_numpy(dtype=float)

    tau_i = interpolate(x, tau, grid, args.method, args.edge_fill)
    u_tau_i = interpolate(x, u_tau, grid, args.method, args.edge_fill)
    valid_tau = np.isfinite(tau_i) & (tau_i > 0) & np.isfinite(u_tau_i)

    u_loss_ppm = np.full_like(grid, np.nan, dtype=float)
    u_loss_ppm[valid_tau] = TAU_US_TO_PPM_PER_CM * np.abs(u_tau_i[valid_tau]) / tau_i[valid_tau] ** 2
    u_alpha_ppm = u_loss_ppm.copy()
    u_alpha_cm_inv = u_alpha_ppm * 1e-6

    out = pd.DataFrame({
        "wavenumber": grid,
        "tau_fit_us_mc_mean_us": tau_i,
        "tau_fit_us_mc_uncertainty_us": u_tau_i,
        "u_loss_fit_ppm_per_cm": u_loss_ppm,
        "u_alpha_ppm_per_cm": u_alpha_ppm,
        "u_alpha_cm_inv": u_alpha_cm_inv,
    })
    summary = {
        "group": group_label(path, root),
        "n_points": int(len(out)),
        "nan_count": int(out["u_alpha_cm_inv"].isna().sum()),
        "mean_u_alpha_ppm_per_cm": float(out["u_alpha_ppm_per_cm"].mean()),
        "median_u_alpha_ppm_per_cm": float(out["u_alpha_ppm_per_cm"].median()),
        "max_u_alpha_ppm_per_cm": float(out["u_alpha_ppm_per_cm"].max()),
        "mean_u_alpha_cm_inv": float(out["u_alpha_cm_inv"].mean()),
        "median_u_alpha_cm_inv": float(out["u_alpha_cm_inv"].median()),
        "max_u_alpha_cm_inv": float(out["u_alpha_cm_inv"].max()),
        "source_csv": str(path),
    }
    return out, summary


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args.start, args.end, args.step)

    paths = sorted(root.glob("*/*/continuum_step2_fit_mc_uncertainty.csv"))
    if not paths:
        raise SystemExit(f"No continuum_step2_fit_mc_uncertainty.csv files found under {root}")

    summaries = []
    wide_ppm = pd.DataFrame({"wavenumber": grid})
    wide_cm = pd.DataFrame({"wavenumber": grid})
    for path in paths:
        out, summary = process_file(path, root, grid, args)
        safe_name = safe_group_name(path, root)
        out_path = output_dir / f"{safe_name}.csv"
        out.to_csv(out_path, index=False)
        summary["output_csv"] = str(out_path)
        summaries.append(summary)
        wide_ppm[f"{safe_name}_u_alpha_ppm_per_cm"] = out["u_alpha_ppm_per_cm"].to_numpy(dtype=float)
        wide_cm[f"{safe_name}_u_alpha_cm_inv"] = out["u_alpha_cm_inv"].to_numpy(dtype=float)
        print(f"Wrote: {out_path}")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "summary.csv"
    wide_ppm_path = output_dir / "all_groups_wide_ppm_per_cm.csv"
    wide_cm_path = output_dir / "all_groups_wide_cm_inv.csv"
    summary_df.to_csv(summary_path, index=False)
    wide_ppm.to_csv(wide_ppm_path, index=False)
    wide_cm.to_csv(wide_cm_path, index=False)

    print(f"Summary: {summary_path}")
    print(f"Wide ppm/cm: {wide_ppm_path}")
    print(f"Wide cm^-1: {wide_cm_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
