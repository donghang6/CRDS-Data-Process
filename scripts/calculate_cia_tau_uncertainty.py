#!/usr/bin/env python3
"""Calculate Type-A uncertainty of averaged ringdown time for CIA raw data.

使用说明
--------
默认扫描 `data/raw/CIA` 下所有温度和气体压力目录。每个原始 txt 文件对应一个
波数点，文件内部包含多次 ringdown 事件。脚本对每个波数点执行：

    1. 读取原始衰荡事件 tau_i。
    2. 按 sigma-clip 或 IQR 剔除 tau_i 异常值。
    3. 计算平均衰荡时间 tau_mean_us。
    4. 计算标准差 tau_std_us。
    5. 计算平均值标准不确定度 tau_sem_us = tau_std_us / sqrt(n_kept)。

这里的 `tau_sem_us` 就是平均衰荡时间的 A 类标准不确定度，可以继续传播到
loss、alpha 和二元吸收系数 B。

温度和压力只做异常值剔除后的平均值统计，输出 `temperature_*_mean` 和
`pressure_torr_mean`。它们不在本脚本中计算 A 类不确定度；后续误差传播时，
温度和压力的不确定度应使用仪器精度或标定结果给出的 B 类不确定度。

常用命令:
    # 处理全部 CIA 温度数据
    python scripts/calculate_cia_tau_uncertainty.py

    # 改变异常剔除阈值
    python scripts/calculate_cia_tau_uncertainty.py --sigma 4

    # 使用 IQR 异常剔除
    python scripts/calculate_cia_tau_uncertainty.py --filter-method iqr --iqr-factor 1.5

    # 只处理某一组
    python scripts/calculate_cia_tau_uncertainty.py --target '333K/Ar 500Torr'

输出:
    output/results/uncertainty/CIA/{temperature}/{gas pressure}/tau_uncertainty.csv
    output/results/uncertainty/CIA/cia_tau_uncertainty_all.csv
    output/results/uncertainty/CIA/cia_tau_uncertainty_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from crds_process.io.readers import load_scan_directory
from crds_process.ringdown.filtering import filter_ringdown_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate averaged ringdown-time uncertainty for CIA raw data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/CIA"),
        help="CIA raw data root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/results/uncertainty/CIA"),
        help="Output root.",
    )
    parser.add_argument(
        "--filter-method",
        choices=("sigma_clip", "iqr"),
        default="sigma_clip",
        help="Outlier rejection method for tau events inside each raw file.",
    )
    parser.add_argument("--sigma", type=float, default=3.0, help="Sigma threshold for sigma_clip.")
    parser.add_argument("--iqr-factor", type=float, default=1.5, help="IQR factor for iqr filtering.")
    parser.add_argument("--min-events", type=int, default=10, help="Minimum kept events per wavenumber.")
    parser.add_argument(
        "--min-pressure-deviation",
        type=float,
        default=0.5,
        help="Minimum absolute pressure deviation required to reject pressure events, in Torr.",
    )
    parser.add_argument(
        "--min-temperature-deviation",
        type=float,
        default=0.05,
        help="Minimum absolute temperature deviation required to reject temperature events, in deg C.",
    )
    parser.add_argument(
        "--temperature-tolerance-c",
        type=float,
        default=15.0,
        help=(
            "Allowed deviation from the temperature folder label, in deg C. "
            "For example, 273K means about -0.15 deg C."
        ),
    )
    parser.add_argument(
        "--pressure-tolerance-torr",
        type=float,
        default=100.0,
        help="Allowed deviation from the pressure label, in Torr.",
    )
    parser.add_argument(
        "--no-folder-physical-filter",
        action="store_true",
        help="Disable physical filtering from folder labels such as 273K and 500Torr.",
    )
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "Optional relative target under raw root, e.g. '333K/Ar 500Torr'. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--no-per-group",
        action="store_true",
        help="Do not write per-group tau_uncertainty.csv files.",
    )
    return parser.parse_args()


def discover_groups(raw_root: Path, targets: list[str] | None) -> list[Path]:
    if not raw_root.exists():
        raise SystemExit(f"Raw root does not exist: {raw_root}")

    if targets:
        groups = [raw_root / target for target in targets]
        missing = [str(path) for path in groups if not path.is_dir()]
        if missing:
            raise SystemExit("Target directories do not exist:\n" + "\n".join(missing))
        return groups

    txt_files = [
        path
        for path in raw_root.rglob("*.txt")
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    ]
    groups = sorted({path.parent for path in txt_files})
    if not groups:
        raise SystemExit(f"No raw txt files found under {raw_root}")
    return groups


def relative_group_parts(group_dir: Path, raw_root: Path) -> tuple[str, str, str]:
    rel = group_dir.relative_to(raw_root)
    parts = rel.parts
    temperature_group = parts[0] if len(parts) >= 1 else ""
    gas_pressure = parts[1] if len(parts) >= 2 else ""
    group = "/".join(parts)
    return group, temperature_group, gas_pressure


def temperature_c_from_label(label: str) -> float | None:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*K", label, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1)) - 273.15


def pressure_torr_from_label(label: str) -> float | None:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*Torr", label, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def robust_keep_mask(
    values: np.ndarray,
    *,
    sigma: float,
    iqr_factor: float,
    min_points: int,
    min_abs_deviation: float,
) -> np.ndarray:
    finite = np.isfinite(values)
    if int(finite.sum()) < min_points:
        return finite

    finite_values = values[finite]
    median = float(np.nanmedian(finite_values))
    mad = float(np.nanmedian(np.abs(finite_values - median)))
    if np.isfinite(mad) and mad > 0:
        robust_z = np.full(len(values), np.nan, dtype=float)
        robust_z[finite] = 0.67448975 * (values[finite] - median) / mad
        outlier = finite & (np.abs(robust_z) > sigma)
        outlier &= np.abs(values - median) >= min_abs_deviation
        return finite & ~outlier

    q1, q3 = np.nanpercentile(finite_values, [25, 75])
    iqr = float(q3 - q1)
    if np.isfinite(iqr) and iqr > 0:
        lo = q1 - iqr_factor * iqr
        hi = q3 + iqr_factor * iqr
        outlier = finite & ((values < lo) | (values > hi))
        outlier &= np.abs(values - median) >= min_abs_deviation
        return finite & ~outlier

    return finite


def robust_mean(
    values: np.ndarray,
    *,
    sigma: float,
    iqr_factor: float,
    min_points: int,
    min_abs_deviation: float,
    expected_value: float | None = None,
    expected_tolerance: float | None = None,
) -> tuple[float, int, int]:
    finite = np.isfinite(values)
    candidate = finite.copy()
    if expected_value is not None and expected_tolerance is not None and expected_tolerance > 0:
        candidate = finite & (np.abs(values - expected_value) <= expected_tolerance)
        if not candidate.any():
            return np.nan, 0, int(finite.sum())

    keep = robust_keep_mask(
        values[candidate],
        sigma=sigma,
        iqr_factor=iqr_factor,
        min_points=min_points,
        min_abs_deviation=min_abs_deviation,
    )
    full_keep = np.zeros(len(values), dtype=bool)
    full_keep[np.where(candidate)[0][keep]] = True
    if not full_keep.any():
        return np.nan, 0, int(finite.sum())
    n_finite = int(finite.sum())
    n_kept = int(full_keep.sum())
    return float(np.nanmean(values[full_keep])), n_kept, n_finite - n_kept


def weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    finite = mask & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    return float(np.average(values[finite], weights=weights[finite]))


def process_group(
    group_dir: Path,
    raw_root: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    scans = load_scan_directory(group_dir)
    if not scans:
        return pd.DataFrame()

    group, temperature_group, gas_pressure = relative_group_parts(group_dir, raw_root)
    expected_temperature_c = None
    expected_pressure_torr = None
    if not args.no_folder_physical_filter:
        expected_temperature_c = temperature_c_from_label(temperature_group)
        expected_pressure_torr = pressure_torr_from_label(gas_pressure)

    records = []
    for scan in scans:
        tau_filtered = filter_ringdown_times(
            scan.tau,
            method=args.filter_method,
            sigma=args.sigma,
            iqr_factor=args.iqr_factor,
        )
        if len(tau_filtered) < args.min_events:
            continue

        tau_std = float(np.std(tau_filtered, ddof=1))
        tau_sem = float(tau_std / np.sqrt(len(tau_filtered)))
        temperature_c, n_temperature_kept, n_temperature_removed = robust_mean(
            np.asarray(scan.temperature, dtype=float),
            sigma=args.sigma,
            iqr_factor=args.iqr_factor,
            min_points=args.min_events,
            min_abs_deviation=args.min_temperature_deviation,
            expected_value=expected_temperature_c,
            expected_tolerance=args.temperature_tolerance_c,
        )
        pressure_torr, n_pressure_kept, n_pressure_removed = robust_mean(
            np.asarray(scan.pressure, dtype=float),
            sigma=args.sigma,
            iqr_factor=args.iqr_factor,
            min_points=args.min_events,
            min_abs_deviation=args.min_pressure_deviation,
            expected_value=expected_pressure_torr,
            expected_tolerance=args.pressure_tolerance_torr,
        )
        n_removed = int(len(scan.tau) - len(tau_filtered))
        records.append({
            "group": group,
            "temperature_group": temperature_group,
            "gas_pressure": gas_pressure,
            "wavenumber": scan.meta.wavenumber,
            "tau_mean_us": float(np.mean(tau_filtered)),
            "tau_std_us": tau_std,
            "tau_sem_us": tau_sem,
            "tau_uncertainty_us": tau_sem,
            "n_raw": int(len(scan.tau)),
            "n_kept": int(len(tau_filtered)),
            "n_removed": n_removed,
            "removed_fraction": n_removed / len(scan.tau) if len(scan.tau) else np.nan,
            "temperature_c_mean": temperature_c,
            "temperature_k_mean": temperature_c + 273.15 if np.isfinite(temperature_c) else np.nan,
            "pressure_torr_mean": pressure_torr,
            "n_temperature_kept": n_temperature_kept,
            "n_temperature_removed": n_temperature_removed,
            "n_pressure_kept": n_pressure_kept,
            "n_pressure_removed": n_pressure_removed,
            "filter_method": args.filter_method,
            "sigma": args.sigma,
            "iqr_factor": args.iqr_factor,
            "min_events": args.min_events,
            "source_dir": str(group_dir),
        })

    df = pd.DataFrame(records).sort_values("wavenumber")
    return df


def summarize(all_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for group, group_df in all_df.groupby("group", sort=True):
        temperature_k = pd.to_numeric(group_df["temperature_k_mean"], errors="coerce").to_numpy(dtype=float)
        pressure_torr = pd.to_numeric(group_df["pressure_torr_mean"], errors="coerce").to_numpy(dtype=float)
        weights = pd.to_numeric(group_df["n_kept"], errors="coerce").to_numpy(dtype=float)
        temperature_keep = robust_keep_mask(
            temperature_k,
            sigma=args.sigma,
            iqr_factor=args.iqr_factor,
            min_points=args.min_events,
            min_abs_deviation=args.min_temperature_deviation,
        )
        pressure_keep = robust_keep_mask(
            pressure_torr,
            sigma=args.sigma,
            iqr_factor=args.iqr_factor,
            min_points=args.min_events,
            min_abs_deviation=args.min_pressure_deviation,
        )
        finite_both = np.isfinite(temperature_k) & np.isfinite(pressure_torr)
        keep = finite_both & temperature_keep & pressure_keep
        n_used = int(keep.sum())
        n_invalid_or_outlier = int(len(group_df) - n_used)
        rows.append({
            "group": group,
            "temperature_group": group_df["temperature_group"].iloc[0],
            "gas_pressure": group_df["gas_pressure"].iloc[0],
            "n_wavenumber_points": int(len(group_df)),
            "n_pressure_temperature_points_used": n_used,
            "n_pressure_temperature_points_removed": n_invalid_or_outlier,
            "n_events_raw": int(group_df["n_raw"].sum()),
            "n_events_kept": int(group_df["n_kept"].sum()),
            "n_events_removed": int(group_df["n_removed"].sum()),
            "mean_tau_uncertainty_us": float(group_df["tau_uncertainty_us"].mean()),
            "median_tau_uncertainty_us": float(group_df["tau_uncertainty_us"].median()),
            "max_tau_uncertainty_us": float(group_df["tau_uncertainty_us"].max()),
            "mean_tau_std_us": float(group_df["tau_std_us"].mean()),
            "temperature_k_mean": weighted_mean(temperature_k, weights, keep),
            "pressure_torr_mean": weighted_mean(pressure_torr, weights, keep),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    groups = discover_groups(raw_root, args.target)
    all_frames = []
    for group_dir in groups:
        df = process_group(group_dir, raw_root, args)
        if df.empty:
            print(f"Skip empty group: {group_dir}")
            continue
        all_frames.append(df)
        if not args.no_per_group:
            group, _, _ = relative_group_parts(group_dir, raw_root)
            out_path = output_root / group / "tau_uncertainty.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"Wrote group: {out_path}")

    if not all_frames:
        raise SystemExit("No usable CIA raw groups were processed.")

    all_df = pd.concat(all_frames, ignore_index=True)
    all_path = output_root / "cia_tau_uncertainty_all.csv"
    summary_path = output_root / "cia_tau_uncertainty_summary.csv"
    all_df.to_csv(all_path, index=False)
    summarize(all_df, args).to_csv(summary_path, index=False)

    print(f"Raw root: {raw_root}")
    print(f"Output root: {output_root}")
    print(f"Groups processed: {all_df['group'].nunique()}")
    print(f"Wavenumber points: {len(all_df)}")
    print(f"All points: {all_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
