#!/usr/bin/env python3
"""Summarize mean pressure and temperature for every raw CRDS data group.

使用说明
--------
默认只扫描 `data/raw/CIA` 下的原始 txt 文件。每个包含 txt 文件的叶子目录被视为
一组数据，例如 `333K/Ar 500Torr`。

原始 txt 文件格式按本项目约定读取：
    第 1 列: 衰荡时间 tau_us
    第 2 列: 拟合残差
    第 3 列: 温度 temperature_c
    第 4 列: 压力 pressure_torr

异常值剔除分两层进行：
    1. 单个 txt 文件内部，对温度/压力事件值做 robust 异常剔除，再得到该文件均值。
    2. 同一数据组内，对每个文件均值再次做 robust 异常剔除，再统计组平均值。

robust 方法优先使用 MAD robust z-score；如果 MAD 为 0 或不可用，则回退到 IQR。
为了避免把温控/压控的微小自然漂移误判为异常，只有超过绝对偏差下限的点才会
被剔除。默认压力下限为 0.5 Torr，温度下限为 0.05 °C。

常用命令:
    # 扫描 CIA raw 数据，输出简洁表格
    python scripts/summarize_pressure_temperature.py

    # 指定输出文件
    python scripts/summarize_pressure_temperature.py \
      --output output/results/CIA_pressure_temperature_summary.csv

    # 输出详细统计列和异常点明细
    python scripts/summarize_pressure_temperature.py \
      --detailed \
      --write-outliers

    # 调整异常值阈值
    python scripts/summarize_pressure_temperature.py \
      --sigma 4 \
      --iqr-factor 2

输出:
    - CIA_pressure_temperature_summary.csv: 每组剔除异常点后的平均压力/温度。
    - 默认不输出异常点明细；如需检查异常点，添加 --write-outliers。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY_FILENAME_PATTERN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+(\d{14})\.txt$")
WAVENUMBER_FILENAME_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\.txt$")

PRESSURE_COLUMNS = (
    "pressure",
    "pressure_torr",
    "pressure_mean",
    "Cavity Pressure /Torr",
)
TEMPERATURE_COLUMNS = (
    "temperature",
    "temperature_c",
    "temperature_mean",
    "Cavity Temperature Side 2 /C",
)
WAVENUMBER_COLUMNS = ("wavenumber", "Wavenumber", "nu", "Total Frequency /MHz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize raw CRDS pressure/temperature with outlier rejection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/CIA"),
        help="Root directory to scan. Default only scans CIA raw data.",
    )
    parser.add_argument(
        "--source",
        choices=("raw", "ringdown"),
        default="raw",
        help="Input type. raw scans txt files; ringdown scans ringdown_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results/CIA_pressure_temperature_summary.csv"),
        help="Output summary CSV.",
    )
    parser.add_argument(
        "--outliers-output",
        type=Path,
        help="Output outlier detail CSV. Default: <output stem>_outliers.csv.",
    )
    parser.add_argument("--pressure-column", help="Ringdown pressure column name. Default: auto detect.")
    parser.add_argument("--temperature-column", help="Ringdown temperature column name. Default: auto detect.")
    parser.add_argument("--x-column", help="Wavenumber column name. Default: auto detect.")
    parser.add_argument("--sigma", type=float, default=3.5, help="MAD robust z-score threshold.")
    parser.add_argument("--iqr-factor", type=float, default=1.5, help="IQR fallback threshold.")
    parser.add_argument(
        "--min-pressure-deviation",
        type=float,
        default=0.5,
        help="Minimum absolute pressure deviation required to reject a point, in Torr.",
    )
    parser.add_argument(
        "--min-temperature-deviation",
        type=float,
        default=0.05,
        help="Minimum absolute temperature deviation required to reject a point, in deg C.",
    )
    parser.add_argument("--min-points", type=int, default=5, help="Minimum points needed to reject outliers.")
    parser.add_argument(
        "--no-event-rejection",
        action="store_true",
        help="For raw txt files, do not reject event-level outliers inside each file.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Write all diagnostic columns. Default writes a compact summary table.",
    )
    parser.add_argument("--write-outliers", action="store_true", help="Write outlier detail CSV.")
    parser.add_argument("--no-outliers-output", action="store_true", help="Alias to disable outlier detail CSV.")
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: tuple[str, ...], explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if explicit in df.columns else None
    for name in candidates:
        if name in df.columns:
            return name
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        match = lower_map.get(name.strip().lower())
        if match is not None:
            return str(match)
    return None


def robust_outlier_result(
    values: np.ndarray,
    sigma: float,
    iqr_factor: float,
    min_points: int,
    min_abs_deviation: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    finite = np.isfinite(values)
    outlier = np.zeros(len(values), dtype=bool)
    if int(finite.sum()) < min_points:
        return finite, outlier, "none_not_enough_points"

    finite_values = values[finite]
    median = float(np.nanmedian(finite_values))
    mad = float(np.nanmedian(np.abs(finite_values - median)))
    if np.isfinite(mad) and mad > 0:
        robust_z = np.full(len(values), np.nan, dtype=float)
        robust_z[finite] = 0.67448975 * (values[finite] - median) / mad
        outlier = finite & (np.abs(robust_z) > sigma)
        outlier &= np.abs(values - median) >= min_abs_deviation
        return finite & ~outlier, outlier, "mad"

    q1, q3 = np.nanpercentile(finite_values, [25, 75])
    iqr = float(q3 - q1)
    if np.isfinite(iqr) and iqr > 0:
        lo = q1 - iqr_factor * iqr
        hi = q3 + iqr_factor * iqr
        outlier = finite & ((values < lo) | (values > hi))
        outlier &= np.abs(values - median) >= min_abs_deviation
        return finite & ~outlier, outlier, "iqr"

    return finite, outlier, "none_constant"


def combined_keep_mask(
    pressure: np.ndarray,
    temperature: np.ndarray,
    sigma: float,
    iqr_factor: float,
    min_points: int,
    min_pressure_deviation: float,
    min_temperature_deviation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    pressure_keep, pressure_outlier, pressure_method = robust_outlier_result(
        pressure,
        sigma,
        iqr_factor,
        min_points,
        min_pressure_deviation,
    )
    temperature_keep, temperature_outlier, temperature_method = robust_outlier_result(
        temperature,
        sigma,
        iqr_factor,
        min_points,
        min_temperature_deviation,
    )
    finite_both = np.isfinite(pressure) & np.isfinite(temperature)
    outlier = pressure_outlier | temperature_outlier
    keep = finite_both & pressure_keep & temperature_keep & ~outlier
    return keep, finite_both & pressure_outlier, finite_both & temperature_outlier, pressure_method, temperature_method


def parse_wavenumber_from_name(path: Path) -> float:
    legacy = LEGACY_FILENAME_PATTERN.match(path.name)
    if legacy:
        return float(legacy.group(2))
    simple = WAVENUMBER_FILENAME_PATTERN.match(path.name)
    if simple:
        return float(simple.group(1))
    return np.nan


def discover_raw_groups(root: Path) -> list[Path]:
    txt_files = [
        path
        for path in root.rglob("*.txt")
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    ]
    return sorted({path.parent for path in txt_files})


def rel_parts(path: Path, root: Path) -> tuple[str, str, str, str]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = rel.parts
    group = "/".join(parts) if parts else path.name
    first = parts[0] if len(parts) >= 1 else ""
    second = parts[1] if len(parts) >= 2 else ""
    third = parts[2] if len(parts) >= 3 else ""
    return group, first, second, third


def read_raw_file(path: Path, args: argparse.Namespace) -> dict | None:
    try:
        data = np.loadtxt(path)
    except Exception as exc:
        print(f"Skip unreadable file: {path} ({exc})")
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        print(f"Skip file with <4 columns: {path}")
        return None

    temperature = np.asarray(data[:, 2], dtype=float)
    pressure = np.asarray(data[:, 3], dtype=float)
    finite_both = np.isfinite(pressure) & np.isfinite(temperature)
    if not finite_both.any():
        return None

    if args.no_event_rejection:
        keep = finite_both
        outlier = np.zeros(len(pressure), dtype=bool)
        pressure_method = "none_disabled"
        temperature_method = "none_disabled"
    else:
        keep, pressure_outlier, temperature_outlier, pressure_method, temperature_method = combined_keep_mask(
            pressure,
            temperature,
            args.sigma,
            args.iqr_factor,
            args.min_points,
            args.min_pressure_deviation,
            args.min_temperature_deviation,
        )
        outlier = pressure_outlier | temperature_outlier
    if not keep.any():
        return None

    return {
        "file": str(path),
        "wavenumber": parse_wavenumber_from_name(path),
        "pressure_torr": float(np.nanmean(pressure[keep])),
        "temperature_c": float(np.nanmean(temperature[keep])),
        "n_events_total": int(len(pressure)),
        "n_events_finite": int(finite_both.sum()),
        "n_events_kept": int(keep.sum()),
        "n_event_outliers": int((finite_both & outlier).sum()),
        "event_pressure_method": pressure_method,
        "event_temperature_method": temperature_method,
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    return float(np.average(values[finite], weights=weights[finite]))


def stats(values: np.ndarray, weights: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    kept = values[mask]
    kept_weights = weights[mask]
    if len(kept) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_file_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": weighted_mean(kept, kept_weights),
        f"{prefix}_file_mean": float(np.nanmean(kept)),
        f"{prefix}_median": float(np.nanmedian(kept)),
        f"{prefix}_std": float(np.nanstd(kept, ddof=1)) if len(kept) > 1 else 0.0,
        f"{prefix}_min": float(np.nanmin(kept)),
        f"{prefix}_max": float(np.nanmax(kept)),
    }


def summarize_file_rows(
    rows: list[dict],
    group_path: Path,
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]] | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pressure = pd.to_numeric(df["pressure_torr"], errors="coerce").to_numpy(dtype=float)
    temperature = pd.to_numeric(df["temperature_c"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(df["n_events_kept"], errors="coerce").to_numpy(dtype=float)
    keep, pressure_outlier, temperature_outlier, pressure_method, temperature_method = combined_keep_mask(
        pressure,
        temperature,
        args.sigma,
        args.iqr_factor,
        args.min_points,
        args.min_pressure_deviation,
        args.min_temperature_deviation,
    )
    file_outlier = pressure_outlier | temperature_outlier
    group, level_1, level_2, level_3 = rel_parts(group_path, root)
    finite_both = np.isfinite(pressure) & np.isfinite(temperature)

    summary = {
        "group": group,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
        "source_dir": str(group_path),
        "n_files_total": int(len(df)),
        "n_files_finite_pressure_temperature": int(finite_both.sum()),
        "n_files_kept": int(keep.sum()),
        "n_file_outliers": int((finite_both & file_outlier).sum()),
        "file_outlier_fraction": (
            float((finite_both & file_outlier).sum() / finite_both.sum())
            if int(finite_both.sum()) > 0 else np.nan
        ),
        "n_events_total": int(df["n_events_total"].sum()),
        "n_events_finite": int(df["n_events_finite"].sum()),
        "n_events_kept_before_file_rejection": int(df["n_events_kept"].sum()),
        "n_event_outliers": int(df["n_event_outliers"].sum()),
        "n_events_used": int(df.loc[keep, "n_events_kept"].sum()),
        "pressure_outlier_method": pressure_method,
        "temperature_outlier_method": temperature_method,
    }
    summary.update(stats(pressure, weights, keep, "pressure_torr"))
    summary.update(stats(temperature, weights, keep, "temperature_c"))
    if np.isfinite(summary["temperature_c_mean"]):
        summary["temperature_k_mean"] = summary["temperature_c_mean"] + 273.15
    else:
        summary["temperature_k_mean"] = np.nan
    if np.isfinite(summary["temperature_c_median"]):
        summary["temperature_k_median"] = summary["temperature_c_median"] + 273.15
    else:
        summary["temperature_k_median"] = np.nan

    outlier_rows: list[dict] = []
    for idx in np.where(finite_both & file_outlier)[0]:
        reasons = []
        if pressure_outlier[idx]:
            reasons.append("pressure")
        if temperature_outlier[idx]:
            reasons.append("temperature")
        outlier_rows.append({
            "group": group,
            "file": df.loc[idx, "file"],
            "wavenumber": df.loc[idx, "wavenumber"],
            "pressure_torr": pressure[idx],
            "temperature_c": temperature[idx],
            "reason": "+".join(reasons) or "pressure_or_temperature",
            "n_events_kept": int(df.loc[idx, "n_events_kept"]),
        })
    return summary, outlier_rows


def process_raw(root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = discover_raw_groups(root)
    if not groups:
        raise SystemExit(f"No raw txt files found under {root}")

    summaries: list[dict] = []
    outliers: list[dict] = []
    for group_path in groups:
        rows = []
        for txt_path in sorted(group_path.glob("*.txt")):
            row = read_raw_file(txt_path, args)
            if row is not None:
                rows.append(row)
        result = summarize_file_rows(rows, group_path, root, args)
        if result is None:
            continue
        summary, group_outliers = result
        summaries.append(summary)
        outliers.extend(group_outliers)

    if not summaries:
        raise SystemExit("No usable raw data groups found.")
    return pd.DataFrame(summaries).sort_values("group"), pd.DataFrame(outliers)


def discover_ringdown_csvs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("ringdown_results.csv") if path.is_file())


def process_ringdown(root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    csvs = discover_ringdown_csvs(root)
    if not csvs:
        raise SystemExit(f"No ringdown_results.csv files found under {root}")

    summaries: list[dict] = []
    outliers: list[dict] = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        pressure_col = find_column(df, PRESSURE_COLUMNS, args.pressure_column)
        temperature_col = find_column(df, TEMPERATURE_COLUMNS, args.temperature_column)
        if pressure_col is None or temperature_col is None:
            print(f"Skip missing pressure/temperature columns: {csv_path}")
            continue

        x_col = find_column(df, WAVENUMBER_COLUMNS, args.x_column)
        pressure = pd.to_numeric(df[pressure_col], errors="coerce").to_numpy(dtype=float)
        temperature = pd.to_numeric(df[temperature_col], errors="coerce").to_numpy(dtype=float)
        keep, pressure_outlier, temperature_outlier, pressure_method, temperature_method = combined_keep_mask(
            pressure,
            temperature,
            args.sigma,
            args.iqr_factor,
            args.min_points,
            args.min_pressure_deviation,
            args.min_temperature_deviation,
        )
        outlier = pressure_outlier | temperature_outlier
        group_path = csv_path.parent
        group, level_1, level_2, level_3 = rel_parts(group_path, root)
        weights = np.ones(len(df), dtype=float)
        finite_both = np.isfinite(pressure) & np.isfinite(temperature)
        summary = {
            "group": group,
            "level_1": level_1,
            "level_2": level_2,
            "level_3": level_3,
            "source_dir": str(group_path),
            "n_files_total": int(len(df)),
            "n_files_finite_pressure_temperature": int(finite_both.sum()),
            "n_files_kept": int(keep.sum()),
            "n_file_outliers": int((finite_both & outlier).sum()),
            "file_outlier_fraction": (
                float((finite_both & outlier).sum() / finite_both.sum())
                if int(finite_both.sum()) > 0 else np.nan
            ),
            "n_events_total": np.nan,
            "n_events_finite": np.nan,
            "n_events_kept_before_file_rejection": np.nan,
            "n_event_outliers": np.nan,
            "n_events_used": np.nan,
            "pressure_outlier_method": pressure_method,
            "temperature_outlier_method": temperature_method,
        }
        summary.update(stats(pressure, weights, keep, "pressure_torr"))
        summary.update(stats(temperature, weights, keep, "temperature_c"))
        summary["temperature_k_mean"] = summary["temperature_c_mean"] + 273.15
        summary["temperature_k_median"] = summary["temperature_c_median"] + 273.15
        summaries.append(summary)

        x_values = (
            pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
            if x_col is not None else np.full(len(df), np.nan, dtype=float)
        )
        for idx in np.where(finite_both & outlier)[0]:
            outliers.append({
                "group": group,
                "file": str(csv_path),
                "wavenumber": x_values[idx],
                "pressure_torr": pressure[idx],
                "temperature_c": temperature[idx],
                "reason": "pressure_or_temperature",
                "n_events_kept": np.nan,
            })

    if not summaries:
        raise SystemExit("No usable ringdown CSV files found.")
    return pd.DataFrame(summaries).sort_values("group"), pd.DataFrame(outliers)


def compact_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for the final CIA pressure/temperature table."""
    records = []
    for _, row in summary_df.iterrows():
        if row.get("level_1") == "CIA":
            temperature_group = row.get("level_2", "")
            gas_pressure = row.get("level_3", "")
        else:
            temperature_group = row.get("level_1", "")
            gas_pressure_parts = [
                str(row.get("level_2", "") or ""),
                str(row.get("level_3", "") or ""),
            ]
            gas_pressure = "/".join(part for part in gas_pressure_parts if part)

        records.append({
            "temperature_group": temperature_group,
            "gas_pressure": gas_pressure,
            "pressure_torr": row.get("pressure_torr_mean", np.nan),
            "temperature_c": row.get("temperature_c_mean", np.nan),
            "temperature_k": row.get("temperature_k_mean", np.nan),
            "used_points": row.get("n_files_kept", np.nan),
            "removed_points": row.get("n_file_outliers", np.nan),
        })

    out = pd.DataFrame(records)
    for col in ("pressure_torr", "temperature_c", "temperature_k"):
        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
    for col in ("used_points", "removed_points"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out.sort_values(["temperature_group", "gas_pressure"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    if args.source == "raw":
        summary_df, outlier_df = process_raw(root, args)
    else:
        summary_df, outlier_df = process_ringdown(root, args)

    output_df = summary_df if args.detailed else compact_summary(summary_df)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output, index=False)

    outliers_output = None
    if args.write_outliers and not args.no_outliers_output:
        outliers_output = (
            args.outliers_output.expanduser().resolve()
            if args.outliers_output
            else output.with_name(f"{output.stem}_outliers.csv")
        )
        outliers_output.parent.mkdir(parents=True, exist_ok=True)
        outlier_df.to_csv(outliers_output, index=False)

    print(f"Input source: {args.source}")
    print(f"Input root: {root}")
    print(f"Summarized groups: {len(summary_df)}")
    print(f"Output summary: {output}")
    print(f"Output format: {'detailed' if args.detailed else 'compact'}")
    if outliers_output is not None:
        print(f"Output outliers: {outliers_output}")
    print(f"Total kept file points: {int(summary_df['n_files_kept'].sum())}")
    print(f"Total rejected file points: {int(summary_df['n_file_outliers'].sum())}")
    if "n_event_outliers" in summary_df.columns:
        print(f"Total rejected raw events: {int(pd.to_numeric(summary_df['n_event_outliers'], errors='coerce').sum())}")


if __name__ == "__main__":
    main()
