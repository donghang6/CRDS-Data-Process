#!/usr/bin/env python3
"""Smooth a selected region of processed ring-down times in a CSV.

使用说明
--------
这个脚本用于对已经处理好的 CSV 中某个波数范围内的衰荡时间做局部平滑。
它适合修正局部小折点或轻微不平滑，而不是重新全局拟合整条曲线。

默认只 dry-run 预览，不会修改 CSV；确认无误后加 --apply 才会写回。
默认不覆盖原列，而是写入新列：{column}_smooth。
如果希望直接替换原列，使用 --overwrite。

推荐用法
--------
优先使用默认的 Savitzky-Golay 局部多项式平滑：
    - 它比全局三次样条更不容易改变整体形状。
    - 只处理 --range 指定的区域。
    - 可用 --blend-width 在区域边界平滑接回原曲线，避免新跳变。

常用命令:
    # 预览：对 tau_fit_us 在 9281~9286 cm-1 范围做局部平滑
    conda run -n CRDS-Data-Process python scripts/smooth_processed_tau_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --range 9281 9286 \
      --column tau_fit_us \
      --window-cm1 2 \
      --blend-width 0.3

    # 执行：写入新列 tau_fit_us_smooth，不覆盖原 tau_fit_us
    conda run -n CRDS-Data-Process python scripts/smooth_processed_tau_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --range 9281 9286 \
      --column tau_fit_us \
      --window-cm1 2 \
      --blend-width 0.3 \
      --apply

    # 执行：直接覆盖 tau_fit_us，并同步更新 loss_fit_ppm_per_cm 和残差列
    conda run -n CRDS-Data-Process python scripts/smooth_processed_tau_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --range 9281 9286 \
      --column tau_fit_us \
      --window-cm1 2 \
      --blend-width 0.3 \
      --overwrite \
      --update-derived \
      --apply

参数说明:
    csv_path             要修改的 CSV。
    --range START END    需要平滑的波数范围，闭区间，单位 cm-1。
    --column NAME        要平滑的列，默认 tau_fit_us。
    --x-column NAME      波数列，默认 wavenumber。
    --method METHOD      平滑方法：savgol 或 spline，默认 savgol。
    --window-cm1 W       平滑窗口宽度，单位 cm-1。默认 2。
    --polyorder N        Savitzky-Golay 多项式阶数，默认 3。
    --context-width W    平滑时额外使用目标范围左右 W cm-1 的数据。默认等于 window-cm1。
    --blend-width W      在目标范围两端各用 W cm-1 渐变接回原曲线。默认 0.3。
    --output-column N    不覆盖时写入的新列名；默认 {column}_smooth。
    --overwrite          直接把平滑结果写回 --column。
    --update-derived     覆盖 tau_fit_us 或 loss_fit_ppm_per_cm 时同步更新成对列和残差列。
    --plot PATH          输出一张局部对比图。
    --no-backup          写回 CSV 时不生成 .bak 备份。
    --apply              真正写回 CSV；不加时只预览。
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

C_CM_PER_S = 2.99792458e10
TAU_US_TO_PPM_PER_CM = 1e12 / C_CM_PER_S
np = None
pd = None
savgol_filter = None
UnivariateSpline = None


@dataclass(frozen=True)
class SmoothResult:
    output_column: str
    target_rows: int
    fit_rows: int
    window_points: int | None
    max_abs_delta: float
    first_before: float
    first_after: float


def load_dependencies(*, need_plot: bool = False) -> None:
    global UnivariateSpline, np, pd, savgol_filter
    if np is None or pd is None or savgol_filter is None or UnivariateSpline is None:
        try:
            import numpy as numpy_module
            import pandas as pandas_module
            import scipy.interpolate as scipy_interpolate
            import scipy.signal as scipy_signal
        except ImportError as exc:
            raise SystemExit(
                "This script needs numpy, pandas, and scipy. Run it in the project "
                "environment, for example: conda run -n CRDS-Data-Process python "
                "scripts/smooth_processed_tau_region.py ..."
            ) from exc
        np = numpy_module
        pd = pandas_module
        savgol_filter = scipy_signal.savgol_filter
        UnivariateSpline = scipy_interpolate.UnivariateSpline

    if need_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise SystemExit(
                "Plot output needs matplotlib. Use the project conda environment."
            ) from exc
        return plt
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth a selected CSV region of processed ring-down times.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Processed CSV to edit.")
    parser.add_argument(
        "--range",
        required=True,
        nargs=2,
        metavar=("START", "END"),
        type=float,
        help="Wavenumber range to smooth, inclusive.",
    )
    parser.add_argument(
        "--column",
        default="tau_fit_us",
        help="Column to smooth. Default: tau_fit_us.",
    )
    parser.add_argument(
        "--x-column",
        default="wavenumber",
        help="X/wavenumber column. Default: wavenumber.",
    )
    parser.add_argument(
        "--method",
        choices=("savgol", "spline"),
        default="savgol",
        help="Smoothing method. Default: savgol.",
    )
    parser.add_argument(
        "--window-cm1",
        type=float,
        default=2.0,
        help="Smoothing window width in cm-1. Default: 2.",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay smoothing. Default: 3.",
    )
    parser.add_argument(
        "--context-width",
        type=float,
        help="Extra data width on each side used for smoothing. Default: window-cm1.",
    )
    parser.add_argument(
        "--blend-width",
        type=float,
        default=0.3,
        help="Edge blend width inside the target region in cm-1. Default: 0.3.",
    )
    parser.add_argument(
        "--spline-s",
        type=float,
        help="Smoothing factor for --method spline. Default: estimated from data.",
    )
    parser.add_argument(
        "--output-column",
        help="New column for smoothed values. Default: <column>_smooth.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --column instead of writing a new column.",
    )
    parser.add_argument(
        "--update-derived",
        action="store_true",
        help=(
            "When overwriting tau_fit_us or loss_fit_ppm_per_cm, update the paired "
            "fit column and residual columns in the smoothed region."
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional PNG path for original-vs-smoothed comparison.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file before writing.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the CSV. Without this flag, only print a dry run.",
    )
    return parser.parse_args()


def normalize_range(start: float, end: float) -> tuple[float, float]:
    return (start, end) if start <= end else (end, start)


def ensure_columns(df, columns: list[str], csv_path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {', '.join(missing)}")


def build_unique_xy(x, y):
    smooth_df = (
        pd.DataFrame({"x": x, "y": y})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
    )
    return (
        smooth_df["x"].to_numpy(dtype=float),
        smooth_df["y"].to_numpy(dtype=float),
    )


def robust_scale(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    center = np.nanmedian(values)
    return 1.4826 * float(np.nanmedian(np.abs(values - center)))


def savgol_window_points(x_fit, window_cm1: float, polyorder: int) -> int:
    spacing = np.diff(np.sort(x_fit))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    if len(spacing) == 0:
        raise SystemExit("Cannot estimate x spacing for Savitzky-Golay smoothing.")

    median_spacing = float(np.median(spacing))
    window_points = max(int(round(window_cm1 / median_spacing)), polyorder + 2, 5)
    if window_points % 2 == 0:
        window_points += 1
    if window_points >= len(x_fit):
        window_points = len(x_fit) - 1
        if window_points % 2 == 0:
            window_points -= 1
    if window_points <= polyorder or window_points < 5:
        raise SystemExit(
            "Not enough points for Savitzky-Golay smoothing. "
            "Increase --range/--context-width or reduce --polyorder."
        )
    return window_points


def smooth_values(
    x_fit,
    y_fit,
    x_target,
    method: str,
    window_cm1: float,
    polyorder: int,
    spline_s: float | None,
) -> tuple[object, int | None]:
    if method == "savgol":
        window_points = savgol_window_points(
            x_fit=x_fit,
            window_cm1=window_cm1,
            polyorder=polyorder,
        )
        y_smooth = savgol_filter(
            y_fit,
            window_length=window_points,
            polyorder=polyorder,
            mode="interp",
        )
        return np.interp(x_target, x_fit, y_smooth), window_points

    if len(x_fit) < 4:
        raise SystemExit("Spline smoothing needs at least 4 unique finite fit points.")
    if spline_s is None:
        scale = robust_scale(np.diff(y_fit))
        spline_s = max(len(x_fit) * scale**2, 1e-12)
    spline = UnivariateSpline(x_fit, y_fit, k=3, s=spline_s)
    return spline(x_target), None


def blend_with_original(x_target, original, smoothed, start: float, end: float, width: float):
    if width <= 0:
        return smoothed
    left_weight = np.clip((x_target - start) / width, 0.0, 1.0)
    right_weight = np.clip((end - x_target) / width, 0.0, 1.0)
    weight = np.minimum(left_weight, right_weight)
    return original * (1.0 - weight) + smoothed * weight


def apply_region_smoothing(
    df,
    x_column: str,
    y_column: str,
    output_column: str,
    wave_range: tuple[float, float],
    method: str,
    window_cm1: float,
    polyorder: int,
    context_width: float,
    blend_width: float,
    spline_s: float | None,
) -> SmoothResult:
    ensure_columns(df, [x_column, y_column], Path("<csv>"))
    if window_cm1 <= 0:
        raise SystemExit("--window-cm1 must be > 0")
    if context_width < 0:
        raise SystemExit("--context-width must be >= 0")
    if blend_width < 0:
        raise SystemExit("--blend-width must be >= 0")
    if polyorder < 1:
        raise SystemExit("--polyorder must be >= 1")

    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    start, end = wave_range

    target_mask = np.isfinite(x) & np.isfinite(y) & (x >= start) & (x <= end)
    fit_mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= start - context_width)
        & (x <= end + context_width)
    )
    if int(target_mask.sum()) == 0:
        raise SystemExit(f"No finite rows found in target range: {start:g} ~ {end:g} cm-1")

    x_fit, y_fit = build_unique_xy(x[fit_mask], y[fit_mask])
    if len(x_fit) < max(polyorder + 2, 5):
        raise SystemExit(
            "Not enough finite fit points. Increase --context-width or target range."
        )

    x_target = x[target_mask]
    original = y[target_mask]
    smoothed, window_points = smooth_values(
        x_fit=x_fit,
        y_fit=y_fit,
        x_target=x_target,
        method=method,
        window_cm1=window_cm1,
        polyorder=polyorder,
        spline_s=spline_s,
    )
    final = blend_with_original(
        x_target=x_target,
        original=original,
        smoothed=smoothed,
        start=start,
        end=end,
        width=blend_width,
    )

    if output_column not in df.columns:
        df[output_column] = df[y_column]
    df.loc[target_mask, output_column] = final
    delta = final - original

    return SmoothResult(
        output_column=output_column,
        target_rows=int(target_mask.sum()),
        fit_rows=len(x_fit),
        window_points=window_points,
        max_abs_delta=float(np.nanmax(np.abs(delta))),
        first_before=float(original[0]),
        first_after=float(final[0]),
    )


def update_derived_columns(df, target_mask, edited_column: str) -> None:
    if edited_column == "tau_fit_us":
        tau = pd.to_numeric(df.loc[target_mask, "tau_fit_us"], errors="coerce").to_numpy(
            dtype=float
        )
        df.loc[target_mask, "loss_fit_ppm_per_cm"] = TAU_US_TO_PPM_PER_CM / tau
    elif edited_column == "loss_fit_ppm_per_cm":
        loss = pd.to_numeric(
            df.loc[target_mask, "loss_fit_ppm_per_cm"],
            errors="coerce",
        ).to_numpy(dtype=float)
        df.loc[target_mask, "tau_fit_us"] = TAU_US_TO_PPM_PER_CM / loss
    else:
        raise SystemExit(
            "--update-derived only supports --column tau_fit_us or "
            "--column loss_fit_ppm_per_cm"
        )

    if {"loss_ppm_per_cm", "loss_fit_ppm_per_cm"}.issubset(df.columns):
        df.loc[target_mask, "loss_residual_ppm_per_cm"] = (
            pd.to_numeric(df.loc[target_mask, "loss_ppm_per_cm"], errors="coerce")
            - pd.to_numeric(df.loc[target_mask, "loss_fit_ppm_per_cm"], errors="coerce")
        )
    if {"tau_us", "tau_fit_us"}.issubset(df.columns):
        df.loc[target_mask, "tau_residual_us"] = (
            pd.to_numeric(df.loc[target_mask, "tau_us"], errors="coerce")
            - pd.to_numeric(df.loc[target_mask, "tau_fit_us"], errors="coerce")
        )


def unique_backup_path(csv_path: Path) -> Path:
    candidate = csv_path.with_name(f"{csv_path.name}.bak")
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = csv_path.with_name(f"{csv_path.name}.bak{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def save_plot(
    df,
    x_column: str,
    source_column: str,
    output_column: str,
    wave_range: tuple[float, float],
    context_width: float,
    plot_path: Path,
) -> None:
    plt = load_dependencies(need_plot=True)
    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    source = pd.to_numeric(df[source_column], errors="coerce").to_numpy(dtype=float)
    output = pd.to_numeric(df[output_column], errors="coerce").to_numpy(dtype=float)
    start, end = wave_range
    view_mask = (
        np.isfinite(x)
        & (x >= start - context_width)
        & (x <= end + context_width)
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.plot(x[view_mask], source[view_mask], "-", lw=1.1, color="tab:red", label=source_column)
    ax.plot(x[view_mask], output[view_mask], "-", lw=1.2, color="tab:blue", label=output_column)
    ax.axvspan(start, end, color="0.9", alpha=0.4, lw=0)
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel(source_column)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_dependencies()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV does not exist: {csv_path}")
    if args.update_derived and not args.overwrite:
        raise SystemExit("--update-derived requires --overwrite")

    wave_range = normalize_range(args.range[0], args.range[1])
    context_width = args.context_width if args.context_width is not None else args.window_cm1
    output_column = (
        args.column
        if args.overwrite
        else args.output_column or f"{args.column}_smooth"
    )

    df = pd.read_csv(csv_path)
    ensure_columns(df, [args.x_column, args.column], csv_path)
    original_column_for_plot = args.column
    if args.overwrite and args.plot:
        original_column_for_plot = f"{args.column}__before_smooth"
        df[original_column_for_plot] = df[args.column]

    result = apply_region_smoothing(
        df=df,
        x_column=args.x_column,
        y_column=args.column,
        output_column=output_column,
        wave_range=wave_range,
        method=args.method,
        window_cm1=args.window_cm1,
        polyorder=args.polyorder,
        context_width=context_width,
        blend_width=args.blend_width,
        spline_s=args.spline_s,
    )

    x = pd.to_numeric(df[args.x_column], errors="coerce").to_numpy(dtype=float)
    target_mask = np.isfinite(x) & (x >= wave_range[0]) & (x <= wave_range[1])
    if args.update_derived:
        update_derived_columns(df, target_mask=target_mask, edited_column=args.column)

    if args.plot:
        save_plot(
            df=df,
            x_column=args.x_column,
            source_column=original_column_for_plot,
            output_column=output_column,
            wave_range=wave_range,
            context_width=context_width,
            plot_path=args.plot.expanduser().resolve(),
        )

    if original_column_for_plot != args.column:
        df = df.drop(columns=[original_column_for_plot])

    print(f"CSV: {csv_path}")
    print(f"Range: {wave_range[0]:g} ~ {wave_range[1]:g} cm-1")
    print(f"Column: {args.column}")
    print(f"Output column: {result.output_column}")
    print(f"Method: {args.method}")
    print(f"Window: {args.window_cm1:g} cm-1")
    print(f"Context width: {context_width:g} cm-1")
    print(f"Blend width: {args.blend_width:g} cm-1")
    print(f"Target rows: {result.target_rows}")
    print(f"Fit rows: {result.fit_rows}")
    if result.window_points is not None:
        print(f"Savitzky-Golay window points: {result.window_points}")
    print(f"First target value: {result.first_before:.10g} -> {result.first_after:.10g}")
    print(f"Max abs delta in target region: {result.max_abs_delta:.10g}")
    if args.plot:
        print(f"Plot: {args.plot.expanduser().resolve()}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to modify the CSV.")
        return

    if not args.no_backup:
        backup_path = unique_backup_path(csv_path)
        shutil.copy2(csv_path, backup_path)
        print(f"Backup: {backup_path}")

    df.to_csv(csv_path, index=False)
    print(f"Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()
