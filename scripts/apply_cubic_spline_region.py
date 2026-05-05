#!/usr/bin/env python3
"""Apply cubic-spline interpolation or regional B-spline fitting to a CSV.

使用说明
--------
这个脚本用于对已经处理好的 CSV 中某个或多个波数范围做局部三次样条处理。
典型用途是：某一小段拟合结果不够理想，但两侧数据可信，则用两侧锚点
对该小段重新插值。

默认只 dry-run 预览，不会修改 CSV；确认无误后加 --apply 才会写回。
默认不覆盖原列，而是把结果写入新列：{column}_cubic_spline。
如果希望直接替换原列，使用 --overwrite。

插值方式
--------
单区域锚点插值（旧用法）:
    1. 用 --range START END 指定要替换/生成的波数范围。
    2. 用 --anchor-width 指定该范围左右两侧各取多宽的数据作为锚点。
    3. 区间内部数据不参与样条拟合，只由两侧锚点插值得到。

多区域 B-spline 拟合（新增）:
    1. 重复使用 --region START END KNOTS_EVERY 指定多个区域。
    2. 每个区域使用自己的 KNOTS_EVERY，单位 cm-1。
    3. 样条拟合使用目标区域加上左右 anchor-width 的数据，只把目标区域写回。
    4. 多个区域彼此独立拟合，最后拼接到同一个输出列。
    5. KNOTS_EVERY 越小，曲线越贴近局部数据；越大，曲线越平滑。

常用命令:
    # 预览：对 tau_fit_us 在 9281~9286 cm-1 范围做局部三次样条插值，
    # 结果写入新列 tau_fit_us_cubic_spline
    python scripts/apply_cubic_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --range 9281 9286 \
      --column tau_fit_us \
      --anchor-width 5

    # 执行：写入新列，不覆盖原 tau_fit_us
    python scripts/apply_cubic_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --range 9281 9286 \
      --column tau_fit_us \
      --anchor-width 5 \
      --apply

    # 执行：直接覆盖 tau_fit_us，并同步更新 loss_fit_ppm_per_cm 和残差列
    python scripts/apply_cubic_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --range 9281 9286 \
      --column tau_fit_us \
      --anchor-width 5 \
      --overwrite \
      --update-derived \
      --apply

    # 多区域：每个区域使用不同的 knots_every，独立拟合后拼接到新列
    python scripts/apply_cubic_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --column tau_fit_us \
      --anchor-width 5 \
      --region 9281 9286 0.5 \
      --region 9400 9420 2.0 \
      --apply

参数说明:
    csv_path            要修改的 CSV。
    --range START END   需要锚点插值的波数范围，闭区间，单位 cm-1。
    --region A B K      多区域 B-spline 拟合。A/B 为波数范围，K 为该区域 knots_every。
    --column NAME       要插值的列，默认 tau_fit_us。
    --x-column NAME     波数列，默认 wavenumber。
    --anchor-width W    目标范围左右两侧各取 W cm-1 作为样条锚点，默认 5。
    --output-column N   不覆盖时写入的新列名；默认 {column}_cubic_spline。
    --overwrite         直接把插值结果写回 --column。
    --update-derived    覆盖 tau_fit_us 或 loss_fit_ppm_per_cm 时同步更新成对列和残差列。
    --bc-type TYPE      样条边界条件，默认 natural；也可用 not-a-knot。
    --no-backup         写回 CSV 时不生成 .bak 备份。
    --apply             真正写回 CSV；不加时只预览。
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
CubicSpline = None
LSQUnivariateSpline = None


@dataclass(frozen=True)
class InterpolationResult:
    output_column: str
    start: float
    end: float
    target_rows: int
    left_anchor_rows: int
    right_anchor_rows: int
    max_abs_delta: float
    first_before: float
    first_after: float
    mode: str = "anchor"
    knots_every: float | None = None
    internal_knots: int = 0
    fit_rows: int = 0


def load_dependencies() -> None:
    global CubicSpline, LSQUnivariateSpline, np, pd
    if np is not None and pd is not None and CubicSpline is not None and LSQUnivariateSpline is not None:
        return
    try:
        import numpy as numpy_module
        import pandas as pandas_module
        import scipy.interpolate as scipy_interpolate
    except ImportError as exc:
        raise SystemExit(
            "This script needs numpy, pandas, and scipy. Run it in the project "
            "environment, for example: conda run -n CRDS-Data-Process python "
            "scripts/apply_cubic_spline_region.py ..."
        ) from exc
    np = numpy_module
    pd = pandas_module
    CubicSpline = scipy_interpolate.CubicSpline
    LSQUnivariateSpline = scipy_interpolate.LSQUnivariateSpline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply cubic-spline interpolation to one CSV region.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Processed CSV to edit.")
    parser.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        type=float,
        help="Single wavenumber range for anchor interpolation, inclusive.",
    )
    parser.add_argument(
        "--region",
        action="append",
        nargs=3,
        metavar=("START", "END", "KNOTS_EVERY"),
        type=float,
        help=(
            "Regional B-spline fit with per-region knot spacing. "
            "Repeat this option for multiple regions."
        ),
    )
    parser.add_argument(
        "--column",
        default="tau_fit_us",
        help="Column to interpolate. Default: tau_fit_us.",
    )
    parser.add_argument(
        "--x-column",
        default="wavenumber",
        help="X/wavenumber column. Default: wavenumber.",
    )
    parser.add_argument(
        "--anchor-width",
        type=float,
        default=5.0,
        help="Anchor width on each side of the target region, in cm-1. Default: 5.",
    )
    parser.add_argument(
        "--output-column",
        help="New column for interpolated values. Default: <column>_cubic_spline.",
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
            "fit column and residual columns in the interpolated region."
        ),
    )
    parser.add_argument(
        "--bc-type",
        choices=("natural", "not-a-knot"),
        default="natural",
        help="CubicSpline boundary condition. Default: natural.",
    )
    parser.add_argument(
        "--min-anchor-points",
        type=int,
        default=4,
        help="Minimum finite anchor points required on each side. Default: 4.",
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


def ensure_columns(df: pd.DataFrame, columns: list[str], csv_path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {', '.join(missing)}")


def build_unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor_df = (
        pd.DataFrame({"x": x, "y": y})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
    )
    return (
        anchor_df["x"].to_numpy(dtype=float),
        anchor_df["y"].to_numpy(dtype=float),
    )


def make_internal_knots(x_min: float, x_max: float, knots_every: float) -> np.ndarray:
    if knots_every <= 0:
        raise SystemExit("KNOTS_EVERY in --region must be > 0")
    return np.arange(x_min + knots_every, x_max, knots_every, dtype=float)


def apply_spline_region(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_column: str,
    wave_range: tuple[float, float],
    anchor_width: float,
    bc_type: str,
    min_anchor_points: int,
) -> InterpolationResult:
    ensure_columns(df, [x_column, y_column], Path("<csv>"))
    if anchor_width <= 0:
        raise SystemExit("--anchor-width must be > 0")
    if min_anchor_points < 2:
        raise SystemExit("--min-anchor-points must be >= 2")

    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    start, end = wave_range

    target_mask = np.isfinite(x) & (x >= start) & (x <= end)
    left_mask = np.isfinite(x) & (x >= start - anchor_width) & (x < start)
    right_mask = np.isfinite(x) & (x > end) & (x <= end + anchor_width)
    left_mask &= np.isfinite(y)
    right_mask &= np.isfinite(y)

    target_rows = int(target_mask.sum())
    left_rows = int(left_mask.sum())
    right_rows = int(right_mask.sum())
    if target_rows == 0:
        raise SystemExit(f"No rows found in target range: {start:g} ~ {end:g} cm-1")
    if left_rows < min_anchor_points or right_rows < min_anchor_points:
        raise SystemExit(
            "Not enough anchor points. "
            f"left={left_rows}, right={right_rows}, required={min_anchor_points}. "
            "Increase --anchor-width or lower --min-anchor-points."
        )

    x_anchor, y_anchor = build_unique_xy(
        x=np.concatenate([x[left_mask], x[right_mask]]),
        y=np.concatenate([y[left_mask], y[right_mask]]),
    )
    if len(x_anchor) < 4:
        raise SystemExit("Cubic spline needs at least 4 unique finite anchor points.")

    spline = CubicSpline(x_anchor, y_anchor, bc_type=bc_type, extrapolate=False)
    interpolated = spline(x[target_mask])
    if not np.all(np.isfinite(interpolated)):
        raise SystemExit(
            "Spline produced non-finite values. Check that the target range lies "
            "between the left and right anchor regions."
        )

    if output_column not in df.columns:
        df[output_column] = df[y_column]

    before = pd.to_numeric(df.loc[target_mask, y_column], errors="coerce").to_numpy(dtype=float)
    df.loc[target_mask, output_column] = interpolated
    delta = interpolated - before

    return InterpolationResult(
        output_column=output_column,
        start=float(start),
        end=float(end),
        target_rows=target_rows,
        left_anchor_rows=left_rows,
        right_anchor_rows=right_rows,
        max_abs_delta=float(np.nanmax(np.abs(delta))),
        first_before=float(before[0]),
        first_after=float(interpolated[0]),
    )


def apply_bspline_fit_region(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_column: str,
    wave_range: tuple[float, float],
    anchor_width: float,
    knots_every: float,
    min_anchor_points: int,
) -> InterpolationResult:
    ensure_columns(df, [x_column, y_column], Path("<csv>"))
    if anchor_width < 0:
        raise SystemExit("--anchor-width must be >= 0 when using --region")
    if min_anchor_points < 0:
        raise SystemExit("--min-anchor-points must be >= 0")

    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    start, end = wave_range

    target_mask = np.isfinite(x) & (x >= start) & (x <= end)
    fit_mask = target_mask.copy()
    if anchor_width > 0:
        left_mask = np.isfinite(x) & (x >= start - anchor_width) & (x < start)
        right_mask = np.isfinite(x) & (x > end) & (x <= end + anchor_width)
        fit_mask |= left_mask | right_mask
    else:
        left_mask = np.zeros_like(target_mask, dtype=bool)
        right_mask = np.zeros_like(target_mask, dtype=bool)

    finite_y = np.isfinite(y)
    left_mask &= finite_y
    right_mask &= finite_y
    fit_mask &= finite_y

    target_rows = int(target_mask.sum())
    left_rows = int(left_mask.sum())
    right_rows = int(right_mask.sum())
    if target_rows == 0:
        raise SystemExit(f"No rows found in target range: {start:g} ~ {end:g} cm-1")
    if anchor_width > 0:
        x_finite = x[np.isfinite(x)]
        data_min = float(np.nanmin(x_finite))
        data_max = float(np.nanmax(x_finite))
        needs_left_anchor = start > data_min
        needs_right_anchor = end < data_max
        if needs_left_anchor and left_rows < min_anchor_points:
            raise SystemExit(
                "Not enough left anchor points for regional fit. "
                f"left={left_rows}, required={min_anchor_points}. "
                "Increase --anchor-width, lower --min-anchor-points, or move the region start."
            )
        if needs_right_anchor and right_rows < min_anchor_points:
            raise SystemExit(
                "Not enough right anchor points for regional fit. "
                f"right={right_rows}, required={min_anchor_points}. "
                "Increase --anchor-width, lower --min-anchor-points, or move the region end."
            )

    x_fit, y_fit = build_unique_xy(x=x[fit_mask], y=y[fit_mask])
    if len(x_fit) < 8:
        raise SystemExit("Regional B-spline fit needs at least 8 unique finite fit points.")

    knots = make_internal_knots(float(x_fit[0]), float(x_fit[-1]), knots_every)
    knots = knots[(knots > x_fit[0]) & (knots < x_fit[-1])]
    if len(knots) >= len(x_fit) - 4:
        raise SystemExit(
            "Too many internal knots for this region. Increase KNOTS_EVERY "
            "or widen the region/anchor-width."
        )

    if len(knots) == 0:
        spline = CubicSpline(x_fit, y_fit, bc_type="natural", extrapolate=False)
        internal_knots = 0
    else:
        spline = LSQUnivariateSpline(x_fit, y_fit, t=knots, k=3)
        internal_knots = len(knots)

    interpolated = spline(x[target_mask])
    if not np.all(np.isfinite(interpolated)):
        raise SystemExit(
            "Regional B-spline produced non-finite values. Check the region and anchors."
        )

    if output_column not in df.columns:
        df[output_column] = df[y_column]

    before = pd.to_numeric(df.loc[target_mask, y_column], errors="coerce").to_numpy(dtype=float)
    df.loc[target_mask, output_column] = interpolated
    delta = interpolated - before

    return InterpolationResult(
        output_column=output_column,
        start=float(start),
        end=float(end),
        target_rows=target_rows,
        left_anchor_rows=left_rows,
        right_anchor_rows=right_rows,
        max_abs_delta=float(np.nanmax(np.abs(delta))),
        first_before=float(before[0]),
        first_after=float(interpolated[0]),
        mode="regional-bspline",
        knots_every=float(knots_every),
        internal_knots=internal_knots,
        fit_rows=len(x_fit),
    )


def update_derived_columns(
    df: pd.DataFrame,
    target_mask: np.ndarray,
    edited_column: str,
) -> None:
    if edited_column == "tau_fit_us":
        ensure_columns(df, ["tau_fit_us"], Path("<csv>"))
        tau = pd.to_numeric(df.loc[target_mask, "tau_fit_us"], errors="coerce").to_numpy(dtype=float)
        loss = TAU_US_TO_PPM_PER_CM / tau
        df.loc[target_mask, "loss_fit_ppm_per_cm"] = loss
    elif edited_column == "loss_fit_ppm_per_cm":
        ensure_columns(df, ["loss_fit_ppm_per_cm"], Path("<csv>"))
        loss = pd.to_numeric(
            df.loc[target_mask, "loss_fit_ppm_per_cm"],
            errors="coerce",
        ).to_numpy(dtype=float)
        tau = TAU_US_TO_PPM_PER_CM / loss
        df.loc[target_mask, "tau_fit_us"] = tau
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


def main() -> None:
    args = parse_args()
    load_dependencies()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV does not exist: {csv_path}")
    if args.anchor_width <= 0:
        raise SystemExit("--anchor-width must be > 0")
    if bool(args.range) == bool(args.region):
        raise SystemExit("Use either --range START END or one/more --region START END KNOTS_EVERY")

    output_column = (
        args.column
        if args.overwrite
        else args.output_column or f"{args.column}_cubic_spline"
    )
    if args.update_derived and not args.overwrite:
        raise SystemExit("--update-derived requires --overwrite")

    df = pd.read_csv(csv_path)
    ensure_columns(df, [args.x_column, args.column], csv_path)

    x = pd.to_numeric(df[args.x_column], errors="coerce").to_numpy(dtype=float)
    target_mask = np.zeros(len(df), dtype=bool)
    results: list[InterpolationResult] = []
    if args.region:
        source_df = df.copy(deep=True)
        if output_column != args.column:
            df[output_column] = source_df[args.column]
        for start, end, knots_every in args.region:
            wave_range = normalize_range(start, end)
            region_df = source_df.copy(deep=True)
            result = apply_bspline_fit_region(
                df=region_df,
                x_column=args.x_column,
                y_column=args.column,
                output_column=output_column,
                wave_range=wave_range,
                anchor_width=args.anchor_width,
                knots_every=knots_every,
                min_anchor_points=args.min_anchor_points,
            )
            results.append(result)
            region_mask = np.isfinite(x) & (x >= wave_range[0]) & (x <= wave_range[1])
            target_mask |= region_mask
            df.loc[region_mask, output_column] = region_df.loc[region_mask, output_column]
    else:
        wave_range = normalize_range(args.range[0], args.range[1])
        result = apply_spline_region(
            df=df,
            x_column=args.x_column,
            y_column=args.column,
            output_column=output_column,
            wave_range=wave_range,
            anchor_width=args.anchor_width,
            bc_type=args.bc_type,
            min_anchor_points=args.min_anchor_points,
        )
        results.append(result)
        target_mask |= np.isfinite(x) & (x >= wave_range[0]) & (x <= wave_range[1])

    if args.update_derived:
        update_derived_columns(df, target_mask=target_mask, edited_column=args.column)

    print(f"CSV: {csv_path}")
    print(f"Column: {args.column}")
    print(f"Output column: {output_column}")
    print(f"Anchor width: {args.anchor_width:g} cm-1")
    print(f"Boundary condition: {args.bc_type}")
    print(f"Regions: {len(results)}")
    for idx, result in enumerate(results, start=1):
        print(f"\nRegion {idx}: {result.start:g} ~ {result.end:g} cm-1")
        print(f"  Mode: {result.mode}")
        if result.knots_every is not None:
            print(f"  Knots every: {result.knots_every:g} cm-1")
            print(f"  Internal knots: {result.internal_knots}")
            print(f"  Fit points: {result.fit_rows}")
        print(f"  Target rows: {result.target_rows}")
        print(f"  Left/right anchor rows: {result.left_anchor_rows}/{result.right_anchor_rows}")
        print(f"  First target value: {result.first_before:.10g} -> {result.first_after:.10g}")
        print(f"  Max abs delta in target region: {result.max_abs_delta:.10g}")

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
