#!/usr/bin/env python3
"""Interpolate a processed CRDS CSV onto a uniform wavenumber grid.

使用说明
--------
这个脚本用于把已经处理好的结果 CSV/TXT 按固定波数间隔重新插值输出。
默认范围为 9120~9820 cm-1，默认步长为 0.01 cm-1。

默认会生成一个新的 CSV，不会修改原 CSV。
默认插值所有数值列，文本列/布尔列会按最近邻保留，例如 step2_fit_mode。
如果输入文件没有表头，脚本会把第 1 列当作波数，第 2 列当作衰荡时间 tau_us。

常用命令:
    # Ar 结果：插值到 9120~9820 cm-1，步长 0.01 cm-1
    python scripts/interpolate_processed_result_grid.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv

    # 无表头两列数据：第 1 列波数，第 2 列衰荡时间
    python scripts/interpolate_processed_result_grid.py \
      processed_tau.txt \
      --no-header

    # 指定输出文件
    python scripts/interpolate_processed_result_grid.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --output output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit_interp_0p01.csv

    # 只插值指定列
    python scripts/interpolate_processed_result_grid.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --columns tau_fit_us tau_fit_us_lmfit_spline loss_fit_ppm_per_cm

    # 如果输出文件已经存在，允许覆盖
    python scripts/interpolate_processed_result_grid.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --overwrite-output

参数说明:
    csv_path              要插值的处理结果 CSV。
    --start VALUE         输出起始波数，默认 9120。
    --end VALUE           输出结束波数，默认 9820。
    --step VALUE          输出波数步长，默认 0.01。
    --x-column NAME       波数列名，默认 wavenumber。
    --y-column NAME       无表头两列数据中第 2 列的输出列名，默认 tau_us。
    --no-header           输入文件没有表头；第 1 列为波数，第 2 列为衰荡时间。
    --two-column          即使有表头，也强制只使用前两列：第 1 列波数，第 2 列衰荡时间。
    --sep VALUE           输入分隔符；不加则 CSV 用逗号，TXT/其他文件自动兼容空格/逗号/Tab。
    --columns A B ...     只插值这些数值列；不加则自动插值所有数值列。
    --method METHOD       插值方法：pchip、linear、cubic，默认 pchip。
    --edge-fill MODE      网格略超出原数据范围时的边界填充方式：nearest 或 none。
                           默认 nearest，适合 9120 与首个实测点只差很小的情况。
    --duplicate-policy P  重复波数处理方式：mean、first、last，默认 mean。
    --output PATH         输出 CSV 路径；不加则自动生成。
    --overwrite-output    允许覆盖已有输出文件。
    --drop-metadata       不输出文本列/布尔列等非插值元数据。
    --plot PATH           可选，输出一张插值前后对比图。
    --plot-column NAME    画图使用的列；默认优先 tau_fit_us_lmfit_spline，再 tau_fit_us。
    --dry-run             只打印计划，不写 CSV。
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

np = None
pd = None
CubicSpline = None
PchipInterpolator = None
interp1d = None


@dataclass(frozen=True)
class InterpolationPlan:
    output_path: Path
    start: Decimal
    end: Decimal
    step: Decimal
    n_grid: int
    columns: list[str]
    metadata_columns: list[str]


def prepare_cache_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "crds-data-process-cache"
    mpl_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def load_dependencies(*, need_plot: bool = False):
    global CubicSpline, PchipInterpolator, interp1d, np, pd
    prepare_cache_env()
    if np is None or pd is None or CubicSpline is None or PchipInterpolator is None or interp1d is None:
        try:
            import numpy as numpy_module
            import pandas as pandas_module
            import scipy.interpolate as scipy_interpolate
        except ImportError as exc:
            raise SystemExit(
                "This script needs numpy, pandas, and scipy. Run it in the project "
                "environment, for example: conda run -n CRDS-Data-Process python "
                "scripts/interpolate_processed_result_grid.py ..."
            ) from exc
        np = numpy_module
        pd = pandas_module
        CubicSpline = scipy_interpolate.CubicSpline
        PchipInterpolator = scipy_interpolate.PchipInterpolator
        interp1d = scipy_interpolate.interp1d

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
        description="Interpolate a processed CRDS CSV onto a uniform wavenumber grid.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Processed CSV/TXT to interpolate.")
    parser.add_argument("--start", type=parse_decimal, default=Decimal("9120"), help="Start wavenumber. Default: 9120.")
    parser.add_argument("--end", type=parse_decimal, default=Decimal("9820"), help="End wavenumber. Default: 9820.")
    parser.add_argument("--step", type=parse_decimal, default=Decimal("0.01"), help="Grid step in cm-1. Default: 0.01.")
    parser.add_argument("--x-column", default="wavenumber", help="Wavenumber column. Default: wavenumber.")
    parser.add_argument(
        "--y-column",
        default="tau_us",
        help="Tau column name for headerless/two-column input. Default: tau_us.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Input has no header: first column is wavenumber, second column is tau.",
    )
    parser.add_argument(
        "--two-column",
        action="store_true",
        help="Force using only the first two columns as wavenumber and tau.",
    )
    parser.add_argument(
        "--sep",
        help=(
            "Input separator. Default: comma for .csv; otherwise a regex that "
            "accepts comma, spaces, or tabs."
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Numeric columns to interpolate. Default: all numeric non-boolean columns.",
    )
    parser.add_argument(
        "--method",
        choices=("pchip", "linear", "cubic"),
        default="pchip",
        help="Interpolation method. Default: pchip.",
    )
    parser.add_argument(
        "--edge-fill",
        choices=("nearest", "none"),
        default="nearest",
        help="How to fill grid points outside measured x range. Default: nearest.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("mean", "first", "last"),
        default="mean",
        help="How to collapse duplicate wavenumbers. Default: mean.",
    )
    parser.add_argument("--output", type=Path, help="Output CSV path. Default: auto-generated next to input.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing output CSV.",
    )
    parser.add_argument(
        "--drop-metadata",
        action="store_true",
        help="Do not carry non-interpolated metadata columns by nearest neighbor.",
    )
    parser.add_argument("--plot", type=Path, help="Optional comparison plot path.")
    parser.add_argument("--plot-column", help="Column to plot. Default: tau_fit_us_lmfit_spline, then tau_fit_us.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing output.")
    return parser.parse_args()


def is_decimal_like(value) -> bool:
    try:
        Decimal(str(value).strip())
    except InvalidOperation:
        return False
    return True


def input_separator(path: Path, explicit_sep: str | None) -> tuple[str, str]:
    if explicit_sep is not None:
        return explicit_sep, "python"
    if path.suffix.lower() == ".csv":
        return ",", "c"
    return r"[,\s]+", "python"


def read_input_table(args: argparse.Namespace):
    sep, engine = input_separator(args.csv_path, args.sep)
    header = None if args.no_header else "infer"
    df = pd.read_csv(args.csv_path, sep=sep, engine=engine, header=header)

    if not args.no_header and len(df.columns) >= 2:
        first_two_headers_are_numbers = all(is_decimal_like(col) for col in list(df.columns[:2]))
        if first_two_headers_are_numbers:
            print("Detected headerless numeric data; re-reading with --no-header behavior.")
            df = pd.read_csv(args.csv_path, sep=sep, engine=engine, header=None)
            args.no_header = True

    if args.no_header:
        rename = {}
        if len(df.columns) >= 1:
            rename[df.columns[0]] = args.x_column
        if len(df.columns) >= 2:
            rename[df.columns[1]] = args.y_column
        for index, col in enumerate(df.columns[2:], start=3):
            rename[col] = f"col_{index}"
        df = df.rename(columns=rename)

    if args.two_column:
        if len(df.columns) < 2:
            raise SystemExit("--two-column needs at least two input columns.")
        df = df.iloc[:, :2].copy()
        df.columns = [args.x_column, args.y_column]
        if not args.columns:
            args.columns = [args.y_column]

    return coerce_obvious_numeric_columns(df)


def coerce_obvious_numeric_columns(df):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        original_non_null = out[col].notna()
        if original_non_null.any() and converted[original_non_null].notna().all():
            out[col] = converted
    return out


def make_grid(start: Decimal, end: Decimal, step: Decimal):
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


def default_output_path(csv_path: Path, start: Decimal, end: Decimal, step: Decimal) -> Path:
    def clean(value: Decimal) -> str:
        text = format(value.normalize(), "f")
        return text.replace("-", "m").replace(".", "p")

    suffix = f"_interp_{clean(start)}_{clean(end)}_step{clean(step)}"
    output_suffix = ".csv" if csv_path.suffix.lower() != ".csv" else csv_path.suffix
    return csv_path.with_name(f"{csv_path.stem}{suffix}{output_suffix}")


def collapse_duplicates(df, x_column: str, duplicate_policy: str):
    if not df[x_column].duplicated().any():
        return df

    if duplicate_policy in {"first", "last"}:
        keep = "first" if duplicate_policy == "first" else "last"
        return df.drop_duplicates(subset=x_column, keep=keep).reset_index(drop=True)

    numeric_cols = [
        col for col in df.columns if col != x_column and pd.api.types.is_numeric_dtype(df[col])
    ]
    other_cols = [col for col in df.columns if col not in numeric_cols and col != x_column]
    agg = {col: "mean" for col in numeric_cols}
    agg.update({col: "first" for col in other_cols})
    return df.groupby(x_column, as_index=False, sort=True).agg(agg)


def select_interpolation_columns(df, x_column: str, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise SystemExit(f"Columns not found: {', '.join(missing)}")
        non_numeric = [
            col
            for col in requested
            if not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ]
        if non_numeric:
            raise SystemExit(f"Columns are not numeric interpolation columns: {', '.join(non_numeric)}")
        return [col for col in requested if col != x_column]

    columns = []
    for col in df.columns:
        if col == x_column:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return columns


def nearest_values(x_source, values, x_grid):
    idx = np.searchsorted(x_source, x_grid, side="left")
    idx = np.clip(idx, 0, len(x_source) - 1)
    left_idx = np.clip(idx - 1, 0, len(x_source) - 1)
    choose_left = np.abs(x_grid - x_source[left_idx]) <= np.abs(x_grid - x_source[idx])
    nearest_idx = np.where(choose_left, left_idx, idx)
    return np.asarray(values)[nearest_idx]


def interpolate_column(x_source, y_source, x_grid, *, method: str, edge_fill: str):
    valid = np.isfinite(x_source) & np.isfinite(y_source)
    x_valid = x_source[valid]
    y_valid = y_source[valid]
    if len(x_valid) < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    if method == "cubic" and len(x_valid) < 4:
        raise SystemExit("Cubic interpolation needs at least 4 valid points.")

    if method == "pchip":
        interpolator = PchipInterpolator(x_valid, y_valid, extrapolate=False)
        out = interpolator(x_grid)
    elif method == "cubic":
        interpolator = CubicSpline(x_valid, y_valid, extrapolate=False)
        out = interpolator(x_grid)
    else:
        interpolator = interp1d(
            x_valid,
            y_valid,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        out = interpolator(x_grid)

    if edge_fill == "nearest":
        out = np.asarray(out, dtype=float)
        out[x_grid < x_valid[0]] = y_valid[0]
        out[x_grid > x_valid[-1]] = y_valid[-1]
    return out


def build_interpolated_df(
    df,
    *,
    x_column: str,
    x_grid,
    interpolation_columns: list[str],
    keep_metadata: bool,
    method: str,
    edge_fill: str,
):
    x_source = df[x_column].to_numpy(dtype=float)
    output = {x_column: x_grid}

    for col in interpolation_columns:
        output[col] = interpolate_column(
            x_source,
            df[col].to_numpy(dtype=float),
            x_grid,
            method=method,
            edge_fill=edge_fill,
        )

    if keep_metadata:
        for col in df.columns:
            if col == x_column or col in interpolation_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                continue
            output[col] = nearest_values(x_source, df[col].to_numpy(), x_grid)

    return pd.DataFrame(output)


def choose_plot_column(columns: list[str], requested: str | None) -> str | None:
    if requested:
        return requested
    for candidate in ("tau_fit_us_lmfit_spline", "tau_fit_us", "loss_fit_ppm_per_cm"):
        if candidate in columns:
            return candidate
    return columns[0] if columns else None


def write_plot(original, interpolated, x_column: str, y_column: str, plot_path: Path) -> None:
    plt = load_dependencies(need_plot=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(original[x_column], original[y_column], ".", ms=2, alpha=0.35, label="original")
    ax.plot(interpolated[x_column], interpolated[y_column], "-", lw=1.2, label="interpolated")
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel(y_column)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_dependencies(need_plot=bool(args.plot))

    if not args.csv_path.exists():
        raise SystemExit(f"CSV not found: {args.csv_path}")
    df = read_input_table(args)
    if args.x_column not in df.columns:
        if len(df.columns) < 2:
            raise SystemExit(f"X column not found: {args.x_column}")
        print(
            f"X column '{args.x_column}' not found; using first two columns "
            f"as {args.x_column} and {args.y_column}."
        )
        df = df.iloc[:, :2].copy()
        df.columns = [args.x_column, args.y_column]
        if not args.columns:
            args.columns = [args.y_column]

    df = df.copy()
    df[args.x_column] = pd.to_numeric(df[args.x_column], errors="coerce")
    df = df[np.isfinite(df[args.x_column].to_numpy(dtype=float))]
    if df.empty:
        raise SystemExit(f"No valid x values in column: {args.x_column}")
    df = df.sort_values(args.x_column).reset_index(drop=True)
    df = collapse_duplicates(df, args.x_column, args.duplicate_policy)

    x_grid = make_grid(args.start, args.end, args.step)
    interpolation_columns = select_interpolation_columns(df, args.x_column, args.columns)
    if not interpolation_columns:
        raise SystemExit("No numeric columns selected for interpolation.")

    output_path = args.output or default_output_path(args.csv_path, args.start, args.end, args.step)
    metadata_columns = [
        col
        for col in df.columns
        if col != args.x_column
        and col not in interpolation_columns
        and (not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))
    ]
    if args.drop_metadata:
        metadata_columns = []

    plan = InterpolationPlan(
        output_path=output_path,
        start=args.start,
        end=args.end,
        step=args.step,
        n_grid=len(x_grid),
        columns=interpolation_columns,
        metadata_columns=metadata_columns,
    )

    print(f"Input CSV: {args.csv_path.resolve()}")
    print(f"Output CSV: {plan.output_path.resolve()}")
    print(f"Grid: {plan.start} ~ {plan.end} cm-1, step {plan.step} cm-1")
    print(f"Grid points: {plan.n_grid}")
    print(f"Input x coverage: {df[args.x_column].min():.8g} ~ {df[args.x_column].max():.8g} cm-1")
    print(f"Method: {args.method}")
    print(f"Edge fill: {args.edge_fill}")
    print(f"Interpolated columns ({len(plan.columns)}): {', '.join(plan.columns)}")
    if plan.metadata_columns:
        print(f"Metadata columns ({len(plan.metadata_columns)}): {', '.join(plan.metadata_columns)}")

    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to write the output CSV.")
        return

    if output_path.exists() and not args.overwrite_output:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite-output to replace it.")

    interpolated = build_interpolated_df(
        df,
        x_column=args.x_column,
        x_grid=x_grid,
        interpolation_columns=interpolation_columns,
        keep_metadata=not args.drop_metadata,
        method=args.method,
        edge_fill=args.edge_fill,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interpolated.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")

    if args.plot:
        plot_column = choose_plot_column(interpolation_columns, args.plot_column)
        if plot_column is None or plot_column not in interpolated.columns:
            raise SystemExit("No valid plot column found.")
        write_plot(df, interpolated, args.x_column, plot_column, args.plot)
        print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
