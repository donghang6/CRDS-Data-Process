#!/usr/bin/env python3
"""Apply lmfit cubic B-spline fitting to a processed CRDS CSV.

使用说明
--------
这个脚本用于对已经处理好的 CSV 做全局或分区域三次样条拟合。它不是把每个数据点
都当作样条节点，而是使用较少的 B-spline 节点，用 lmfit 最小化残差。
这样可以让曲线整体更平滑，同时避免完全追随局部小折点和噪声。

默认只 dry-run 预览，不会修改 CSV；确认无误后加 --apply 才会写回。
默认不覆盖原列，而是写入新列：{column}_lmfit_spline。

核心建议
--------
    - --knots-every 越小，节点越密，曲线越贴近数据，但更容易保留小折点。
    - --knots-every 越大，曲线越平滑，但可能压掉真实结构。
    - --smooth-lambda > 0 时会额外惩罚样条系数的二阶差分，让曲线更平滑。
    - 如果只追求残差最小，可以设 --smooth-lambda 0；如果想平滑，建议从
      0.01、0.1、1 逐步试。

常用命令:
    # 预览：全局拟合 tau_fit_us，结果写入新列 tau_fit_us_lmfit_spline
    conda run -n CRDS-Data-Process python scripts/apply_lmfit_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --column tau_fit_us \
      --knots-every 40 \
      --smooth-lambda 0.1 \
      --plot output/results/continuum/CIA/273K/Ar\\ 500Torr/lmfit_spline_preview.png

    # 执行：写入新列，不覆盖原 tau_fit_us
    conda run -n CRDS-Data-Process python scripts/apply_lmfit_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --column tau_fit_us \
      --knots-every 40 \
      --smooth-lambda 0.1 \
      --apply

    # 执行：直接覆盖 tau_fit_us，并同步更新 loss_fit_ppm_per_cm 和残差列
    conda run -n CRDS-Data-Process python scripts/apply_lmfit_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --column tau_fit_us \
      --knots-every 40 \
      --smooth-lambda 0.1 \
      --overwrite \
      --update-derived \
      --apply

    # 多区域：每个区域使用不同的 knots_every，独立 lmfit 后拼接到新列
    conda run -n CRDS-Data-Process python scripts/apply_lmfit_spline_region.py \
      output/results/continuum/CIA/273K/Ar\\ 500Torr/continuum_step2_fit.csv \
      --column tau_fit_us \
      --anchor-width 5 \
      --smooth-lambda 0.1 \
      --region 9100 9200 12 \
      --region 9200 9600 10 \
      --region 9600 9900 15 \
      --apply

参数说明:
    csv_path              要处理的 CSV。
    --column NAME         要拟合的列，默认 tau_fit_us。
    --x-column NAME       波数列，默认 wavenumber。
    --knots-every W       内部节点间隔，单位 cm-1，默认 40。
    --n-knots N           内部节点数量；若提供，则覆盖 --knots-every。
    --region A B K        多区域拟合。A/B 为只写回的目标范围，K 为该区域 knots_every。
    --anchor-width W      多区域拟合时，每段左右额外用于拟合的缓冲宽度，默认 5 cm-1。
    --smooth-lambda L     平滑惩罚强度，默认 0。
    --fit-range A B       可选，只用指定波数范围参与全局拟合。
    --weights-column NAME 可选，权重列。若为 tau_stats_us，权重约为 1/sigma。
    --output-column NAME  不覆盖时写入的新列名；默认 {column}_lmfit_spline。
    --overwrite           直接把拟合结果写回 --column。
    --update-derived      覆盖 tau_fit_us 或 loss_fit_ppm_per_cm 时同步更新成对列和残差列。
    --plot PATH           输出拟合对比图。
    --no-backup           写回 CSV 时不生成 .bak 备份。
    --apply               真正写回 CSV；不加时只预览。
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

C_CM_PER_S = 2.99792458e10
TAU_US_TO_PPM_PER_CM = 1e12 / C_CM_PER_S
np = None
pd = None
BSpline = None
Minimizer = None
Parameters = None


@dataclass(frozen=True)
class FitResult:
    output_column: str
    n_fit_points: int
    n_coefficients: int
    n_internal_knots: int
    rmse: float
    max_abs_residual: float
    first_before: float
    first_after: float
    success: bool
    message: str
    start: float | None = None
    end: float | None = None
    knots_every: float | None = None
    fit_start: float | None = None
    fit_end: float | None = None


def prepare_cache_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "crds-data-process-cache"
    mpl_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def load_dependencies(*, need_plot: bool = False):
    global BSpline, Minimizer, Parameters, np, pd
    prepare_cache_env()
    if np is None or pd is None or BSpline is None or Minimizer is None or Parameters is None:
        try:
            import lmfit as lmfit_module
            import numpy as numpy_module
            import pandas as pandas_module
            import scipy.interpolate as scipy_interpolate
        except ImportError as exc:
            raise SystemExit(
                "This script needs numpy, pandas, scipy, and lmfit. Run it in the "
                "project environment, for example: conda run -n CRDS-Data-Process "
                "python scripts/apply_lmfit_spline_region.py ..."
            ) from exc
        np = numpy_module
        pd = pandas_module
        BSpline = scipy_interpolate.BSpline
        Minimizer = lmfit_module.Minimizer
        Parameters = lmfit_module.Parameters

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
        description="Fit a global cubic B-spline to a processed CSV with lmfit.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Processed CSV to edit.")
    parser.add_argument(
        "--column",
        default="tau_fit_us",
        help="Column to fit. Default: tau_fit_us.",
    )
    parser.add_argument(
        "--x-column",
        default="wavenumber",
        help="X/wavenumber column. Default: wavenumber.",
    )
    parser.add_argument(
        "--knots-every",
        type=float,
        default=40.0,
        help="Internal knot spacing in cm-1. Default: 40.",
    )
    parser.add_argument(
        "--n-knots",
        type=int,
        help="Number of internal knots. Overrides --knots-every.",
    )
    parser.add_argument(
        "--region",
        action="append",
        nargs=3,
        metavar=("START", "END", "KNOTS_EVERY"),
        type=float,
        help=(
            "Regional lmfit B-spline fit with per-region knot spacing. "
            "Repeat this option for multiple regions."
        ),
    )
    parser.add_argument(
        "--anchor-width",
        type=float,
        default=5.0,
        help="Extra fit width on each side of every --region, in cm-1. Default: 5.",
    )
    parser.add_argument(
        "--smooth-lambda",
        type=float,
        default=0.0,
        help="Penalty strength for second differences of spline coefficients. Default: 0.",
    )
    parser.add_argument(
        "--fit-range",
        nargs=2,
        metavar=("START", "END"),
        type=float,
        help="Optional wavenumber range used for fitting.",
    )
    parser.add_argument(
        "--weights-column",
        help="Optional uncertainty column; residuals are weighted by 1/value.",
    )
    parser.add_argument(
        "--output-column",
        help="New column for fitted values. Default: <column>_lmfit_spline.",
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
            "fit column and residual columns."
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional PNG path for original-vs-global-spline comparison.",
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


def make_internal_knots(x_min: float, x_max: float, knots_every: float, n_knots: int | None):
    if n_knots is not None:
        if n_knots < 0:
            raise SystemExit("--n-knots must be >= 0")
        if n_knots == 0:
            return np.array([], dtype=float)
        return np.linspace(x_min, x_max, n_knots + 2, dtype=float)[1:-1]

    if knots_every <= 0:
        raise SystemExit("--knots-every must be > 0")
    return np.arange(x_min + knots_every, x_max, knots_every, dtype=float)


def build_bspline_basis(x, knots, degree: int = 3):
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    knot_vector = np.r_[
        np.repeat(x_min, degree + 1),
        knots,
        np.repeat(x_max, degree + 1),
    ]
    n_coefficients = len(knot_vector) - degree - 1
    basis = np.empty((len(x), n_coefficients), dtype=float)
    for idx in range(n_coefficients):
        coeff = np.zeros(n_coefficients, dtype=float)
        coeff[idx] = 1.0
        basis[:, idx] = BSpline(knot_vector, coeff, degree, extrapolate=True)(x)
    return basis


def initial_coefficients(basis, y, weights):
    weighted_basis = basis * weights[:, None]
    weighted_y = y * weights
    coeff, *_ = np.linalg.lstsq(weighted_basis, weighted_y, rcond=None)
    return coeff


def params_from_coefficients(coefficients):
    params = Parameters()
    for idx, value in enumerate(coefficients):
        params.add(f"c{idx}", value=float(value))
    return params


def coefficients_from_params(params):
    return np.array([params[f"c{idx}"].value for idx in range(len(params))], dtype=float)


def residual_function(params, basis, y, weights, smooth_lambda: float):
    coeff = coefficients_from_params(params)
    residual = (basis @ coeff - y) * weights
    if smooth_lambda > 0 and len(coeff) >= 3:
        penalty = np.sqrt(smooth_lambda) * np.diff(coeff, n=2)
        residual = np.r_[residual, penalty]
    return residual


def build_fit_mask(df, x_column: str, y_column: str, fit_range: tuple[float, float] | None):
    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if fit_range is not None:
        mask &= (x >= fit_range[0]) & (x <= fit_range[1])
    return x, y, mask


def build_weights(df, weights_column: str | None, mask):
    if not weights_column:
        return np.ones(int(mask.sum()), dtype=float)
    ensure_columns(df, [weights_column], Path("<csv>"))
    sigma = pd.to_numeric(df.loc[mask, weights_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(sigma) & (sigma > 0)
    if int(finite.sum()) < 2:
        raise SystemExit(f"No usable positive finite weights in column: {weights_column}")
    fallback = float(np.nanmedian(sigma[finite]))
    sigma = np.where(finite, sigma, fallback)
    return 1.0 / sigma


def fit_global_spline(
    df,
    x_column: str,
    y_column: str,
    output_column: str,
    knots_every: float,
    n_knots: int | None,
    smooth_lambda: float,
    fit_range: tuple[float, float] | None,
    weights_column: str | None,
) -> FitResult:
    ensure_columns(df, [x_column, y_column], Path("<csv>"))
    if smooth_lambda < 0:
        raise SystemExit("--smooth-lambda must be >= 0")

    x_all = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    x, y, fit_mask = build_fit_mask(df, x_column, y_column, fit_range)
    if int(fit_mask.sum()) < 8:
        raise SystemExit("Need at least 8 finite fit points for global cubic spline.")

    fit_df = (
        pd.DataFrame({"x": x[fit_mask], "y": y[fit_mask]})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
    )
    x_fit = fit_df["x"].to_numpy(dtype=float)
    y_fit = fit_df["y"].to_numpy(dtype=float)
    if len(x_fit) < 8:
        raise SystemExit("Need at least 8 unique finite x points for global cubic spline.")

    x_min = float(x_fit[0])
    x_max = float(x_fit[-1])
    knots = make_internal_knots(
        x_min=x_min,
        x_max=x_max,
        knots_every=knots_every,
        n_knots=n_knots,
    )
    if len(knots) >= len(x_fit) - 4:
        raise SystemExit(
            "Too many knots for the number of fit points. Increase --knots-every "
            "or reduce --n-knots."
        )

    basis = build_bspline_basis(x_fit, knots=knots, degree=3)
    if weights_column:
        # Rebuild weights after grouping by x; mean uncertainty is adequate for duplicate x.
        weight_df = pd.DataFrame({
            "x": x[fit_mask],
            "weight": build_weights(df, weights_column, fit_mask),
        })
        weights = (
            weight_df.groupby("x", as_index=False)["weight"]
            .mean()
            .sort_values("x")["weight"]
            .to_numpy(dtype=float)
        )
    else:
        weights = np.ones(len(x_fit), dtype=float)

    init_coeff = initial_coefficients(basis=basis, y=y_fit, weights=weights)
    params = params_from_coefficients(init_coeff)
    minner = Minimizer(
        residual_function,
        params,
        fcn_args=(basis, y_fit, weights, smooth_lambda),
    )
    result = minner.minimize(method="least_squares")
    coeff = coefficients_from_params(result.params)
    y_fit_model = basis @ coeff
    residual = y_fit - y_fit_model

    output_values = np.full(len(df), np.nan, dtype=float)
    valid_x = np.isfinite(x_all)
    basis_all = build_bspline_basis(x_all[valid_x], knots=knots, degree=3)
    output_values[valid_x] = basis_all @ coeff

    if output_column not in df.columns:
        df[output_column] = df[y_column]
    df.loc[valid_x, output_column] = output_values[valid_x]

    first_idx = int(np.where(valid_x)[0][0])
    return FitResult(
        output_column=output_column,
        n_fit_points=len(x_fit),
        n_coefficients=len(coeff),
        n_internal_knots=len(knots),
        rmse=float(np.sqrt(np.nanmean(residual**2))),
        max_abs_residual=float(np.nanmax(np.abs(residual))),
        first_before=float(pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)[first_idx]),
        first_after=float(output_values[first_idx]),
        success=bool(result.success),
        message=str(result.message),
    )


def fit_regional_splines(
    df,
    x_column: str,
    y_column: str,
    output_column: str,
    regions: list[list[float]],
    anchor_width: float,
    smooth_lambda: float,
    weights_column: str | None,
) -> tuple[list[FitResult], np.ndarray]:
    if anchor_width < 0:
        raise SystemExit("--anchor-width must be >= 0")

    source_df = df.copy(deep=True)
    if output_column != y_column:
        df[output_column] = source_df[y_column]

    x = pd.to_numeric(source_df[x_column], errors="coerce").to_numpy(dtype=float)
    finite_x = x[np.isfinite(x)]
    if len(finite_x) == 0:
        raise SystemExit(f"No finite values in x column: {x_column}")
    data_min = float(np.nanmin(finite_x))
    data_max = float(np.nanmax(finite_x))

    updated_mask = np.zeros(len(df), dtype=bool)
    results: list[FitResult] = []
    for start_raw, end_raw, knots_every in regions:
        start, end = normalize_range(start_raw, end_raw)
        target_mask = np.isfinite(x) & (x >= start) & (x <= end)
        if int(target_mask.sum()) == 0:
            raise SystemExit(f"No rows found in region: {start:g} ~ {end:g} cm-1")

        fit_start = max(data_min, start - anchor_width)
        fit_end = min(data_max, end + anchor_width)
        region_df = source_df.copy(deep=True)
        region_output = f"{output_column}__region_tmp"
        raw_result = fit_global_spline(
            df=region_df,
            x_column=x_column,
            y_column=y_column,
            output_column=region_output,
            knots_every=float(knots_every),
            n_knots=None,
            smooth_lambda=smooth_lambda,
            fit_range=(fit_start, fit_end),
            weights_column=weights_column,
        )
        df.loc[target_mask, output_column] = region_df.loc[target_mask, region_output]
        updated_mask |= target_mask

        before = pd.to_numeric(
            source_df.loc[target_mask, y_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        after = pd.to_numeric(
            region_df.loc[target_mask, region_output],
            errors="coerce",
        ).to_numpy(dtype=float)
        results.append(
            FitResult(
                output_column=output_column,
                n_fit_points=raw_result.n_fit_points,
                n_coefficients=raw_result.n_coefficients,
                n_internal_knots=raw_result.n_internal_knots,
                rmse=raw_result.rmse,
                max_abs_residual=raw_result.max_abs_residual,
                first_before=float(before[0]),
                first_after=float(after[0]),
                success=raw_result.success,
                message=raw_result.message,
                start=float(start),
                end=float(end),
                knots_every=float(knots_every),
                fit_start=float(fit_start),
                fit_end=float(fit_end),
            )
        )
    return results, updated_mask


def update_derived_columns(df, edited_column: str, target_mask=None) -> None:
    mask = np.ones(len(df), dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool).copy()
    if edited_column == "tau_fit_us":
        tau = pd.to_numeric(df["tau_fit_us"], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(tau) & (tau != 0)
        df.loc[mask, "loss_fit_ppm_per_cm"] = TAU_US_TO_PPM_PER_CM / tau[mask]
    elif edited_column == "loss_fit_ppm_per_cm":
        loss = pd.to_numeric(df["loss_fit_ppm_per_cm"], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(loss) & (loss != 0)
        df.loc[mask, "tau_fit_us"] = TAU_US_TO_PPM_PER_CM / loss[mask]
    else:
        raise SystemExit(
            "--update-derived only supports --column tau_fit_us or "
            "--column loss_fit_ppm_per_cm"
        )

    if {"loss_ppm_per_cm", "loss_fit_ppm_per_cm"}.issubset(df.columns):
        df.loc[mask, "loss_residual_ppm_per_cm"] = (
            pd.to_numeric(df.loc[mask, "loss_ppm_per_cm"], errors="coerce")
            - pd.to_numeric(df.loc[mask, "loss_fit_ppm_per_cm"], errors="coerce")
        )
    if {"tau_us", "tau_fit_us"}.issubset(df.columns):
        df.loc[mask, "tau_residual_us"] = (
            pd.to_numeric(df.loc[mask, "tau_us"], errors="coerce")
            - pd.to_numeric(df.loc[mask, "tau_fit_us"], errors="coerce")
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


def save_plot(df, x_column: str, source_column: str, output_column: str, plot_path: Path) -> None:
    plt = load_dependencies(need_plot=True)
    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    source = pd.to_numeric(df[source_column], errors="coerce").to_numpy(dtype=float)
    output = pd.to_numeric(df[output_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(source) & np.isfinite(output)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    ax.plot(x[mask], source[mask], "-", lw=0.9, color="tab:red", label=source_column)
    ax.plot(x[mask], output[mask], "-", lw=1.2, color="tab:blue", label=output_column)
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
    if args.region and args.fit_range:
        raise SystemExit("--region cannot be combined with --fit-range")
    if args.region and args.n_knots is not None:
        raise SystemExit("--region uses per-region KNOTS_EVERY; do not use --n-knots")

    output_column = (
        args.column
        if args.overwrite else args.output_column or f"{args.column}_lmfit_spline"
    )
    fit_range = normalize_range(args.fit_range[0], args.fit_range[1]) if args.fit_range else None

    df = pd.read_csv(csv_path)
    ensure_columns(df, [args.x_column, args.column], csv_path)
    source_column_for_plot = args.column
    if args.overwrite and args.plot:
        source_column_for_plot = f"{args.column}__before_lmfit_spline"
        df[source_column_for_plot] = df[args.column]

    updated_mask = None
    if args.region:
        results, updated_mask = fit_regional_splines(
            df=df,
            x_column=args.x_column,
            y_column=args.column,
            output_column=output_column,
            regions=args.region,
            anchor_width=args.anchor_width,
            smooth_lambda=args.smooth_lambda,
            weights_column=args.weights_column,
        )
    else:
        results = [
            fit_global_spline(
                df=df,
                x_column=args.x_column,
                y_column=args.column,
                output_column=output_column,
                knots_every=args.knots_every,
                n_knots=args.n_knots,
                smooth_lambda=args.smooth_lambda,
                fit_range=fit_range,
                weights_column=args.weights_column,
            )
        ]
    if args.update_derived:
        update_derived_columns(df, edited_column=args.column, target_mask=updated_mask)
    if args.plot:
        save_plot(
            df=df,
            x_column=args.x_column,
            source_column=source_column_for_plot,
            output_column=output_column,
            plot_path=args.plot.expanduser().resolve(),
        )
    if source_column_for_plot != args.column:
        df = df.drop(columns=[source_column_for_plot])

    print(f"CSV: {csv_path}")
    print(f"Column: {args.column}")
    print(f"Output column: {output_column}")
    print(f"Smooth lambda: {args.smooth_lambda:g}")
    if args.region:
        print(f"Anchor width: {args.anchor_width:g} cm-1")
        print(f"Regions: {len(results)}")
    else:
        if fit_range is None:
            print("Fit range: all finite data")
        else:
            print(f"Fit range: {fit_range[0]:g} ~ {fit_range[1]:g} cm-1")

    for idx, result in enumerate(results, start=1):
        if result.start is not None and result.end is not None:
            print(f"\nRegion {idx}: {result.start:g} ~ {result.end:g} cm-1")
            print(f"  Fit range: {result.fit_start:g} ~ {result.fit_end:g} cm-1")
            print(f"  Knots every: {result.knots_every:g} cm-1")
        print(f"  Internal knots: {result.n_internal_knots}")
        print(f"  Spline coefficients: {result.n_coefficients}")
        print(f"  Fit points: {result.n_fit_points}")
        print(f"  RMSE: {result.rmse:.10g}")
        print(f"  Max abs residual: {result.max_abs_residual:.10g}")
        print(f"  First value: {result.first_before:.10g} -> {result.first_after:.10g}")
        print(f"  lmfit success: {result.success}")
        print(f"  lmfit message: {result.message}")
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
