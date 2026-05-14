#!/usr/bin/env python3
"""Calculate dB/dT and uncertainty from a B summary table.

The input table is expected to contain B and U(B) columns like:

    波数, B 273, U B 273,
    B 303 500, U B 303 500,
    B 303 600, U B 303 600,
    B 303 700, U B 303 700,
    B 333 500, U B 333 500

For each wavenumber, all available B values are fitted with weighted linear
least squares:

    B(nu, T) = a(nu) + b(nu) T

where b(nu) = dB/dT.  The standard uncertainty of b is obtained from the
weighted least-squares covariance matrix.  To avoid underestimating the
uncertainty when the repeated 303 K pressure groups disagree beyond their
reported U(B), the final uncertainty is multiplied by a Birge factor:

    Birge = max(1, sqrt(chi2 / dof))

The output also includes the inverse-variance weighted 303 K B value for
diagnostic/plotting purposes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/summary.txt")
DEFAULT_OUTPUT = Path("output/results/analysis/B_temperature_dependence_from_summary")
DEFAULT_SELECTED = (9200.0, 9323.0, 9420.0, 9520.0, 9800.0)


MEASUREMENTS = [
    ("273K_500Torr", 273.0, "B 273", "U B 273"),
    ("303K_500Torr", 303.0, "B 303 500", "U B 303 500"),
    ("303K_600Torr", 303.0, "B 303 600", "U B 303 600"),
    ("303K_700Torr", 303.0, "B 303 700", "U B 303 700"),
    ("333K_500Torr", 333.0, "B 333 500", "U B 333 500"),
]


COLUMN_DESCRIPTIONS_ZH = {
    "wavenumber": "波数，单位 cm^-1。",
    "B_273K_500Torr": "273 K、500 Torr 条件下的 O2-O2 二元碰撞吸收系数 B，单位 cm^-1 amagat^-2。",
    "u_B_273K_500Torr": "273 K、500 Torr 的 B 标准不确定度，单位 cm^-1 amagat^-2。",
    "B_303K_500Torr": "303 K、500 Torr 条件下的 O2-O2 二元碰撞吸收系数 B，单位 cm^-1 amagat^-2。",
    "u_B_303K_500Torr": "303 K、500 Torr 的 B 标准不确定度，单位 cm^-1 amagat^-2。",
    "B_303K_600Torr": "303 K、600 Torr 条件下的 O2-O2 二元碰撞吸收系数 B，单位 cm^-1 amagat^-2。",
    "u_B_303K_600Torr": "303 K、600 Torr 的 B 标准不确定度，单位 cm^-1 amagat^-2。",
    "B_303K_700Torr": "303 K、700 Torr 条件下的 O2-O2 二元碰撞吸收系数 B，单位 cm^-1 amagat^-2。",
    "u_B_303K_700Torr": "303 K、700 Torr 的 B 标准不确定度，单位 cm^-1 amagat^-2。",
    "B_303K_weighted": "303 K 三个压力点按 1/u_B^2 加权平均后的 B，单位 cm^-1 amagat^-2。",
    "u_B_303K_weighted_internal": "303 K 加权平均的内部标准不确定度，即 sqrt(1/sum(w_i))。",
    "u_B_303K_weighted_scaled": "303 K 加权平均经压力组 Birge ratio 放大后的标准不确定度。",
    "B_303K_pressure_reduced_chi2": "303 K 三个压力点相对加权平均值的约化卡方，用于判断压力组离散程度。",
    "B_333K_500Torr": "333 K、500 Torr 条件下的 O2-O2 二元碰撞吸收系数 B，单位 cm^-1 amagat^-2。",
    "u_B_333K_500Torr": "333 K、500 Torr 的 B 标准不确定度，单位 cm^-1 amagat^-2。",
    "fit_intercept": "温度线性拟合 B(T)=a+bT 的截距 a。",
    "u_fit_intercept_unscaled": "由加权线性拟合协方差矩阵得到的截距内部标准不确定度，未乘 Birge ratio。",
    "u_fit_intercept": "截距最终标准不确定度，已乘温度拟合 Birge ratio。",
    "dB_dT_cm_inv_amagat_neg2_K_neg1": "温度依赖系数 dB/dT，即 B 对温度的线性拟合斜率，单位 cm^-1 amagat^-2 K^-1。",
    "u_dB_dT_unscaled": "dB/dT 的内部标准不确定度，来自加权线性拟合协方差矩阵，未乘 Birge ratio。",
    "u_dB_dT_cm_inv_amagat_neg2_K_neg1": "dB/dT 的最终标准不确定度，已乘温度拟合 Birge ratio，单位 cm^-1 amagat^-2 K^-1。",
    "fit_chi2": "温度线性拟合的卡方 chi^2。",
    "fit_reduced_chi2": "温度线性拟合的约化卡方 chi^2_red = chi^2 / dof。",
    "fit_birge_ratio": "温度线性拟合的 Birge ratio，等于 max(1, sqrt(chi^2_red))。",
    "n_fit_points": "参与该波数点温度线性拟合的数据点数量。",
    "fit_dof": "温度线性拟合自由度，等于 n_fit_points - 2。",
    "fit_mode": "温度依赖系数的拟合模式说明。",
    "fit_temperature_points": "参与拟合的温度点说明。",
    "fit_303_mode": "303 K 数据处理方式：separate 表示三组压力分别参与拟合，combined 表示先合成一个 303 K 点。",
    "combined_303_uncertainty": "303 K 合成点使用的不确定度类型：scaled 为经 Birge ratio 放大，internal 为内部不确定度。",
    "input_uncertainty_scale": "输入 B 标准不确定度的统一放大倍数；例如 2 表示所有 u_B 在计算前均乘以 2。",
    "B_fit_273K": "温度线性拟合在 273 K 处的拟合 B 值。",
    "B_fit_303K": "温度线性拟合在 303 K 处的拟合 B 值。",
    "B_fit_333K": "温度线性拟合在 333 K 处的拟合 B 值。",
    "B_residual_273K_500Torr": "273 K、500 Torr 实测 B 与拟合 B 的残差。",
    "B_residual_303K_500Torr": "303 K、500 Torr 实测 B 与拟合 B 的残差。",
    "B_residual_303K_600Torr": "303 K、600 Torr 实测 B 与拟合 B 的残差。",
    "B_residual_303K_700Torr": "303 K、700 Torr 实测 B 与拟合 B 的残差。",
    "B_residual_333K_500Torr": "333 K、500 Torr 实测 B 与拟合 B 的残差。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate weighted dB/dT and uncertainty from summary.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input summary table.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--encoding", default="gbk", help="Input text encoding.")
    parser.add_argument(
        "--fit-303-mode",
        choices=("separate", "combined"),
        default="separate",
        help=(
            "How to use the three 303 K pressure groups. 'separate' fits all "
            "three pressure points independently; 'combined' first combines "
            "them into one inverse-variance weighted 303 K point."
        ),
    )
    parser.add_argument(
        "--combined-303-uncertainty",
        choices=("scaled", "internal"),
        default="scaled",
        help=(
            "Uncertainty used for the combined 303 K point. 'scaled' applies "
            "the 303 K pressure-group Birge ratio; 'internal' uses only the "
            "inverse-variance weighted internal uncertainty."
        ),
    )
    parser.add_argument(
        "--input-uncertainty-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to every input U(B) before weighted averaging and fitting.",
    )
    parser.add_argument("--selected", type=float, nargs="+", default=list(DEFAULT_SELECTED))
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def read_summary(path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path.expanduser(), sep="\t", encoding=encoding, header=0, skiprows=[1])
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "wavenumber"})
    required = ["wavenumber"] + [col for _, _, b_col, u_col in MEASUREMENTS for col in (b_col, u_col)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["wavenumber"]).sort_values("wavenumber").reset_index(drop=True)


def weighted_mean(values: np.ndarray, uncertainties: np.ndarray) -> tuple[float, float, float, float]:
    mask = np.isfinite(values) & np.isfinite(uncertainties) & (uncertainties > 0)
    if int(mask.sum()) == 0:
        return np.nan, np.nan, np.nan, np.nan
    values = values[mask]
    uncertainties = uncertainties[mask]
    weights = 1.0 / uncertainties**2
    mean = float(np.sum(weights * values) / np.sum(weights))
    u_internal = float(np.sqrt(1.0 / np.sum(weights)))
    if len(values) > 1:
        chi2 = float(np.sum(weights * (values - mean) ** 2))
        red_chi2 = chi2 / (len(values) - 1)
        birge = max(1.0, float(np.sqrt(red_chi2)))
    else:
        red_chi2 = np.nan
        birge = 1.0
    return mean, u_internal, u_internal * birge, red_chi2


def weighted_linear_fit(
    temperatures: np.ndarray,
    values: np.ndarray,
    uncertainties: np.ndarray,
) -> dict:
    mask = (
        np.isfinite(temperatures)
        & np.isfinite(values)
        & np.isfinite(uncertainties)
        & (uncertainties > 0)
    )
    t = temperatures[mask]
    y = values[mask]
    u = uncertainties[mask]
    n = len(y)
    if n < 2 or len(np.unique(t)) < 2:
        return {
            "fit_intercept": np.nan,
            "u_fit_intercept_unscaled": np.nan,
            "u_fit_intercept": np.nan,
            "dB_dT_cm_inv_amagat_neg2_K_neg1": np.nan,
            "u_dB_dT_unscaled": np.nan,
            "u_dB_dT_cm_inv_amagat_neg2_K_neg1": np.nan,
            "fit_chi2": np.nan,
            "fit_reduced_chi2": np.nan,
            "fit_birge_ratio": np.nan,
            "n_fit_points": n,
            "fit_dof": 0,
        }
    x = np.column_stack([np.ones(n), t])
    w = 1.0 / u**2
    xtwx = x.T @ (w[:, None] * x)
    xtwy = x.T @ (w * y)
    cov = np.linalg.inv(xtwx)
    beta = cov @ xtwy
    fitted = x @ beta
    residual = y - fitted
    chi2 = float(np.sum((residual / u) ** 2))
    dof = max(n - 2, 0)
    reduced_chi2 = chi2 / dof if dof else np.nan
    birge = max(1.0, float(np.sqrt(reduced_chi2))) if dof else 1.0
    u_intercept_unscaled = float(np.sqrt(cov[0, 0]))
    u_slope_unscaled = float(np.sqrt(cov[1, 1]))
    return {
        "fit_intercept": float(beta[0]),
        "u_fit_intercept_unscaled": u_intercept_unscaled,
        "u_fit_intercept": u_intercept_unscaled * birge,
        "dB_dT_cm_inv_amagat_neg2_K_neg1": float(beta[1]),
        "u_dB_dT_unscaled": u_slope_unscaled,
        "u_dB_dT_cm_inv_amagat_neg2_K_neg1": u_slope_unscaled * birge,
        "fit_chi2": chi2,
        "fit_reduced_chi2": reduced_chi2,
        "fit_birge_ratio": birge,
        "n_fit_points": n,
        "fit_dof": dof,
    }


def calculate(
    df: pd.DataFrame,
    fit_303_mode: str,
    combined_303_uncertainty: str,
    input_uncertainty_scale: float,
) -> pd.DataFrame:
    if input_uncertainty_scale <= 0:
        raise ValueError("--input-uncertainty-scale must be positive")
    records: list[dict] = []
    for _, row in df.iterrows():
        u273 = float(row["U B 273"]) * input_uncertainty_scale
        u333 = float(row["U B 333 500"]) * input_uncertainty_scale
        b303 = np.asarray([row["B 303 500"], row["B 303 600"], row["B 303 700"]], dtype=float)
        u303 = (
            np.asarray([row["U B 303 500"], row["U B 303 600"], row["U B 303 700"]], dtype=float)
            * input_uncertainty_scale
        )
        b303_mean, b303_u_internal, b303_u_scaled, b303_reduced_chi2 = weighted_mean(b303, u303)
        if fit_303_mode == "combined":
            temperatures = np.asarray([273.0, 303.0, 333.0], dtype=float)
            values = np.asarray([row["B 273"], b303_mean, row["B 333 500"]], dtype=float)
            u303_fit = b303_u_scaled if combined_303_uncertainty == "scaled" else b303_u_internal
            uncertainties = np.asarray([u273, u303_fit, u333], dtype=float)
            fit_mode_label = f"weighted_linear_combined_303_{combined_303_uncertainty}_uncertainty_birge_scaled"
            fit_temperature_points = "273,303_combined,333"
        else:
            temperatures = np.asarray([temperature for _, temperature, _, _ in MEASUREMENTS], dtype=float)
            values = np.asarray([row[b_col] for _, _, b_col, _ in MEASUREMENTS], dtype=float)
            uncertainties = (
                np.asarray([row[u_col] for _, _, _, u_col in MEASUREMENTS], dtype=float)
                * input_uncertainty_scale
            )
            fit_mode_label = "weighted_linear_all_pressure_points_birge_scaled"
            fit_temperature_points = "273,303,303,303,333"
        fit = weighted_linear_fit(temperatures, values, uncertainties)
        rec = {
            "wavenumber": row["wavenumber"],
            "B_273K_500Torr": row["B 273"],
            "u_B_273K_500Torr": u273,
            "B_303K_500Torr": row["B 303 500"],
            "u_B_303K_500Torr": u303[0],
            "B_303K_600Torr": row["B 303 600"],
            "u_B_303K_600Torr": u303[1],
            "B_303K_700Torr": row["B 303 700"],
            "u_B_303K_700Torr": u303[2],
            "B_303K_weighted": b303_mean,
            "u_B_303K_weighted_internal": b303_u_internal,
            "u_B_303K_weighted_scaled": b303_u_scaled,
            "B_303K_pressure_reduced_chi2": b303_reduced_chi2,
            "B_333K_500Torr": row["B 333 500"],
            "u_B_333K_500Torr": u333,
            **fit,
            "fit_mode": fit_mode_label,
            "fit_temperature_points": fit_temperature_points,
            "fit_303_mode": fit_303_mode,
            "combined_303_uncertainty": combined_303_uncertainty if fit_303_mode == "combined" else "",
            "input_uncertainty_scale": input_uncertainty_scale,
        }
        for temperature in (273.0, 303.0, 333.0):
            rec[f"B_fit_{temperature:.0f}K"] = rec["fit_intercept"] + rec[
                "dB_dT_cm_inv_amagat_neg2_K_neg1"
            ] * temperature
        rec["B_residual_273K_500Torr"] = rec["B_273K_500Torr"] - rec["B_fit_273K"]
        rec["B_residual_303K_500Torr"] = rec["B_303K_500Torr"] - rec["B_fit_303K"]
        rec["B_residual_303K_600Torr"] = rec["B_303K_600Torr"] - rec["B_fit_303K"]
        rec["B_residual_303K_700Torr"] = rec["B_303K_700Torr"] - rec["B_fit_303K"]
        rec["B_residual_333K_500Torr"] = rec["B_333K_500Torr"] - rec["B_fit_333K"]
        records.append(rec)
    return pd.DataFrame(records)


def nearest_rows(df: pd.DataFrame, selected: list[float]) -> pd.DataFrame:
    x = df["wavenumber"].to_numpy(dtype=float)
    rows = []
    for target in selected:
        idx = int(np.nanargmin(np.abs(x - target)))
        rec = df.iloc[idx].copy()
        rec["selected_target_wavenumber"] = target
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(out: pd.DataFrame) -> pd.DataFrame:
    slope = out["dB_dT_cm_inv_amagat_neg2_K_neg1"]
    u_slope = out["u_dB_dT_cm_inv_amagat_neg2_K_neg1"]
    max_row = out.loc[slope.idxmax()]
    min_row = out.loc[slope.idxmin()]
    peak_row = out.loc[out["B_303K_weighted"].idxmax()]
    return pd.DataFrame(
        [
            {
                "n_points": int(len(out)),
                "wavenumber_min": float(out["wavenumber"].min()),
                "wavenumber_max": float(out["wavenumber"].max()),
                "dB_dT_min": float(min_row["dB_dT_cm_inv_amagat_neg2_K_neg1"]),
                "dB_dT_min_wavenumber": float(min_row["wavenumber"]),
                "dB_dT_max": float(max_row["dB_dT_cm_inv_amagat_neg2_K_neg1"]),
                "dB_dT_max_wavenumber": float(max_row["wavenumber"]),
                "u_dB_dT_median": float(u_slope.median()),
                "u_dB_dT_mean": float(u_slope.mean()),
                "u_dB_dT_max": float(u_slope.max()),
                "dB_dT_relative_u_median_percent": float((u_slope / slope.abs()).median() * 100.0),
                "fit_birge_ratio_median": float(out["fit_birge_ratio"].median()),
                "fit_birge_ratio_mean": float(out["fit_birge_ratio"].mean()),
                "B_303K_weighted_peak": float(peak_row["B_303K_weighted"]),
                "B_303K_weighted_peak_wavenumber": float(peak_row["wavenumber"]),
            }
        ]
    )


def plot_result(out: pd.DataFrame, selected: pd.DataFrame, output_png: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
        }
    )
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.3, 3.7), constrained_layout=True)

    temps = np.asarray([273.0, 303.0, 333.0], dtype=float)
    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2))]
    for i, (_, row) in enumerate(selected.iterrows()):
        b_vals = np.asarray(
            [row["B_273K_500Torr"], row["B_303K_weighted"], row["B_333K_500Torr"]],
            dtype=float,
        )
        u_vals = np.asarray(
            [
                row["u_B_273K_500Torr"],
                row["u_B_303K_weighted_scaled"],
                row["u_B_333K_500Torr"],
            ],
            dtype=float,
        )
        t_line = np.linspace(268.0, 338.0, 120)
        fit_line = row["fit_intercept"] + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        ax_left.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=1.0,
            label=f"{row['wavenumber']:.0f} cm$^{{-1}}$",
        )
        ax_left.errorbar(
            temps,
            b_vals / 1e-6,
            yerr=u_vals / 1e-6,
            marker=markers[i % len(markers)],
            color="black",
            linestyle="none",
            ms=4.2,
            capsize=2.0,
            lw=0.8,
            zorder=4,
        )

    ax_left.set_xlabel("Temperature (K)")
    ax_left.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_left.set_xlim(268.0, 338.0)
    ax_left.legend(frameon=False, fontsize=8, loc="best")
    ax_left.minorticks_on()

    x = out["wavenumber"].to_numpy(dtype=float)
    slope = out["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    u_slope = out["u_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    ax_right.plot(x, slope, color="#1f77b4", lw=1.1, label=r"$dB/dT$")
    ax_right.fill_between(x, slope - u_slope, slope + u_slope, color="#1f77b4", alpha=0.18, lw=0)
    ax_right.axhline(0, color="black", lw=0.8)
    for i, (_, row) in enumerate(selected.iterrows()):
        x0 = float(row["wavenumber"])
        y0 = float(row["dB_dT_cm_inv_amagat_neg2_K_neg1"]) / 1e-9
        ax_right.axvline(x0, color="0.75", lw=0.8, ls="--", zorder=0)
        ax_right.plot(x0, y0, "o", ms=3.5, color="black", zorder=3)
        offset = 8 if i % 2 == 0 else -14
        ax_right.annotate(
            f"{x0:.0f}",
            xy=(x0, y0),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=8,
        )
    ax_right.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_right.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_right.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax_right.legend(frameon=False, loc="best")
    ax_right.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def write_chinese_annotations(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Write a Chinese column-description table and a CSV with a comment row."""
    notes = pd.DataFrame(
        {
            "column": list(df.columns),
            "中文注释": [COLUMN_DESCRIPTIONS_ZH.get(col, "") for col in df.columns],
        }
    )
    notes_path = output_dir / "temperature_dependence_weighted_column_notes_zh.csv"
    annotated_path = output_dir / "temperature_dependence_weighted_fit_with_zh_notes.csv"
    notes.to_csv(notes_path, index=False, encoding="utf-8-sig")

    comment_row = {col: COLUMN_DESCRIPTIONS_ZH.get(col, "") for col in df.columns}
    annotated = pd.concat([pd.DataFrame([comment_row]), df], ignore_index=True)
    annotated.to_csv(annotated_path, index=False, encoding="utf-8-sig", float_format="%.15g")
    return notes_path, annotated_path


def write_origin_selected_table(selected: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Write Origin-friendly selected-wavenumber tables.

    The data table is arranged as:

        temperature_K, B_9200, u_B_9200, B_9323, u_B_9323, ...

    This lets Origin use the first column as X, every B column as Y, and the
    adjacent u_B column as the corresponding Y-error column.
    """
    temperatures = np.asarray([273.0, 303.0, 333.0], dtype=float)
    table = pd.DataFrame({"temperature_K": temperatures})
    fit_table = pd.DataFrame({"temperature_K": np.linspace(268.0, 338.0, 141)})

    for _, row in selected.iterrows():
        wn = float(row["wavenumber"])
        label = f"{wn:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        table[f"B_{label}"] = [
            row["B_273K_500Torr"],
            row["B_303K_weighted"],
            row["B_333K_500Torr"],
        ]
        table[f"u_B_{label}"] = [
            row["u_B_273K_500Torr"],
            row["u_B_303K_weighted_scaled"],
            row["u_B_333K_500Torr"],
        ]
        fit_table[f"B_fit_{label}"] = (
            row["fit_intercept"]
            + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * fit_table["temperature_K"]
        )

    data_path = output_dir / "temperature_dependence_origin_left_panel_points.csv"
    fit_path = output_dir / "temperature_dependence_origin_left_panel_fit_lines.csv"
    table.to_csv(data_path, index=False, float_format="%.15g")
    fit_table.to_csv(fit_path, index=False, float_format="%.15g")
    return data_path, fit_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = read_summary(args.input, args.encoding)
    out = calculate(
        source,
        fit_303_mode=args.fit_303_mode,
        combined_303_uncertainty=args.combined_303_uncertainty,
        input_uncertainty_scale=args.input_uncertainty_scale,
    )
    selected = nearest_rows(out, args.selected)
    summary = summarize(out)

    fit_csv = output_dir / "temperature_dependence_weighted_fit.csv"
    selected_csv = output_dir / "temperature_dependence_weighted_selected_wavenumbers.csv"
    summary_csv = output_dir / "temperature_dependence_weighted_summary.csv"
    figure = output_dir / "temperature_dependence_weighted.png"
    out.to_csv(fit_csv, index=False, float_format="%.15g")
    selected.to_csv(selected_csv, index=False, float_format="%.15g")
    summary.to_csv(summary_csv, index=False, float_format="%.15g")
    notes_csv, annotated_csv = write_chinese_annotations(out, output_dir)
    origin_points_csv, origin_fit_lines_csv = write_origin_selected_table(selected, output_dir)
    plot_result(out, selected, figure, args.dpi)

    print(f"Fit CSV: {fit_csv}")
    print(f"Fit CSV with Chinese notes: {annotated_csv}")
    print(f"Column notes CSV: {notes_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Origin left-panel points CSV: {origin_points_csv}")
    print(f"Origin left-panel fit-lines CSV: {origin_fit_lines_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {figure}")
    print(summary.to_string(index=False))
    print("Selected wavenumbers:")
    print(
        selected[
            [
                "wavenumber",
                "B_273K_500Torr",
                "u_B_273K_500Torr",
                "B_303K_weighted",
                "u_B_303K_weighted_scaled",
                "B_333K_500Torr",
                "u_B_333K_500Torr",
                "dB_dT_cm_inv_amagat_neg2_K_neg1",
                "u_dB_dT_cm_inv_amagat_neg2_K_neg1",
                "fit_birge_ratio",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
