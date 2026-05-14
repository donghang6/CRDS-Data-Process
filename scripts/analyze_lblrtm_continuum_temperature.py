#!/usr/bin/env python3
"""Process LBLRTM total/line absorption simulations for O2 continuum.

The input directory is expected to contain paired files such as:

    LBLRTM_O2_9120_9820_296_Kelvin_500_torr.txt
    LBLRTM_O2_9120_9820_296_Kelvin_500_torr线吸收.txt

The file without "线吸收" is interpreted as line + continuum absorption.
The file with "线吸收" is interpreted as line absorption only. Their
difference is the continuum absorption coefficient:

    alpha_cont = alpha_total - alpha_line

The O2-O2 binary collision-induced absorption coefficient is calculated with:

    rho = (P / 760) * (273.15 / T)        [amagat]
    B   = alpha_cont / rho^2              [cm^-1 amagat^-2]

Outputs include long and wide CSV tables plus figures for:

1. continuum absorption coefficient,
2. binary coefficient B,
3. temperature-dependence coefficient dB/dT.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("/Users/donghang/科研/实验数据/氧气连续吸收温度/LBLRTM 仿真数据")
LOSCHMIDT_TEMPERATURE_K = 273.15
STANDARD_PRESSURE_TORR = 760.0


@dataclass(frozen=True)
class LblrtmMeta:
    gas: str
    wavenumber_start: float
    wavenumber_end: float
    temperature_k: float
    pressure_torr: float
    is_line_only: bool
    path: Path

    @property
    def key(self) -> tuple[str, float, float, float, float]:
        return (
            self.gas,
            self.wavenumber_start,
            self.wavenumber_end,
            self.temperature_k,
            self.pressure_torr,
        )

    @property
    def label(self) -> str:
        return f"{self.temperature_k:g}K_{self.pressure_torr:g}Torr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate LBLRTM O2 continuum, B coefficient, and dB/dT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="LBLRTM data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/LBLRTM_continuum_temperature"),
        help="Output directory.",
    )
    parser.add_argument(
        "--pattern",
        default="LBLRTM_O2_*_Kelvin_*_torr*.txt",
        help="Input glob pattern.",
    )
    parser.add_argument(
        "--selected",
        type=float,
        nargs="+",
        default=[9200.0, 9322.95, 9427.60, 9700.0, 9800.0],
        help="Selected wavenumbers for B-vs-T panel.",
    )
    parser.add_argument(
        "--clip-negative-continuum",
        action="store_true",
        help="Clip negative continuum values to zero after subtraction.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def parse_file_name(path: Path) -> LblrtmMeta | None:
    pattern = re.compile(
        r"^LBLRTM_(?P<gas>[^_]+)_"
        r"(?P<wn0>\d+(?:\.\d+)?)_"
        r"(?P<wn1>\d+(?:\.\d+)?)_"
        r"(?P<temp>\d+(?:\.\d+)?)_Kelvin_"
        r"(?P<pressure>\d+(?:\.\d+)?)_torr"
        r"(?P<line>线吸收)?\.txt$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    return LblrtmMeta(
        gas=match.group("gas"),
        wavenumber_start=float(match.group("wn0")),
        wavenumber_end=float(match.group("wn1")),
        temperature_k=float(match.group("temp")),
        pressure_torr=float(match.group("pressure")),
        is_line_only=bool(match.group("line")),
        path=path,
    )


def discover_pairs(input_dir: Path, pattern: str) -> tuple[list[tuple[LblrtmMeta, LblrtmMeta]], pd.DataFrame]:
    total_files: dict[tuple[str, float, float, float, float], LblrtmMeta] = {}
    line_files: dict[tuple[str, float, float, float, float], LblrtmMeta] = {}
    records = []

    for path in sorted(input_dir.expanduser().glob(pattern)):
        meta = parse_file_name(path)
        if meta is None:
            records.append({"path": str(path), "status": "unrecognized_name"})
            continue
        target = line_files if meta.is_line_only else total_files
        if meta.key in target:
            records.append({"path": str(path), "status": "duplicate_key"})
            continue
        target[meta.key] = meta

    pairs = []
    all_keys = sorted(set(total_files) | set(line_files), key=lambda k: (k[3], k[4], k[1], k[2], k[0]))
    for key in all_keys:
        total = total_files.get(key)
        line = line_files.get(key)
        if total is not None and line is not None:
            pairs.append((total, line))
            status = "paired"
        elif total is None:
            status = "missing_total_absorption"
        else:
            status = "missing_line_absorption"
        meta = total if total is not None else line
        records.append(
            {
                "gas": meta.gas,
                "wavenumber_start": meta.wavenumber_start,
                "wavenumber_end": meta.wavenumber_end,
                "temperature_k": meta.temperature_k,
                "pressure_torr": meta.pressure_torr,
                "status": status,
                "total_file": str(total.path) if total is not None else "",
                "line_file": str(line.path) if line is not None else "",
            }
        )

    return pairs, pd.DataFrame(records)


def read_absorption_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path.expanduser(), sep=r"\s+", engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    out = df.iloc[:, :2].copy()
    out.columns = ["wavenumber", "alpha_cm_inv"]
    out["wavenumber"] = pd.to_numeric(out["wavenumber"], errors="coerce")
    out["alpha_cm_inv"] = pd.to_numeric(out["alpha_cm_inv"], errors="coerce")
    out = out.dropna().sort_values("wavenumber").reset_index(drop=True)
    return out


def calculate_continuum_pair(
    total_meta: LblrtmMeta,
    line_meta: LblrtmMeta,
    clip_negative: bool,
) -> pd.DataFrame:
    total = read_absorption_file(total_meta.path).rename(columns={"alpha_cm_inv": "total_alpha_cm_inv"})
    line = read_absorption_file(line_meta.path).rename(columns={"alpha_cm_inv": "line_alpha_cm_inv"})
    merged = total.merge(line, on="wavenumber", how="inner")
    merged["continuum_alpha_cm_inv"] = merged["total_alpha_cm_inv"] - merged["line_alpha_cm_inv"]
    if clip_negative:
        merged["continuum_alpha_cm_inv"] = merged["continuum_alpha_cm_inv"].clip(lower=0.0)

    rho_amagat = (
        (total_meta.pressure_torr / STANDARD_PRESSURE_TORR)
        * (LOSCHMIDT_TEMPERATURE_K / total_meta.temperature_k)
    )
    merged.insert(0, "gas", total_meta.gas)
    merged.insert(1, "temperature_k", total_meta.temperature_k)
    merged.insert(2, "pressure_torr", total_meta.pressure_torr)
    merged["rho_amagat"] = rho_amagat
    merged["B_cm_inv_amagat_neg2"] = merged["continuum_alpha_cm_inv"] / rho_amagat**2
    merged["n_total_points"] = len(total)
    merged["n_line_points"] = len(line)
    merged["n_common_points"] = len(merged)
    merged["total_file"] = str(total_meta.path)
    merged["line_file"] = str(line_meta.path)
    return merged


def make_wide_table(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pieces = []
    for (temperature, pressure), group in long_df.groupby(["temperature_k", "pressure_torr"], sort=True):
        label = f"{temperature:g}K_{pressure:g}Torr"
        piece = group[["wavenumber", value_col]].rename(columns={value_col: label})
        pieces.append(piece)
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.merge(piece, on="wavenumber", how="outer")
    return wide.sort_values("wavenumber").reset_index(drop=True)


def fit_temperature_dependence(b_wide: pd.DataFrame) -> pd.DataFrame:
    condition_cols = [col for col in b_wide.columns if col != "wavenumber"]
    temperatures = np.asarray([float(col.split("K_")[0]) for col in condition_cols], dtype=float)
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    condition_cols = [condition_cols[i] for i in order]
    y = b_wide[condition_cols].to_numpy(dtype=float)

    valid = np.all(np.isfinite(y), axis=1)
    x_lin = np.vstack([np.ones_like(temperatures), temperatures]).T
    beta_lin = np.full((2, len(b_wide)), np.nan)
    y_fit = np.full_like(y, np.nan)
    beta_lin[:, valid] = np.linalg.lstsq(x_lin, y[valid].T, rcond=None)[0]
    y_fit[valid] = (x_lin @ beta_lin[:, valid]).T
    residual = y - y_fit

    y_mean = np.nanmean(y, axis=1, keepdims=True)
    ss_tot = np.nansum((y - y_mean) ** 2, axis=1)
    ss_res = np.nansum(residual**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(temperatures))
        rmse_rel_percent = 100.0 * rmse / np.abs(np.nanmean(y, axis=1))

    out = pd.DataFrame(
        {
            "wavenumber": b_wide["wavenumber"].to_numpy(dtype=float),
            "fit_temperature_points": ",".join(f"{t:g}" for t in temperatures),
            "fit_intercept": beta_lin[0],
            "dB_dT_cm_inv_amagat_neg2_K_neg1": beta_lin[1],
            "linear_R2": r2,
            "linear_rmse_cm_inv_amagat_neg2": rmse,
            "linear_rmse_rel_percent": rmse_rel_percent,
        }
    )
    for col in condition_cols:
        out[col] = b_wide[col].to_numpy(dtype=float)
    return out


def nearest_rows(df: pd.DataFrame, selected: list[float]) -> pd.DataFrame:
    x = df["wavenumber"].to_numpy(dtype=float)
    rows = []
    for target in selected:
        idx = int(np.nanargmin(np.abs(x - target)))
        row = df.iloc[idx].copy()
        row["selected_target_wavenumber"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(long_df: pd.DataFrame, fit_df: pd.DataFrame, pair_summary: pd.DataFrame) -> pd.DataFrame:
    slope = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"]
    b_peak_row = long_df.loc[long_df["B_cm_inv_amagat_neg2"].idxmax()]
    alpha_peak_row = long_df.loc[long_df["continuum_alpha_cm_inv"].idxmax()]
    negative_count = int((long_df["continuum_alpha_cm_inv"] < 0).sum())
    return pd.DataFrame(
        [
            {
                "n_paired_conditions": int((pair_summary["status"] == "paired").sum()),
                "n_unpaired_or_problem_files": int((pair_summary["status"] != "paired").sum()),
                "temperature_min_k": long_df["temperature_k"].min(),
                "temperature_max_k": long_df["temperature_k"].max(),
                "pressure_min_torr": long_df["pressure_torr"].min(),
                "pressure_max_torr": long_df["pressure_torr"].max(),
                "wavenumber_min": long_df["wavenumber"].min(),
                "wavenumber_max": long_df["wavenumber"].max(),
                "n_negative_continuum_points": negative_count,
                "max_continuum_alpha_cm_inv": alpha_peak_row["continuum_alpha_cm_inv"],
                "max_continuum_alpha_wavenumber": alpha_peak_row["wavenumber"],
                "max_continuum_alpha_temperature_k": alpha_peak_row["temperature_k"],
                "max_B_cm_inv_amagat_neg2": b_peak_row["B_cm_inv_amagat_neg2"],
                "max_B_wavenumber": b_peak_row["wavenumber"],
                "max_B_temperature_k": b_peak_row["temperature_k"],
                "dB_dT_max": slope.max(),
                "dB_dT_max_wavenumber": fit_df.loc[slope.idxmax(), "wavenumber"],
                "dB_dT_min": slope.min(),
                "dB_dT_min_wavenumber": fit_df.loc[slope.idxmin(), "wavenumber"],
                "linear_R2_median": fit_df["linear_R2"].median(),
                "linear_rmse_rel_percent_median": fit_df["linear_rmse_rel_percent"].median(),
            }
        ]
    )


def plot_multi_temperature(
    wide: pd.DataFrame,
    y_label: str,
    title: str,
    output_png: Path,
    dpi: int,
    scale: float = 1.0,
) -> None:
    condition_cols = [col for col in wide.columns if col != "wavenumber"]
    temperatures = np.asarray([float(col.split("K_")[0]) for col in condition_cols], dtype=float)
    order = np.argsort(temperatures)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    for rank, idx in enumerate(order):
        col = condition_cols[idx]
        color = cmap(rank / max(len(order) - 1, 1))
        ax.plot(wide["wavenumber"], wide[col] / scale, lw=1.0, color=color, label=col.replace("_", ", "))
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.minorticks_on()
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def plot_temperature_dependence(
    fit_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_png: Path,
    dpi: int,
) -> None:
    condition_cols = [
        col for col in fit_df.columns if re.match(r"^\d+(?:\.\d+)?K_\d+(?:\.\d+)?Torr$", col)
    ]
    temperatures = np.asarray([float(col.split("K_")[0]) for col in condition_cols], dtype=float)
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    condition_cols = [condition_cols[i] for i in order]

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
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2))]

    for i, (_, row) in enumerate(selected_df.iterrows()):
        b_vals = row[condition_cols].to_numpy(dtype=float)
        t_line = np.linspace(float(np.nanmin(temperatures)), float(np.nanmax(temperatures)), 200)
        fit_line = row["fit_intercept"] + row["dB_dT_cm_inv_amagat_neg2_K_neg1"] * t_line
        label = f"{row['wavenumber']:.1f} cm$^{{-1}}$"
        ax_left.plot(
            t_line,
            fit_line / 1e-6,
            color="black",
            linestyle=linestyles[i % len(linestyles)],
            lw=1.0,
            label=label,
        )
        ax_left.plot(
            temperatures,
            b_vals / 1e-6,
            marker=markers[i % len(markers)],
            color="black",
            linestyle="none",
            ms=3.0,
            zorder=4,
        )
    ax_left.set_xlabel("Temperature (K)")
    ax_left.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax_left.legend(frameon=False, fontsize=7)
    ax_left.minorticks_on()

    x = fit_df["wavenumber"].to_numpy(dtype=float)
    slope = fit_df["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    ax_right.plot(x, slope, color="#1f77b4", lw=1.2, label="LBLRTM")
    for i, (_, row) in enumerate(selected_df.iterrows()):
        x0 = float(row["wavenumber"])
        y0 = float(row["dB_dT_cm_inv_amagat_neg2_K_neg1"]) / 1e-9
        ax_right.axvline(x0, color="0.75", lw=0.8, ls="--", zorder=0)
        ax_right.plot(x0, y0, marker="o", ms=3.5, color="black", zorder=3)
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
    ax_right.axhline(0, color="black", lw=0.8)
    ax_right.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_right.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)")
    ax_right.legend(frameon=False)
    ax_right.minorticks_on()

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, pair_summary = discover_pairs(args.input_dir, args.pattern)
    if not pairs:
        raise RuntimeError("No paired total/line absorption files were found.")

    long_df = pd.concat(
        [calculate_continuum_pair(total, line, args.clip_negative_continuum) for total, line in pairs],
        ignore_index=True,
    )
    alpha_wide = make_wide_table(long_df, "continuum_alpha_cm_inv")
    b_wide = make_wide_table(long_df, "B_cm_inv_amagat_neg2")
    fit_df = fit_temperature_dependence(b_wide)
    selected_df = nearest_rows(fit_df, args.selected)
    summary_df = summarize(long_df, fit_df, pair_summary)

    pair_summary_csv = output_dir / "lblrtm_file_pairing_summary.csv"
    long_csv = output_dir / "lblrtm_continuum_and_B_long.csv"
    alpha_wide_csv = output_dir / "lblrtm_continuum_absorption_wide.csv"
    b_wide_csv = output_dir / "lblrtm_binary_coefficient_wide.csv"
    fit_csv = output_dir / "lblrtm_temperature_dependence_fit.csv"
    selected_csv = output_dir / "lblrtm_temperature_dependence_selected_wavenumbers.csv"
    summary_csv = output_dir / "lblrtm_summary.csv"

    pair_summary.to_csv(pair_summary_csv, index=False)
    long_df.to_csv(long_csv, index=False, float_format="%.15g")
    alpha_wide.to_csv(alpha_wide_csv, index=False, float_format="%.15g")
    b_wide.to_csv(b_wide_csv, index=False, float_format="%.15g")
    fit_df.to_csv(fit_csv, index=False, float_format="%.15g")
    selected_df.to_csv(selected_csv, index=False, float_format="%.15g")
    summary_df.to_csv(summary_csv, index=False, float_format="%.15g")

    pressure_values = sorted(long_df["pressure_torr"].unique())
    temperature_values = sorted(long_df["temperature_k"].unique())
    pressure_title = ", ".join(f"{p:g}" for p in pressure_values)
    temperature_title = f"{temperature_values[0]:g}-{temperature_values[-1]:g}"

    continuum_png = output_dir / "lblrtm_continuum_absorption_all.png"
    b_png = output_dir / "lblrtm_binary_coefficient_all.png"
    temp_dep_png = output_dir / "lblrtm_temperature_dependence.png"
    plot_multi_temperature(
        alpha_wide,
        r"Continuum absorption coefficient $\alpha_{\mathrm{cont}}$ (cm$^{-1}$)",
        f"LBLRTM O2 continuum absorption, P={pressure_title} Torr, T={temperature_title} K",
        continuum_png,
        args.dpi,
    )
    plot_multi_temperature(
        b_wide,
        r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)",
        f"LBLRTM O2-O2 binary coefficient, P={pressure_title} Torr, T={temperature_title} K",
        b_png,
        args.dpi,
        scale=1e-6,
    )
    plot_temperature_dependence(fit_df, selected_df, temp_dep_png, args.dpi)

    print(f"Pairing summary: {pair_summary_csv}")
    print(f"Continuum/B long CSV: {long_csv}")
    print(f"Continuum alpha wide CSV: {alpha_wide_csv}")
    print(f"Binary B wide CSV: {b_wide_csv}")
    print(f"Temperature dependence fit CSV: {fit_csv}")
    print(f"Selected wavenumbers CSV: {selected_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Continuum figure: {continuum_png}")
    print(f"Binary B figure: {b_png}")
    print(f"Temperature dependence figure: {temp_dep_png}")
    print(summary_df.to_string(index=False))
    unpaired = pair_summary[pair_summary["status"] != "paired"]
    if not unpaired.empty:
        print("Unpaired/problem files:")
        print(unpaired[["temperature_k", "pressure_torr", "status", "total_file", "line_file"]].to_string(index=False))
    print("Selected wavenumbers:")
    cols = [
        "wavenumber",
        "dB_dT_cm_inv_amagat_neg2_K_neg1",
        "linear_R2",
        "linear_rmse_rel_percent",
    ]
    print(selected_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
