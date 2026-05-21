#!/usr/bin/env python3
"""Plot B spectra and the 303 K pressure-group comparison.

Upper panel:
    B spectra for 273 K 500 Torr, 303 K 500/600/700 Torr, and 333 K 500 Torr.

Lower panel:
    Relative differences of the 303 K pressure groups using one pressure as
    reference.  By default 600 Torr is used because it is the middle pressure
    and gives a balanced comparison:

        Delta_i = (B_i - B_600) / B_600 * 100 %

    The gray band is the propagated 1-sigma relative uncertainty envelope for
    the relative differences.  The lower-panel CSV also includes explicit
    upper/lower boundary columns so that the uncertainty band can be drawn
    directly in Origin by filling between two curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "output/results/analysis/B_temperature_dependence_from_summary_303_combined_uB_x2/"
    "temperature_dependence_weighted_fit.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/results/analysis/B_pressure_temperature_and_303_pressure_comparison"
)

PRESSURE_TO_COLUMNS = {
    "500": ("B_303K_500Torr", "u_B_303K_500Torr"),
    "600": ("B_303K_600Torr", "u_B_303K_600Torr"),
    "700": ("B_303K_700Torr", "u_B_303K_700Torr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot upper B spectra and lower 303 K pressure comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference-pressure",
        choices=("500", "600", "700", "auto"),
        default="auto",
        help="Reference pressure for lower-panel relative differences.",
    )
    parser.add_argument("--figure-width-mm", type=float, default=95.0)
    parser.add_argument("--figure-height-mm", type=float, default=92.0)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def choose_reference_pressure(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    records = []
    for ref_label, (ref_col, _) in PRESSURE_TO_COLUMNS.items():
        ref = pd.to_numeric(df[ref_col], errors="coerce").to_numpy(dtype=float)
        medians = []
        p5_values = []
        p95_values = []
        max_abs_values = []
        for label, (col, _) in PRESSURE_TO_COLUMNS.items():
            if label == ref_label:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (values - ref) / ref * 100.0
            rel = rel[np.isfinite(rel)]
            medians.append(float(np.nanmedian(np.abs(rel))))
            p5_values.append(float(np.nanpercentile(rel, 5)))
            p95_values.append(float(np.nanpercentile(rel, 95)))
            max_abs_values.append(float(np.nanmax(np.abs(rel))))
        records.append(
            {
                "reference_pressure_torr": ref_label,
                "median_abs_relative_difference_percent_mean": float(np.mean(medians)),
                "p5_percent_min": float(np.min(p5_values)),
                "p95_percent_max": float(np.max(p95_values)),
                "max_abs_relative_difference_percent": float(np.max(max_abs_values)),
            }
        )
    summary = pd.DataFrame(records)
    # Prefer the middle pressure if its spread is comparable; it is easiest to
    # interpret experimentally and keeps the two non-reference groups explicit.
    return "600", summary


def relative_difference_and_uncertainty(
    b: np.ndarray,
    u_b: np.ndarray,
    b_ref: np.ndarray,
    u_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (b - b_ref) / b_ref * 100.0
        u_rel = 100.0 * np.sqrt((u_b / b_ref) ** 2 + (b * u_ref / b_ref**2) ** 2)
    return rel, u_rel


def build_plot_data(df: pd.DataFrame, reference_pressure: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wn = pd.to_numeric(df["wavenumber"], errors="coerce").to_numpy(dtype=float)
    ref_col, ref_u_col = PRESSURE_TO_COLUMNS[reference_pressure]
    b_ref = pd.to_numeric(df[ref_col], errors="coerce").to_numpy(dtype=float)
    u_ref = pd.to_numeric(df[ref_u_col], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame({"wavenumber": wn})
    uncertainty_candidates = []
    for label, (col, u_col) in PRESSURE_TO_COLUMNS.items():
        b = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        u_b = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype=float)
        rel, u_rel = relative_difference_and_uncertainty(b, u_b, b_ref, u_ref)
        if label == reference_pressure:
            rel = np.zeros_like(rel)
            u_rel = np.zeros_like(u_rel)
        out[f"relative_difference_303K_{label}Torr_vs_{reference_pressure}Torr_percent"] = rel
        out[f"u_relative_difference_303K_{label}Torr_vs_{reference_pressure}Torr_percent"] = u_rel
        out[
            f"relative_difference_303K_{label}Torr_vs_{reference_pressure}Torr_lower_1sigma_percent"
        ] = rel - u_rel
        out[
            f"relative_difference_303K_{label}Torr_vs_{reference_pressure}Torr_upper_1sigma_percent"
        ] = rel + u_rel
        if label != reference_pressure:
            uncertainty_candidates.append(u_rel)
    if uncertainty_candidates:
        out["relative_uncertainty_envelope_percent"] = np.nanmax(
            np.vstack(uncertainty_candidates), axis=0
        )
    else:
        out["relative_uncertainty_envelope_percent"] = np.nan
    out["relative_uncertainty_envelope_lower_1sigma_percent"] = (
        -out["relative_uncertainty_envelope_percent"]
    )
    out["relative_uncertainty_envelope_upper_1sigma_percent"] = (
        out["relative_uncertainty_envelope_percent"]
    )
    out["reference_pressure_torr"] = float(reference_pressure)
    return out, pd.DataFrame({"wavenumber": wn})


def write_origin_tables(
    df: pd.DataFrame,
    rel_df: pd.DataFrame,
    reference_pressure: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    upper_cols = [
        "wavenumber",
        "B_273K_500Torr",
        "B_303K_500Torr",
        "B_303K_600Torr",
        "B_303K_700Torr",
        "B_333K_500Torr",
    ]
    upper = df[upper_cols].copy()
    upper.to_csv(output_dir / "B_temperature_pressure_upper_panel_data.csv", index=False, float_format="%.15g")

    rel_path = output_dir / f"relative_303K_pressure_comparison_vs_{reference_pressure}Torr.csv"
    rel_df.to_csv(rel_path, index=False, float_format="%.15g")

    non_ref_labels = [label for label in ("500", "600", "700") if label != reference_pressure]
    origin_columns = [
        "wavenumber",
        "relative_uncertainty_envelope_lower_1sigma_percent",
        "relative_uncertainty_envelope_upper_1sigma_percent",
    ]
    for label in non_ref_labels:
        prefix = f"relative_difference_303K_{label}Torr_vs_{reference_pressure}Torr"
        origin_columns.extend(
            [
                f"{prefix}_percent",
                f"{prefix}_lower_1sigma_percent",
                f"{prefix}_upper_1sigma_percent",
            ]
        )
    origin_path = (
        output_dir
        / f"relative_303K_pressure_comparison_vs_{reference_pressure}Torr_origin_bounds.csv"
    )
    rel_df[origin_columns].to_csv(origin_path, index=False, float_format="%.15g")
    return output_dir / "B_temperature_pressure_upper_panel_data.csv", rel_path, origin_path


def plot_figure(
    df: pd.DataFrame,
    rel_df: pd.DataFrame,
    reference_pressure: str,
    output_dir: Path,
    width_mm: float,
    height_mm: float,
    dpi: int,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.12, top=0.975, hspace=0.08)

    x = pd.to_numeric(df["wavenumber"], errors="coerce").to_numpy(dtype=float)
    upper_specs = [
        ("B_273K_500Torr", "273 K, 500 Torr", "#0072B2", "-"),
        ("B_303K_500Torr", "303 K, 500 Torr", "#009E73", "-"),
        ("B_303K_600Torr", "303 K, 600 Torr", "#D55E00", "-"),
        ("B_303K_700Torr", "303 K, 700 Torr", "#CC79A7", "-"),
        ("B_333K_500Torr", "333 K, 500 Torr", "#E69F00", "-"),
    ]
    for col, label, color, linestyle in upper_specs:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) / 1e-6
        axes[0].plot(x, y, color=color, lw=0.75, ls=linestyle, label=label)
    axes[0].set_ylabel(r"$B$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    axes[0].legend(frameon=False, loc="upper right", ncol=1, handlelength=1.8, labelspacing=0.2)
    axes[0].minorticks_on()

    rel_x = rel_df["wavenumber"].to_numpy(dtype=float)
    envelope = rel_df["relative_uncertainty_envelope_percent"].to_numpy(dtype=float)
    axes[1].fill_between(
        rel_x,
        -envelope,
        envelope,
        color="#BDBDBD",
        alpha=0.42,
        lw=0,
        label=r"$\pm1\sigma$ rel. unc.",
    )
    axes[1].axhline(0.0, color="black", lw=0.55)
    lower_colors = {"500": "#009E73", "600": "#D55E00", "700": "#CC79A7"}
    for label in ("500", "600", "700"):
        if label == reference_pressure:
            continue
        col = f"relative_difference_303K_{label}Torr_vs_{reference_pressure}Torr_percent"
        axes[1].plot(
            rel_x,
            rel_df[col].to_numpy(dtype=float),
            color=lower_colors[label],
            lw=0.75,
            label=f"303 K {label} Torr vs {reference_pressure} Torr",
        )
    axes[1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Relative diff. (%)")
    axes[1].legend(frameon=False, loc="best", handlelength=1.8, labelspacing=0.2)
    axes[1].minorticks_on()

    for ax in axes:
        ax.tick_params(pad=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    axes[0].set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))

    out_png = output_dir / f"B_temperature_pressure_and_303K_pressure_comparison_vs_{reference_pressure}Torr.png"
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_png.with_suffix(".pdf"))
    fig.savefig(out_png.with_suffix(".tif"), dpi=dpi)
    plt.close(fig)
    return out_png


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = [
        "wavenumber",
        "B_273K_500Torr",
        "B_303K_500Torr",
        "u_B_303K_500Torr",
        "B_303K_600Torr",
        "u_B_303K_600Torr",
        "B_303K_700Torr",
        "u_B_303K_700Torr",
        "B_333K_500Torr",
    ]
    require_columns(df, required, input_path)

    auto_ref, reference_summary = choose_reference_pressure(df)
    reference_pressure = auto_ref if args.reference_pressure == "auto" else args.reference_pressure
    reference_summary.to_csv(
        output_dir / "reference_pressure_selection_summary.csv",
        index=False,
        float_format="%.15g",
    )
    rel_df, _ = build_plot_data(df, reference_pressure)
    upper_csv, rel_csv, origin_rel_csv = write_origin_tables(df, rel_df, reference_pressure, output_dir)
    figure_png = plot_figure(
        df=df,
        rel_df=rel_df,
        reference_pressure=reference_pressure,
        output_dir=output_dir,
        width_mm=args.figure_width_mm,
        height_mm=args.figure_height_mm,
        dpi=args.dpi,
    )
    summary = {
        "input": str(input_path),
        "reference_pressure_torr": reference_pressure,
        "figure_png": str(figure_png),
        "upper_panel_csv": str(upper_csv),
        "lower_panel_csv": str(rel_csv),
        "lower_panel_origin_bounds_csv": str(origin_rel_csv),
        "figure_width_mm": args.figure_width_mm,
        "figure_height_mm": args.figure_height_mm,
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "B_temperature_pressure_and_303K_pressure_comparison_summary.csv",
        index=False,
    )
    print(f"Reference pressure: {reference_pressure} Torr")
    print(f"Figure: {figure_png}")
    print(f"Upper data: {upper_csv}")
    print(f"Lower data: {rel_csv}")
    print(f"Lower Origin bounds data: {origin_rel_csv}")
    print(f"Reference selection: {output_dir / 'reference_pressure_selection_summary.csv'}")


if __name__ == "__main__":
    main()
