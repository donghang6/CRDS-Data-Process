#!/usr/bin/env python3
"""Plot Karman ab initio B spectra at different temperatures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Karman B spectra at all available temperatures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/results/analysis/Karman_temperature_linearity/karman_temperature_linearity_fit.csv"),
        help="Karman fit table containing converted B columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results/analysis/Karman_temperature_linearity"),
        help="Output directory.",
    )
    parser.add_argument(
        "--selected-temperatures",
        type=float,
        nargs="+",
        default=[206.0, 246.0, 276.0, 296.15, 306.0, 336.0, 346.0],
        help="Temperatures to include in the selected-temperature figure.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def find_temperature_columns(df: pd.DataFrame) -> list[tuple[float, str]]:
    pattern = re.compile(r"^B_(\d+(?:\.\d+)?)K$")
    cols = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            cols.append((float(match.group(1)), col))
    if not cols:
        raise ValueError("No columns like B_296K were found.")
    return sorted(cols)


def set_plot_style() -> None:
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


def plot_all(df: pd.DataFrame, cols: list[tuple[float, str]], output_png: Path, dpi: int) -> None:
    x = df["wavenumber"].to_numpy(dtype=float)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for i, (temp, col) in enumerate(cols):
        color = cmap(i / max(len(cols) - 1, 1))
        ax.plot(x, df[col].to_numpy(dtype=float) / 1e-6, lw=1.0, color=color, label=f"{temp:g} K")
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax.set_title(r"Karman ab initio $B_{\mathrm{O_2-O_2}}$ at different temperatures")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.minorticks_on()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def plot_selected(
    df: pd.DataFrame,
    cols: list[tuple[float, str]],
    selected_temperatures: list[float],
    output_png: Path,
    dpi: int,
) -> None:
    x = df["wavenumber"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    used: set[str] = set()
    for target in selected_temperatures:
        temp, col = min(cols, key=lambda item: abs(item[0] - target))
        if col in used:
            continue
        used.add(col)
        ax.plot(x, df[col].to_numpy(dtype=float) / 1e-6, lw=1.3, label=f"{temp:g} K")
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)")
    ax.set_title(r"Karman ab initio $B_{\mathrm{O_2-O_2}}$ at selected temperatures")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.minorticks_on()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    cols = find_temperature_columns(df)
    wide = df[["wavenumber"] + [col for _, col in cols]].copy()

    wide_csv = output_dir / "karman_B_all_temperatures_wide.csv"
    all_png = output_dir / "karman_B_all_temperatures.png"
    selected_png = output_dir / "karman_B_selected_temperatures.png"

    wide.to_csv(wide_csv, index=False, float_format="%.15g")
    set_plot_style()
    plot_all(df, cols, all_png, args.dpi)
    plot_selected(df, cols, args.selected_temperatures, selected_png, args.dpi)

    print(f"Wide CSV: {wide_csv}")
    print(f"All-temperature figure: {all_png}")
    print(f"Selected-temperature figure: {selected_png}")
    print("Temperatures:", ", ".join(f"{temp:g} K" for temp, _ in cols))


if __name__ == "__main__":
    main()
