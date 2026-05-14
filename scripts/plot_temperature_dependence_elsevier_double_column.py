#!/usr/bin/env python3
"""Draw an Elsevier double-column version of the temperature-dependence figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("output/results/analysis/B_temperature_dependence_from_summary_303_combined_uB_x2")
FIT_CSV = ROOT / "temperature_dependence_weighted_fit.csv"
SELECTED_CSV = ROOT / "temperature_dependence_weighted_selected_wavenumbers.csv"
OUTPUT_PNG = ROOT / "temperature_dependence_elsevier_double_column.png"
OUTPUT_PDF = ROOT / "temperature_dependence_elsevier_double_column.pdf"

MM_PER_INCH = 25.4


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


def mm_rect(x: float, y: float, width: float, height: float, fig_w: float, fig_h: float) -> list[float]:
    return [x / fig_w, y / fig_h, width / fig_w, height / fig_h]


def main() -> None:
    fit = pd.read_csv(FIT_CSV)
    selected = pd.read_csv(SELECTED_CSV)

    fig_w_mm = 190.0
    fig_h_mm = 78.0
    left_rect_mm = (16.0, 14.0, 76.0, 60.0)
    right_rect_mm = (110.0, 14.0, 76.0, 60.0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
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

    fig = plt.figure(figsize=(mm_to_in(fig_w_mm), mm_to_in(fig_h_mm)))
    ax_left = fig.add_axes(mm_rect(*left_rect_mm, fig_w=fig_w_mm, fig_h=fig_h_mm))
    ax_right = fig.add_axes(mm_rect(*right_rect_mm, fig_w=fig_w_mm, fig_h=fig_h_mm))

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
            lw=0.7,
            label=f"{row['wavenumber']:.0f} cm$^{{-1}}$",
        )
        ax_left.errorbar(
            temps,
            b_vals / 1e-6,
            yerr=u_vals / 1e-6,
            marker=markers[i % len(markers)],
            color="black",
            linestyle="none",
            ms=3.0,
            capsize=1.8,
            elinewidth=0.55,
            capthick=0.55,
            zorder=4,
        )

    ax_left.set_xlabel("Temperature (K)", labelpad=2.0)
    ax_left.set_ylabel(r"$B_{\mathrm{O_2-O_2}}$ ($10^{-6}$ cm$^{-1}$ amagat$^{-2}$)", labelpad=2.0)
    ax_left.set_xlim(268.0, 338.0)
    ax_left.set_ylim(-0.01, 1.03)
    ax_left.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.0,
        labelspacing=0.28,
        borderaxespad=0.25,
    )
    ax_left.minorticks_on()

    x = fit["wavenumber"].to_numpy(dtype=float)
    slope = fit["dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    u_slope = fit["u_dB_dT_cm_inv_amagat_neg2_K_neg1"].to_numpy(dtype=float) / 1e-9
    blue = "#0072B2"
    ax_right.plot(x, slope, color=blue, lw=0.85, label=r"$dB/dT$")
    ax_right.fill_between(x, slope - u_slope, slope + u_slope, color=blue, alpha=0.18, lw=0)
    ax_right.axhline(0, color="black", lw=0.55)
    for i, (_, row) in enumerate(selected.iterrows()):
        x0 = float(row["wavenumber"])
        y0 = float(row["dB_dT_cm_inv_amagat_neg2_K_neg1"]) / 1e-9
        ax_right.axvline(x0, color="0.72", lw=0.55, ls="--", zorder=0)
        ax_right.plot(x0, y0, "o", ms=2.8, color="black", zorder=3)
        offsets = [(0, 7), (-7, -10), (8, 6), (0, -12), (0, 7)]
        dx, dy = offsets[i % len(offsets)]
        ax_right.annotate(
            f"{x0:.0f}",
            xy=(x0, y0),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=6.2,
        )
    ax_right.set_xlabel(r"Wavenumber (cm$^{-1}$)", labelpad=2.0)
    ax_right.set_ylabel(r"$dB/dT$ ($10^{-9}$ cm$^{-1}$ amagat$^{-2}$ K$^{-1}$)", labelpad=1.5)
    ax_right.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax_right.set_ylim(-0.05, 1.55)
    ax_right.legend(frameon=False, loc="upper right", handlelength=2.0, borderaxespad=0.25)
    ax_right.minorticks_on()

    for ax in (ax_left, ax_right):
        ax.tick_params(pad=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=600)
    plt.close(fig)
    print(f"PNG: {OUTPUT_PNG.resolve()}")
    print(f"PDF: {OUTPUT_PDF.resolve()}")
    print(f"Figure: {fig_w_mm:g} mm x {fig_h_mm:g} mm")
    print(f"Left axes: x={left_rect_mm[0]:g} mm, y={left_rect_mm[1]:g} mm, w={left_rect_mm[2]:g} mm, h={left_rect_mm[3]:g} mm")
    print(f"Right axes: x={right_rect_mm[0]:g} mm, y={right_rect_mm[1]:g} mm, w={right_rect_mm[2]:g} mm, h={right_rect_mm[3]:g} mm")


if __name__ == "__main__":
    main()
