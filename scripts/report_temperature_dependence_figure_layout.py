#!/usr/bin/env python3
"""Print the exact matplotlib layout used for the temperature-dependence figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_temperature_dependence_from_summary import calculate, nearest_rows, read_summary


def main() -> None:
    source = read_summary(
        Path("/Users/donghang/科研/实验数据/氧气连续吸收温度/二元碰撞吸收系数/summary.txt"),
        "gbk",
    )
    out = calculate(
        source,
        fit_303_mode="combined",
        combined_303_uncertainty="scaled",
        input_uncertainty_scale=2.0,
    )
    selected = nearest_rows(out, [9200.0, 9323.0, 9420.0, 9520.0, 9800.0])

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
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(9.3, 3.7),
        constrained_layout=True,
    )

    temps = np.asarray([273.0, 303.0, 333.0], dtype=float)
    markers = ["o", "^", "s", "D", "v"]
    linestyles = ["-", "--", ":", "-.", (0, (5, 2))]
    for i, (_, row) in enumerate(selected.iterrows()):
        b_vals = np.asarray(
            [row["B_273K_500Torr"], row["B_303K_weighted"], row["B_333K_500Torr"]],
            dtype=float,
        )
        u_vals = np.asarray(
            [row["u_B_273K_500Torr"], row["u_B_303K_weighted_scaled"], row["u_B_333K_500Torr"]],
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

    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    dpi = fig.dpi
    print(f"figure_inches {fig_w:.6f} {fig_h:.6f}")
    print(f"figure_dpi {dpi:.6f}")
    print(f"figure_pixels {fig_w * dpi:.3f} {fig_h * dpi:.3f}")
    for name, ax in (("left", ax_left), ("right", ax_right)):
        pos = ax.get_position()
        inches = [pos.x0 * fig_w, pos.y0 * fig_h, pos.width * fig_w, pos.height * fig_h]
        pixels = [value * dpi for value in inches]
        print(
            name,
            "fraction",
            f"{pos.x0:.9f}",
            f"{pos.y0:.9f}",
            f"{pos.width:.9f}",
            f"{pos.height:.9f}",
        )
        print(name, "inches", " ".join(f"{value:.6f}" for value in inches))
        print(name, "pixels", " ".join(f"{value:.3f}" for value in pixels))
    gap = ax_right.get_position().x0 * fig_w - (
        ax_left.get_position().x0 + ax_left.get_position().width
    ) * fig_w
    print(f"horizontal_gap_inches {gap:.6f}")
    print(f"horizontal_gap_pixels {gap * dpi:.3f}")


if __name__ == "__main__":
    main()
