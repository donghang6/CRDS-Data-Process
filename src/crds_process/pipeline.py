"""端到端处理流水线

编排从原始衰荡数据到最终光谱拟合的完整处理流程:
    1. 读取原始数据
    2. 衰荡时间统计处理（含离群值过滤）
    3. 计算吸收系数
    4. 基线拟合与扣除
    5. MATS 光谱拟合
    6. 结果保存与可视化
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crds_process.absorption.coefficients import ringdown_results_to_spectrum
from crds_process.baseline.fitting import subtract_baseline
from crds_process.config import Settings, load_config
from crds_process.io.exporters import save_fit_results, save_spectrum_csv
from crds_process.io.readers import load_scan_directory
from crds_process.ringdown.processing import process_all_scans
from crds_process.visualization.plots import (
    plot_absorption_spectrum,
    plot_tau_spectrum,
)


def run_pipeline(
    data_dir: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    skip_mats: bool = False,
) -> pd.DataFrame:
    """执行完整的 CRDS 数据处理流水线

    Parameters
    ----------
    data_dir : str or Path
        原始数据目录
    config_path : str or Path, optional
        配置文件路径
    output_dir : str or Path, optional
        输出目录，默认从配置读取
    skip_mats : bool
        是否跳过 MATS 拟合步骤

    Returns
    -------
    pd.DataFrame
        处理后的光谱数据
    """
    # 0. 加载配置
    cfg = load_config(config_path)
    if output_dir is None:
        output_dir = Path(cfg.paths.output_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] 读取原始数据: {data_dir}")
    scans = load_scan_directory(data_dir)
    print(f"      共读取 {len(scans)} 个扫描点")

    # 2. 衰荡时间处理
    print(f"[2/6] 衰荡时间处理 (方法: {cfg.ringdown.filter_method})")
    rd_results = process_all_scans(
        scans,
        filter_method=cfg.ringdown.filter_method,
        sigma=cfg.ringdown.sigma_clip_threshold,
        iqr_factor=cfg.ringdown.iqr_factor,
        min_events=cfg.ringdown.min_events_per_point,
    )
    print(f"      有效点: {len(rd_results)} / {len(scans)}")

    # 3. 计算吸收系数
    print(f"[3/6] 计算吸收系数 (τ₀ = {cfg.absorption.tau0_us} μs)")
    spectrum_df = ringdown_results_to_spectrum(
        rd_results,
        tau0_us=cfg.absorption.tau0_us,
        c_cm_s=cfg.absorption.speed_of_light_cm_s,
    )

    # 绘制衰荡时间光谱
    plot_tau_spectrum(
        spectrum_df["wavenumber"].values,
        spectrum_df["tau_mean"].values,
        save_path=figures_dir / "tau_spectrum.png",
    )

    # 4. 基线拟合与扣除
    print(f"[4/6] 基线拟合 (方法: {cfg.baseline.method})")
    if cfg.baseline.regions:
        spectrum_df = subtract_baseline(
            spectrum_df,
            regions=cfg.baseline.regions,
            method=cfg.baseline.method,
            poly_order=cfg.baseline.poly_order,
            spline_smoothing=cfg.baseline.spline_smoothing,
        )
    else:
        print("      [SKIP] 未指定基线区域，跳过基线拟合")
        spectrum_df["baseline"] = 0.0
        spectrum_df["alpha_corrected"] = spectrum_df["alpha"]

    # 绘制吸收光谱
    plot_absorption_spectrum(
        spectrum_df,
        save_path=figures_dir / "absorption_spectrum.png",
    )

    # 保存中间结果
    save_spectrum_csv(spectrum_df, results_dir / "spectrum.csv")
    print(f"      光谱数据已保存: {results_dir / 'spectrum.csv'}")

    # 5. MATS 拟合
    if not skip_mats:
        print("[5/6] MATS 光谱拟合")
        try:
            from crds_process.spectral.mats_wrapper import prepare_mats_input, run_mats_fit

            mats_input = prepare_mats_input(spectrum_df)
            fit_result = run_mats_fit(
                mats_input,
                temperature_K=cfg.gas.temperature_K,
                pressure_torr=cfg.gas.pressure_torr,
                species=cfg.gas.species,
                isotopologue=cfg.gas.isotopologue,
                database=cfg.mats.database,
                wavenumber_range=cfg.mats.wavenumber_range,
                fit_parameters=cfg.mats.fit_parameters,
                max_iterations=cfg.mats.max_iterations,
            )

            save_fit_results(
                {"parameters": fit_result.parameters, "uncertainties": fit_result.uncertainties},
                results_dir / "fit_results.json",
            )
            print(f"      拟合结果已保存: {results_dir / 'fit_results.json'}")

        except (ImportError, NotImplementedError) as e:
            print(f"      [SKIP] MATS 拟合跳过: {e}")
    else:
        print("[5/6] MATS 光谱拟合 [SKIP]")

    # 6. 完成
    print("[6/6] 处理完成!")
    print(f"      输出目录: {output_dir}")
    return spectrum_df

