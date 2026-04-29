"""CRDS 数据处理 — 完整五步流水线

运行方式:
    python main.py                              # 处理全部数据
    python main.py O2/9386.2076                  # 仅处理 O2 下 9386.2076 跃迁
    python main.py O2/9386.2076/100Torr          # 仅处理 O2/9386.2076 的 100Torr
    python main.py O2/9386.2076 O2_N2/9386.2076  # 同时处理多个目标

    # 指定参与多光谱联合拟合的压力 (跳过自动筛选)
    python main.py --pressures O2/9386.2076=100Torr,200Torr,300Torr
    python main.py O2/9386.2076 --pressures O2/9386.2076=100Torr,200Torr

    # 自动搜索最优压力组合 (枚举所有组合, 选 QF 最大)
    python main.py O2/9386.2076 --optimize
    python main.py O2/9386.2076 --optimize --min-pressures 4

    # 仅拟合指定跃迁(吸收线); 多个值可用逗号分隔
    python main.py O2/9403.163069 --fit-transitions 9403.163069
    python main.py O2/9403.163069 --fit-lines 9403.163069,9401.731225

    # 仅提取 N2 展宽; 跳过纯 O2 联合拟合, 依赖已有纯 O2 Step 4 结果
    python main.py --n2-only O2_N2/9403.163069
    python main.py --from-ringdown --n2-only O2_N2/9403.163069
    python main.py --from-etalon --n2-only O2_N2/9403.163069

    # 生成建议重测点报告 (只读取已有结果)
    python main.py --remeasure-report
    python main.py --remeasure-report O2/9403.163069
    python main.py --remeasure-report --remeasure-rel 0.05 --remeasure-sigma 3
    python main.py --remeasure-report --remeasure-rel-o2 0.05 --remeasure-rel-o2n2 0.10

    # Monte Carlo Type A 误差分析 (基于已有纯 O2 多光谱联合拟合结果)
    python main.py --type-a-mc O2/9398.306147
    python main.py --type-a-mc O2/9398.306147 --mc-samples 100 --mc-wave-error-mhz 4

    # 连续吸收 / continuum absorption (仅处理 CIA 数据；只做 Step 1，跳过标准具/MATS)
    python main.py --continuum CIA/273K
    python main.py --continuum "CIA/273K/Ar 500Torr"
    python main.py --continuum --from-ringdown "CIA/273K/Ar 500Torr"
    python main.py --continuum "CIA/273K/Ar 500Torr" --continuum-tau0-us 102.3
    python main.py --continuum "CIA/273K/Ar 500Torr" \
        --continuum-ref 'output/results/ringdown/CIA/273K/Ar 500Torr/ringdown_results.csv'

    # 跳过 Step 1, 直接从已有的 ringdown 结果开始执行 Step 2~5
    python main.py --from-ringdown
    python main.py --from-ringdown O2/9386.2076
    python main.py --from-ringdown O2/9386.2076 \
        --pressures O2/9386.2076=100Torr,200Torr,300Torr

    # 跳过 Step 1/2, 直接从已有的去除标准具数据开始执行 Step 3~5
    python main.py --from-etalon
    python main.py --from-etalon O2/9386.2076
"""

import os
import sys
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "crds-data-process-cache"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))


def _parse_continuum_window(raw: str) -> tuple[float, float] | None:
    """Parse continuum window as start,end or start:end."""
    sep = "," if "," in raw else ":"
    parts = [p.strip() for p in raw.split(sep) if p.strip()]
    if len(parts) != 2:
        print(f"警告: --continuum-window 参数无效: {raw}，格式应为 start,end")
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print(f"警告: --continuum-window 参数无效: {raw}，格式应为 start,end")
        return None


def _parse_args(argv: list[str]) -> dict:
    """解析命令行参数

    Returns
    -------
    dict
        CRDSPipeline 构造参数
    """
    targets = []
    multi_fit_pressures: dict[str, list[str]] = {}
    fit_transitions: list[float] = []
    auto_optimize = False
    min_pressures = 3
    n2_only = False
    from_ringdown = False
    from_etalon = False
    remeasure_report = False
    remeasure_rel = None
    remeasure_rel_o2 = None
    remeasure_rel_o2n2 = None
    remeasure_sigma = 3.0
    type_a_mc = False
    mc_samples = 100
    mc_seed = 12345
    mc_wave_error_khz = 4000.0
    continuum = False
    continuum_reference_csv = None
    continuum_tau0_us = None
    continuum_window = None
    continuum_tau_col = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--from-ringdown":
            from_ringdown = True
            i += 1
        elif arg == "--from-etalon":
            from_etalon = True
            i += 1
        elif arg == "--n2-only":
            n2_only = True
            i += 1
        elif arg == "--remeasure-report":
            remeasure_report = True
            i += 1
        elif arg == "--type-a-mc":
            type_a_mc = True
            i += 1
        elif arg == "--continuum":
            continuum = True
            i += 1
        elif arg in ("--continuum-ref", "--continuum-reference"):
            i += 1
            if i < len(argv):
                continuum_reference_csv = argv[i]
            else:
                print(f"警告: {arg} 缺少参数，已忽略")
            i += 1
        elif arg == "--continuum-tau0-us":
            i += 1
            if i < len(argv):
                try:
                    continuum_tau0_us = float(argv[i])
                except ValueError:
                    print(f"警告: --continuum-tau0-us 参数无效: {argv[i]}，将只输出 loss")
            i += 1
        elif arg == "--continuum-window":
            i += 1
            if i < len(argv):
                continuum_window = _parse_continuum_window(argv[i])
            else:
                print("警告: --continuum-window 缺少参数，已忽略")
            i += 1
        elif arg == "--continuum-tau-col":
            i += 1
            if i < len(argv):
                continuum_tau_col = argv[i]
            else:
                print("警告: --continuum-tau-col 缺少参数，已忽略")
            i += 1
        elif arg == "--remeasure-rel":
            i += 1
            if i < len(argv):
                try:
                    remeasure_rel = float(argv[i])
                except ValueError:
                    print(f"警告: --remeasure-rel 参数无效: {argv[i]}，使用默认值 0.05")
            i += 1
        elif arg == "--remeasure-rel-o2":
            i += 1
            if i < len(argv):
                try:
                    remeasure_rel_o2 = float(argv[i])
                except ValueError:
                    print(f"警告: --remeasure-rel-o2 参数无效: {argv[i]}，使用默认值 0.05")
            i += 1
        elif arg == "--remeasure-rel-o2n2":
            i += 1
            if i < len(argv):
                try:
                    remeasure_rel_o2n2 = float(argv[i])
                except ValueError:
                    print(f"警告: --remeasure-rel-o2n2 参数无效: {argv[i]}，使用默认值 0.10")
            i += 1
        elif arg == "--remeasure-sigma":
            i += 1
            if i < len(argv):
                try:
                    remeasure_sigma = float(argv[i])
                except ValueError:
                    print(f"警告: --remeasure-sigma 参数无效: {argv[i]}，使用默认值 3")
            i += 1
        elif arg == "--mc-samples":
            i += 1
            if i < len(argv):
                try:
                    mc_samples = max(int(argv[i]), 1)
                except ValueError:
                    print(f"警告: --mc-samples 参数无效: {argv[i]}，使用默认值 100")
            i += 1
        elif arg == "--mc-seed":
            i += 1
            if i < len(argv):
                try:
                    mc_seed = int(argv[i])
                except ValueError:
                    print(f"警告: --mc-seed 参数无效: {argv[i]}，使用默认值 12345")
            i += 1
        elif arg == "--mc-wave-error-khz":
            i += 1
            if i < len(argv):
                try:
                    mc_wave_error_khz = max(float(argv[i]), 0.0)
                except ValueError:
                    print(f"警告: --mc-wave-error-khz 参数无效: {argv[i]}，使用默认值 4000")
            i += 1
        elif arg == "--mc-wave-error-mhz":
            i += 1
            if i < len(argv):
                try:
                    mc_wave_error_khz = max(float(argv[i]), 0.0) * 1000.0
                except ValueError:
                    print(f"警告: --mc-wave-error-mhz 参数无效: {argv[i]}，使用默认值 4")
            i += 1
        elif arg in ("--pressures", "-p"):
            # 后续参数格式: "气体/跃迁=压力1,压力2,..."
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                spec = argv[i]
                if "=" in spec:
                    key, val = spec.split("=", 1)
                    pressures = [p.strip() for p in val.split(",") if p.strip()]
                    if pressures:
                        multi_fit_pressures[key.strip()] = pressures
                else:
                    print(f"警告: 忽略无效的 --pressures 参数: {spec}")
                    print("  格式应为: 气体类型/跃迁=压力1,压力2,...")
                i += 1
        elif arg in ("--fit-transitions", "--fit-lines", "--fit-nu"):
            i += 1
            if i >= len(argv) or argv[i].startswith("-"):
                print(f"警告: {arg} 缺少参数，已忽略")
                continue
            raw_vals = [v.strip() for v in argv[i].split(",") if v.strip()]
            for raw in raw_vals:
                try:
                    fit_transitions.append(float(raw))
                except ValueError:
                    print(f"警告: 忽略无效的跃迁波数: {raw}")
            i += 1
        elif arg == "--optimize":
            auto_optimize = True
            i += 1
        elif arg == "--min-pressures":
            i += 1
            if i < len(argv):
                try:
                    min_pressures = int(argv[i])
                except ValueError:
                    print(f"警告: --min-pressures 参数无效: {argv[i]}，使用默认值 3")
            i += 1
        else:
            targets.append(arg)
            i += 1

    return {
        "targets": targets or None,
        "multi_fit_pressures": multi_fit_pressures or None,
        "fit_transitions": sorted(set(fit_transitions)) or None,
        "auto_optimize_pressures": auto_optimize,
        "min_multi_pressures": min_pressures,
        "remeasure_rel_threshold": remeasure_rel,
        "remeasure_rel_threshold_o2": remeasure_rel_o2,
        "remeasure_rel_threshold_o2n2": remeasure_rel_o2n2,
        "remeasure_sigma_threshold": remeasure_sigma,
        "type_a_mc_samples": mc_samples,
        "type_a_mc_seed": mc_seed,
        "type_a_mc_wave_error_khz": mc_wave_error_khz,
        "continuum_reference_csv": continuum_reference_csv,
        "continuum_tau0_us": continuum_tau0_us,
        "continuum_window": continuum_window,
        "continuum_tau_col": continuum_tau_col,
        "_n2_only": n2_only,
        "_from_ringdown": from_ringdown,
        "_from_etalon": from_etalon,
        "_remeasure_report": remeasure_report,
        "_type_a_mc": type_a_mc,
        "_continuum": continuum,
    }


if __name__ == "__main__":
    from crds_process.pipeline import CRDSPipeline

    kwargs = _parse_args(sys.argv[1:])
    n2_only = kwargs.pop("_n2_only")
    from_ringdown = kwargs.pop("_from_ringdown")
    from_etalon = kwargs.pop("_from_etalon")
    remeasure_report = kwargs.pop("_remeasure_report")
    type_a_mc = kwargs.pop("_type_a_mc")
    continuum = kwargs.pop("_continuum")

    if from_ringdown and from_etalon:
        print("错误: --from-ringdown 与 --from-etalon 不能同时使用")
        sys.exit(2)
    if continuum and n2_only:
        print("错误: --continuum 与 --n2-only 不能同时使用")
        sys.exit(2)
    if continuum and from_etalon:
        print("错误: --continuum 只使用 Step 1 的 ringdown_results.csv，不做标准具去除")
        print("      请改用 --from-ringdown 复用已有 Step 1 结果")
        sys.exit(2)
    if continuum:
        continuum_targets = kwargs.get("targets")
        if continuum_targets is None:
            kwargs["targets"] = ["CIA"]
        else:
            bad_targets = [
                t for t in continuum_targets
                if (not t.strip("/").split("/")
                    or t.strip("/").split("/")[0] != "CIA")
            ]
            if bad_targets:
                print("错误: --continuum 只能处理 CIA 目录下的数据")
                print("      示例: python main.py --continuum 'CIA/273K/Ar 500Torr'")
                print(f"      无效目标: {', '.join(bad_targets)}")
                sys.exit(2)

    pipeline = CRDSPipeline(**kwargs)
    if type_a_mc:
        pipeline.run_type_a_monte_carlo()
    elif remeasure_report:
        pipeline.generate_remeasure_report()
    elif continuum:
        if from_ringdown:
            pipeline.run_continuum_from_ringdown()
        else:
            pipeline.run_continuum()
    elif n2_only:
        if from_etalon:
            pipeline.run_n2_only_from_etalon()
        elif from_ringdown:
            pipeline.run_n2_only_from_ringdown()
        else:
            pipeline.run_n2_only()
    elif from_etalon:
        pipeline.run_from_etalon()
    elif from_ringdown:
        pipeline.run_from_ringdown()
    else:
        pipeline.run()
