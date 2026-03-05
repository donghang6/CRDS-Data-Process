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

    # 跳过 Step 1/2, 直接从已有的去除标准具数据开始执行 Step 3~5
    python main.py --from-etalon
    python main.py --from-etalon O2/9386.2076
"""

import sys

from crds_process.pipeline import CRDSPipeline


def _parse_args(argv: list[str]) -> dict:
    """解析命令行参数

    Returns
    -------
    dict
        CRDSPipeline 构造参数
    """
    targets = []
    multi_fit_pressures: dict[str, list[str]] = {}
    auto_optimize = False
    min_pressures = 3
    from_etalon = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--from-etalon":
            from_etalon = True
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
                    print(f"  格式应为: 气体类型/跃迁=压力1,压力2,...")
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
        "auto_optimize_pressures": auto_optimize,
        "min_multi_pressures": min_pressures,
        "_from_etalon": from_etalon,
    }


if __name__ == "__main__":
    kwargs = _parse_args(sys.argv[1:])
    from_etalon = kwargs.pop("_from_etalon")
    pipeline = CRDSPipeline(**kwargs)
    if from_etalon:
        pipeline.run_from_etalon()
    else:
        pipeline.run()
