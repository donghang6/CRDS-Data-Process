"""CRDS 数据处理 — 完整五步流水线

运行方式:
    python main.py                              # 处理全部数据
    python main.py O2/9386.2076                  # 仅处理 O2 下 9386.2076 跃迁
    python main.py O2/9386.2076/100Torr          # 仅处理 O2/9386.2076 的 100Torr
    python main.py O2/9386.2076 O2_N2/9386.2076  # 同时处理多个目标

    # 指定参与多光谱联合拟合的压力 (跳过自动筛选)
    python main.py --pressures O2/9386.2076=100Torr,200Torr,300Torr
    python main.py O2/9386.2076 --pressures O2/9386.2076=100Torr,200Torr
"""

import sys

from crds_process.pipeline import CRDSPipeline


def _parse_args(argv: list[str]) -> tuple[list[str] | None, dict[str, list[str]] | None]:
    """解析命令行参数

    Returns
    -------
    targets : list[str] | None
        目标列表 (位置参数)
    multi_fit_pressures : dict[str, list[str]] | None
        多光谱联合拟合指定压力
    """
    targets = []
    multi_fit_pressures: dict[str, list[str]] = {}

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--pressures", "-p"):
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
        else:
            targets.append(arg)
            i += 1

    return targets or None, multi_fit_pressures or None


if __name__ == "__main__":
    targets, multi_fit_pressures = _parse_args(sys.argv[1:])
    pipeline = CRDSPipeline(targets=targets, multi_fit_pressures=multi_fit_pressures)
    pipeline.run()
