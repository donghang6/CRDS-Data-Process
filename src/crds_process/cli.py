"""命令行入口

用法:
    crds-process run --data-dir data/raw --config config/default.yaml
    crds-process run --data-dir data/raw --skip-mats
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="crds-process",
        description="CRDS 数据处理流水线：从衰荡时间到光谱参数拟合",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # --- run 子命令 ---
    run_parser = subparsers.add_parser("run", help="执行完整处理流水线")
    run_parser.add_argument(
        "--data-dir", required=True, help="原始数据目录路径"
    )
    run_parser.add_argument(
        "--config", default=None, help="配置文件路径 (默认: config/default.yaml)"
    )
    run_parser.add_argument(
        "--output-dir", default=None, help="输出目录路径"
    )
    run_parser.add_argument(
        "--skip-mats", action="store_true", help="跳过 MATS 拟合步骤"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        from crds_process.pipeline import run_pipeline

        run_pipeline(
            data_dir=args.data_dir,
            config_path=args.config,
            output_dir=args.output_dir,
            skip_mats=args.skip_mats,
        )


if __name__ == "__main__":
    main()

