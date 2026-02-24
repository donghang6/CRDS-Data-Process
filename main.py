"""CRDS 数据处理 - 主程序入口

运行方式:
    python main.py                              # 使用默认配置处理 data/raw
    python main.py --data-dir data/raw          # 指定数据目录
    python main.py --config config/default.yaml # 指定配置文件
    python main.py --skip-mats                  # 跳过 MATS 拟合
"""

from crds_process.pipeline import run_pipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CRDS 数据处理流水线")
    parser.add_argument("--data-dir", default="data/raw", help="原始数据目录 (默认: data/raw)")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument("--output-dir", default=None, help="输出目录 (默认从配置读取)")
    parser.add_argument("--skip-mats", action="store_true", help="跳过 MATS 拟合步骤")
    args = parser.parse_args()

    print("=" * 60)
    print("  CRDS 数据处理流水线")
    print("  衰荡时间 → 吸收系数 → 基线扣除 → MATS 光谱拟合")
    print("=" * 60)
    print(f"  数据目录: {args.data_dir}")
    print(f"  配置文件: {args.config}")
    print(f"  跳过MATS: {args.skip_mats}")
    print("=" * 60)
    print()

    spectrum_df = run_pipeline(
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        skip_mats=args.skip_mats,
    )

    print()
    print("光谱数据预览:")
    print(spectrum_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

