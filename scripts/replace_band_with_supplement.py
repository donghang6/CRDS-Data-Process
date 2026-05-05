#!/usr/bin/env python3
"""Replace one wavenumber band with tau-adjusted files from a supplement directory.

使用说明
--------
这个脚本用于把同级目录中的“补充”数据替换进主数据目录。

默认目录关系:
    主目录:       .../273K/Ar 500Torr
    补充目录:     .../273K/Ar 500Torr 补充
    替换归档目录: .../273K/Ar 500Torr替换

脚本行为:
    1. 从补充目录中选择 --range 指定的波数范围。
    2. 用 --tau-offset 对这些补充文件内容中的衰荡时间整体增加或减少。
       原始数据 txt 的第 1 列为衰荡时间 tau (us)；补充目录中的原始文件不会被修改。
    3. 将主目录中对应替换范围内的旧文件移动到“替换归档目录”。
    4. 将修正后的补充文件写入主目录，文件名波数保持不变。

常用命令:
    # 先 dry-run 预览，不真正改文件
    python scripts/replace_band_with_supplement.py \
      '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \
      --range 9630 9668 \
      --tau-offset 0

    # 确认预览无误后执行
    python scripts/replace_band_with_supplement.py \
      '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \
      --range 9630 9668 \
      --tau-offset 0 \
      --apply

    # 补充数据衰荡时间整体增加 0.02 us 后替换
    python scripts/replace_band_with_supplement.py \
      '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \
      --range 9630 9668 \
      --tau-offset 0.02 \
      --apply

    # 补充数据衰荡时间整体减少 0.02 us 后替换
    python scripts/replace_band_with_supplement.py \
      '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \
      --range 9630 9668 \
      --tau-offset -0.02 \
      --apply

参数说明:
    main_dir          主数据目录，例如 .../273K/Ar 500Torr
    --range A B       从补充目录中选取原始波数 A 到 B，包含端点。
    --tau-offset X    补充文件写入主目录时，第 1 列衰荡时间整体加 X us。
                      X 为正数表示增加衰荡时间，为负数表示减少衰荡时间。
    --tau-column N    衰荡时间所在列，默认 1，即第 1 列。
    --replace-range A B
                      手动指定主目录中要归档替换的波数范围。
                      不写时，默认使用补充数据的最小/最大波数。
    --supplement-dir  手动指定补充目录；不写时默认是 "<main_dir> 补充"。
    --archive-dir     手动指定旧数据归档目录；不写时默认是 "<main_dir>替换"。
    --apply           真正执行移动和复制；不加时只预览。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

USAGE_EPILOG = """
使用示例:
  # 先 dry-run 预览，不真正改文件
  python scripts/replace_band_with_supplement.py \\
    '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \\
    --range 9630 9668 \\
    --tau-offset 0

  # 确认预览无误后执行
  python scripts/replace_band_with_supplement.py \\
    '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \\
    --range 9630 9668 \\
    --tau-offset 0 \\
    --apply

  # 补充数据衰荡时间整体增加 0.02 us 后替换
  python scripts/replace_band_with_supplement.py \\
    '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \\
    --range 9630 9668 \\
    --tau-offset 0.02 \\
    --apply

  # 补充数据衰荡时间整体减少 0.02 us 后替换
  python scripts/replace_band_with_supplement.py \\
    '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/273K/Ar 500Torr' \\
    --range 9630 9668 \\
    --tau-offset -0.02 \\
    --apply

说明:
  默认补充目录是 "<main_dir> 补充"。
  默认旧数据归档目录是 "<main_dir>替换"。
  --apply 不加时只预览；确认输出无误后再加 --apply。
  --replace-range 可手动指定主目录中要移动到归档目录的波数范围。
  --tau-offset 控制衰荡时间 tau 的整体加减，单位 us。
"""


@dataclass(frozen=True)
class WaveFile:
    path: Path
    wavenumber: Decimal


@dataclass(frozen=True)
class MovePlan:
    source: Path
    target: Path


@dataclass(frozen=True)
class CopyPlan:
    source: Path
    target: Path
    wavenumber: Decimal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace a band in a main CRDS raw-data directory using "
            "tau-adjusted files from its supplement directory."
        ),
        epilog=USAGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "main_dir",
        type=Path,
        help="Main data directory, for example '.../273K/Ar 500Torr'.",
    )
    parser.add_argument(
        "--supplement-dir",
        type=Path,
        help="Supplement directory. Defaults to '<main_dir> 补充'.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Archive directory for replaced files. Defaults to '<main_dir>替换'.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        required=True,
        metavar=("START", "END"),
        type=parse_decimal,
        help="Original supplement wavenumber range to use, inclusive.",
    )
    parser.add_argument(
        "--tau-offset",
        "--offset",
        dest="tau_offset",
        type=parse_decimal,
        default=Decimal("0"),
        help=(
            "Tau offset in microseconds applied to the supplement file content "
            "when it is written into the main directory. Use negative values "
            "to decrease tau. --offset is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--tau-column",
        type=int,
        default=1,
        help="1-based column index of tau in each raw txt file. Default: 1.",
    )
    parser.add_argument(
        "--replace-range",
        nargs=2,
        metavar=("START", "END"),
        type=parse_decimal,
        help=(
            "Main-directory wavenumber range to archive. Defaults to the "
            "min/max wavenumber range of the selected supplement files."
        ),
    )
    parser.add_argument(
        "--tau-decimals",
        type=int,
        default=5,
        help="Decimal places used when writing adjusted tau values. Default: 5.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move/copy files. Without this flag, only print a dry run.",
    )
    return parser.parse_args()


def parse_decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"Invalid decimal value: {value}") from exc


def parse_wavenumber(path: Path) -> Decimal | None:
    if path.suffix.lower() != ".txt":
        return None
    try:
        return Decimal(path.stem)
    except InvalidOperation:
        return None


def scan_wave_files(directory: Path) -> list[WaveFile]:
    items: list[WaveFile] = []
    for path in sorted(directory.glob("*.txt")):
        wavenumber = parse_wavenumber(path)
        if wavenumber is not None:
            items.append(WaveFile(path=path, wavenumber=wavenumber))
    return items


def normalize_range(start: Decimal, end: Decimal) -> tuple[Decimal, Decimal]:
    return (start, end) if start <= end else (end, start)


def in_range(value: Decimal, start: Decimal, end: Decimal) -> bool:
    return start <= value <= end


def format_decimal(value: Decimal, decimals: int) -> str:
    quant = Decimal("1").scaleb(-decimals)
    rounded = value.quantize(quant, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def adjust_tau_content(
    source: Path,
    tau_offset: Decimal,
    tau_column: int,
    tau_decimals: int,
) -> str:
    if tau_column < 1:
        raise ValueError("tau_column must be >= 1")

    adjusted_lines: list[str] = []
    for line_number, raw_line in enumerate(source.read_text().splitlines(), start=1):
        if not raw_line.strip():
            adjusted_lines.append(raw_line)
            continue

        columns = raw_line.split()
        index = tau_column - 1
        if index >= len(columns):
            raise ValueError(
                f"{source}: line {line_number} has fewer than {tau_column} columns"
            )

        try:
            tau = Decimal(columns[index])
        except InvalidOperation as exc:
            raise ValueError(
                f"{source}: line {line_number} has invalid tau value: {columns[index]}"
            ) from exc

        columns[index] = format_decimal(tau + tau_offset, tau_decimals)
        adjusted_lines.append("\t".join(columns))

    return "\n".join(adjusted_lines) + "\n"


def unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path

    idx = 2
    while True:
        candidate = path.with_name(f"{path.stem}__{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def build_plans(
    main_dir: Path,
    main_files: list[WaveFile],
    supplement_files: list[WaveFile],
    archive_dir: Path,
    source_range: tuple[Decimal, Decimal],
    replace_range: tuple[Decimal, Decimal] | None,
) -> tuple[list[MovePlan], list[CopyPlan], tuple[Decimal, Decimal]]:
    selected = [
        item
        for item in supplement_files
        if in_range(item.wavenumber, source_range[0], source_range[1])
    ]
    if not selected:
        raise SystemExit("No supplement files were found in the requested range.")

    copy_plan: list[CopyPlan] = []
    seen_targets: set[Path] = set()
    for item in selected:
        target = main_dir / item.path.name
        if target in seen_targets:
            raise SystemExit(f"Duplicate supplement target: {target.name}")
        seen_targets.add(target)
        copy_plan.append(
            CopyPlan(
                source=item.path,
                target=target,
                wavenumber=item.wavenumber,
            )
        )

    if replace_range is None:
        values = [item.wavenumber for item in copy_plan]
        active_replace_range = (min(values), max(values))
    else:
        active_replace_range = replace_range

    replaced = [
        item
        for item in main_files
        if in_range(item.wavenumber, active_replace_range[0], active_replace_range[1])
    ]
    replaced_paths = {item.path for item in replaced}

    for item in copy_plan:
        if item.target.exists() and item.target not in replaced_paths:
            raise SystemExit(
                "Supplement target already exists outside the replacement "
                f"range: {item.target}"
            )

    move_plan = [
        MovePlan(
            source=item.path,
            target=unique_archive_path(archive_dir / item.path.name),
        )
        for item in replaced
    ]
    return move_plan, copy_plan, active_replace_range


def apply_plans(
    archive_dir: Path,
    move_plan: list[MovePlan],
    copy_plan: list[CopyPlan],
    tau_offset: Decimal,
    tau_column: int,
    tau_decimals: int,
) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for item in move_plan:
        item.source.rename(item.target)
    for item in copy_plan:
        adjusted_content = adjust_tau_content(
            source=item.source,
            tau_offset=tau_offset,
            tau_column=tau_column,
            tau_decimals=tau_decimals,
        )
        item.target.write_text(adjusted_content)


def validate_copy_plan(
    copy_plan: list[CopyPlan],
    tau_offset: Decimal,
    tau_column: int,
    tau_decimals: int,
) -> None:
    for item in copy_plan:
        adjust_tau_content(
            source=item.source,
            tau_offset=tau_offset,
            tau_column=tau_column,
            tau_decimals=tau_decimals,
        )


def main() -> None:
    args = parse_args()

    main_dir = args.main_dir.expanduser().resolve()
    supplement_dir = (
        args.supplement_dir.expanduser().resolve()
        if args.supplement_dir
        else main_dir.with_name(f"{main_dir.name} 补充")
    )
    archive_dir = (
        args.archive_dir.expanduser().resolve()
        if args.archive_dir
        else main_dir.with_name(f"{main_dir.name}替换")
    )

    if not main_dir.is_dir():
        raise SystemExit(f"Main directory does not exist: {main_dir}")
    if not supplement_dir.is_dir():
        raise SystemExit(f"Supplement directory does not exist: {supplement_dir}")
    if args.tau_column < 1:
        raise SystemExit("--tau-column must be >= 1")
    if args.tau_decimals < 0:
        raise SystemExit("--tau-decimals must be >= 0")

    source_range = normalize_range(args.range[0], args.range[1])
    replace_range = (
        normalize_range(args.replace_range[0], args.replace_range[1])
        if args.replace_range
        else None
    )

    main_files = scan_wave_files(main_dir)
    supplement_files = scan_wave_files(supplement_dir)
    move_plan, copy_plan, active_replace_range = build_plans(
        main_dir=main_dir,
        main_files=main_files,
        supplement_files=supplement_files,
        archive_dir=archive_dir,
        source_range=source_range,
        replace_range=replace_range,
    )
    try:
        validate_copy_plan(
            copy_plan=copy_plan,
            tau_offset=args.tau_offset,
            tau_column=args.tau_column,
            tau_decimals=args.tau_decimals,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Main directory: {main_dir}")
    print(f"Supplement directory: {supplement_dir}")
    print(f"Archive directory: {archive_dir}")
    print(f"Supplement source range: {source_range[0]} to {source_range[1]}")
    print(f"Tau offset: {args.tau_offset} us")
    print(f"Tau column: {args.tau_column}")
    print(f"Tau decimals: {args.tau_decimals}")
    print(f"Main replacement range: {active_replace_range[0]} to {active_replace_range[1]}")
    print(f"Main files to move into archive: {len(move_plan)}")
    print(f"Supplement files to copy into main: {len(copy_plan)}")

    preview_count = 10
    for item in move_plan[:preview_count]:
        print(f"MOVE {item.source.name} -> {item.target.parent.name}/{item.target.name}")
    for item in copy_plan[:preview_count]:
        print(
            "COPY "
            f"{item.source.parent.name}/{item.source.name} -> {item.target.name} "
            f"(tau {args.tau_offset:+} us)"
        )

    shown = min(len(move_plan), preview_count) + min(len(copy_plan), preview_count)
    remaining = len(move_plan) + len(copy_plan) - shown
    if remaining > 0:
        print(f"... {remaining} more operations")

    if not args.apply:
        print("Dry run only. Re-run with --apply to move/copy files.")
        return

    apply_plans(
        archive_dir=archive_dir,
        move_plan=move_plan,
        copy_plan=copy_plan,
        tau_offset=args.tau_offset,
        tau_column=args.tau_column,
        tau_decimals=args.tau_decimals,
    )
    print("Done.")


if __name__ == "__main__":
    main()
