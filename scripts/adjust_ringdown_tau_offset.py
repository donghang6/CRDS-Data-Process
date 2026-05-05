#!/usr/bin/env python3
"""Adjust all ring-down times in a directory by a constant offset.

使用说明
--------
这个脚本用于把指定目录下所有原始衰荡数据 txt 文件中的衰荡时间整体加上
或减去同一个数。原始数据文件默认第 1 列为衰荡时间 tau，单位 us。

默认只 dry-run 预览，不会改文件；确认无误后加 --apply 才会真正写回。

常用命令:
    # 预览：所有 tau 增加 0.02 us
    python scripts/adjust_ringdown_tau_offset.py \
      '/path/to/Ar 500Torr' \
      --offset 0.02

    # 执行：所有 tau 增加 0.02 us
    python scripts/adjust_ringdown_tau_offset.py \
      '/path/to/Ar 500Torr' \
      --offset 0.02 \
      --apply

    # 执行：所有 tau 减少 0.02 us
    python scripts/adjust_ringdown_tau_offset.py \
      '/path/to/Ar 500Torr' \
      --offset -0.02 \
      --apply

    # 递归处理子目录
    python scripts/adjust_ringdown_tau_offset.py \
      '/path/to/273K' \
      --offset -0.02 \
      --recursive \
      --apply

参数说明:
    directory       要处理的目录。
    --offset X      tau 整体加 X us；X 为负数时表示减少。
    --tau-column N  tau 所在列，默认 1，即第 1 列。
    --decimals N    写回 tau 时保留的小数位数，默认 5。
    --recursive     递归处理子目录。
    --apply         真正修改文件；不加时只预览。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path


@dataclass(frozen=True)
class FilePlan:
    path: Path
    n_rows: int
    first_before: Decimal | None
    first_after: Decimal | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a constant offset to all ring-down times in txt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("directory", type=Path, help="Directory containing txt files.")
    parser.add_argument(
        "--offset",
        required=True,
        type=parse_decimal,
        help="Tau offset in us. Positive adds; negative subtracts.",
    )
    parser.add_argument(
        "--tau-column",
        type=int,
        default=1,
        help="1-based tau column index. Default: 1.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=5,
        help="Decimal places for adjusted tau values. Default: 5.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process txt files in subdirectories too.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag, only print a dry run.",
    )
    return parser.parse_args()


def parse_decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"Invalid decimal value: {value}") from exc


def iter_txt_files(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def format_decimal(value: Decimal, decimals: int) -> str:
    quant = Decimal("1").scaleb(-decimals)
    rounded = value.quantize(quant, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def adjust_file_content(
    path: Path,
    offset: Decimal,
    tau_column: int,
    decimals: int,
) -> tuple[str, FilePlan]:
    adjusted_lines: list[str] = []
    n_rows = 0
    first_before: Decimal | None = None
    first_after: Decimal | None = None
    tau_index = tau_column - 1

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            adjusted_lines.append(raw_line)
            continue

        columns = raw_line.split()
        if tau_index >= len(columns):
            raise ValueError(
                f"{path}: line {line_number} has fewer than {tau_column} columns"
            )

        try:
            tau = Decimal(columns[tau_index])
        except InvalidOperation as exc:
            raise ValueError(
                f"{path}: line {line_number} has invalid tau value: {columns[tau_index]}"
            ) from exc

        adjusted_tau = tau + offset
        if first_before is None:
            first_before = tau
            first_after = adjusted_tau
        columns[tau_index] = format_decimal(adjusted_tau, decimals)
        adjusted_lines.append("\t".join(columns))
        n_rows += 1

    content = "\n".join(adjusted_lines)
    if adjusted_lines:
        content += "\n"

    return content, FilePlan(
        path=path,
        n_rows=n_rows,
        first_before=first_before,
        first_after=first_after,
    )


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    if args.tau_column < 1:
        raise SystemExit("--tau-column must be >= 1")
    if args.decimals < 0:
        raise SystemExit("--decimals must be >= 0")

    files = iter_txt_files(directory, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No .txt files found in: {directory}")

    plans: list[FilePlan] = []
    adjusted_contents: list[tuple[Path, str]] = []
    for path in files:
        try:
            content, plan = adjust_file_content(
                path=path,
                offset=args.offset,
                tau_column=args.tau_column,
                decimals=args.decimals,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        plans.append(plan)
        adjusted_contents.append((path, content))

    print(f"Directory: {directory}")
    print(f"Files scanned: {len(files)}")
    print(f"Tau column: {args.tau_column}")
    print(f"Offset: {args.offset} us")
    print(f"Decimals: {args.decimals}")
    print(f"Recursive: {args.recursive}")
    print(f"Total tau values to adjust: {sum(p.n_rows for p in plans)}")

    for plan in plans[:10]:
        if plan.first_before is None:
            preview = "empty/no data rows"
        else:
            preview = f"{plan.first_before} -> {plan.first_after}"
        print(f"{plan.path.name}: rows={plan.n_rows}, first tau {preview}")
    if len(plans) > 10:
        print(f"... {len(plans) - 10} more files")

    if not args.apply:
        print("Dry run only. Re-run with --apply to modify files.")
        return

    for path, content in adjusted_contents:
        path.write_text(content)
    print(f"Updated {len(adjusted_contents)} files.")


if __name__ == "__main__":
    main()
