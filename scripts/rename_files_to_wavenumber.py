#!/usr/bin/env python3
"""Rename CRDS raw data files so the filename keeps only the wavenumber.

Expected input filename:
    "{index} {wavenumber} {YYYYMMDDHHMMSS}.txt"

Example output filename:
    "9334.00538.txt"
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

FILENAME_RE = re.compile(r"^\s*(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d{14})(\.[^.]+)$")


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path


@dataclass(frozen=True)
class DeletePlan:
    source: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename raw CRDS files to keep only the wavenumber."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to process. Subdirectories are included by default.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only print a dry run.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only process files directly inside the directory.",
    )
    parser.add_argument(
        "--conflict",
        choices=("error", "suffix", "keep-first"),
        default="error",
        help=(
            "How to handle duplicate wavenumbers or existing targets. "
            "'error' stops; 'suffix' keeps extra files as wavenumber__2.txt; "
            "'keep-first' keeps the first file and deletes duplicate sources."
        ),
    )
    return parser.parse_args()


def iter_files(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def target_for(path: Path) -> Path | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    wavenumber = match.group(2)
    suffix = match.group(4)
    return path.with_name(f"{wavenumber}{suffix}")


def build_plan(
    files: list[Path], conflict: str
) -> tuple[list[RenamePlan], list[DeletePlan], list[Path]]:
    unmatched: list[Path] = []
    raw_targets: list[Path] = []
    matched: list[tuple[Path, Path]] = []

    for source in files:
        target = target_for(source)
        if target is None:
            unmatched.append(source)
            continue
        if source == target:
            continue
        matched.append((source, target))
        raw_targets.append(target)

    target_counts = Counter(raw_targets)
    existing_conflicts = {
        target
        for _, target in matched
        if target.exists() and target not in {source for source, _ in matched}
    }
    duplicate_targets = {target for target, count in target_counts.items() if count > 1}
    conflicts = existing_conflicts | duplicate_targets

    if conflicts and conflict == "error":
        print("Conflicts found. No files were renamed.")
        for target in sorted(conflicts):
            print(f"  {target}")
        print(
            "Use --conflict suffix to keep duplicates with numeric suffixes, "
            "or --conflict keep-first to delete duplicate sources."
        )
        raise SystemExit(2)

    if conflict == "keep-first":
        rename_plan, delete_plan = build_keep_first_plan(matched)
        return rename_plan, delete_plan, unmatched

    used: set[Path] = set()
    plan: list[RenamePlan] = []
    for source, raw_target in matched:
        target = raw_target
        if conflict == "suffix":
            target = next_available_target(raw_target, used)
        used.add(target)
        plan.append(RenamePlan(source=source, target=target))

    return plan, [], unmatched


def build_keep_first_plan(
    matched: list[tuple[Path, Path]]
) -> tuple[list[RenamePlan], list[DeletePlan]]:
    by_target: dict[Path, list[Path]] = defaultdict(list)
    for source, target in matched:
        by_target[target].append(source)

    rename_plan: list[RenamePlan] = []
    delete_plan: list[DeletePlan] = []
    for target, sources in sorted(by_target.items()):
        sources = sorted(sources)

        if target.exists() and target not in sources:
            for source in sources:
                delete_plan.append(
                    DeletePlan(
                        source=source,
                        reason=f"duplicate of existing {target.name}",
                    )
                )
            continue

        winner = sources[0]
        rename_plan.append(RenamePlan(source=winner, target=target))
        for source in sources[1:]:
            delete_plan.append(
                DeletePlan(source=source, reason=f"duplicate of {target.name}")
            )

    return rename_plan, delete_plan


def next_available_target(target: Path, used: set[Path]) -> Path:
    if target not in used and not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    idx = 2
    while True:
        candidate = target.with_name(f"{stem}__{idx}{suffix}")
        if candidate not in used and not candidate.exists():
            return candidate
        idx += 1


def apply_plan(plan: list[RenamePlan], delete_plan: list[DeletePlan]) -> None:
    for item in plan:
        item.source.rename(item.target)
    for item in delete_plan:
        item.source.unlink()


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    files = iter_files(directory, recursive=not args.no_recursive)
    plan, delete_plan, unmatched = build_plan(files, conflict=args.conflict)

    print(f"Directory: {directory}")
    print(f"Files scanned: {len(files)}")
    print(f"Files matched: {len(plan) + len(delete_plan)}")
    print(f"Files to rename: {len(plan)}")
    print(f"Duplicate files to delete: {len(delete_plan)}")
    print(f"Files skipped: {len(unmatched)}")

    for item in plan[:20]:
        print(f"{item.source.name} -> {item.target.name}")
    shown = min(len(plan), 20)
    for item in delete_plan[: 20 - shown]:
        print(f"{item.source.name} -> DELETE ({item.reason})")
    remaining = len(plan) + len(delete_plan) - 20
    if remaining > 0:
        print(f"... {remaining} more")

    if not args.apply:
        print("Dry run only. Re-run with --apply to rename/delete files.")
        return

    apply_plan(plan, delete_plan)
    print(f"Renamed {len(plan)} files.")
    print(f"Deleted {len(delete_plan)} duplicate files.")


if __name__ == "__main__":
    "python scripts/rename_files_to_wavenumber.py '/Users/donghang/科研/实验数据/氧气连续吸收温度/原始数据初步处理/0/273K' --conflict keep-first --apply"
    main()
