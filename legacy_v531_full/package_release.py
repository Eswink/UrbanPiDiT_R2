#!/usr/bin/env python3
"""UrbanPiDiT V5.3.1 MorphoProcessDiT reviewer-evidence 打包脚本。

默认只打包本脚本所在的历史主开发目录，排除模型权重、checkpoint、缓存、
日志、实验输出和常见临时产物。支持 dry-run 预览。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from urbanpidit_version import PACKAGE_DIRECTORY_NAME, RELEASE_NAME, RELEASE_SLUG, VERSION
except ImportError:  # pragma: no cover
    PACKAGE_DIRECTORY_NAME = "UrbanPiDiT_V4_with_baselines"
    RELEASE_NAME = "UrbanPiDiT-V5.3.1-MorphoProcessDiT-ReviewerEvidence"
    RELEASE_SLUG = "urbanpidit_v531_morphoprocessdit_reviewer_evidence"
    VERSION = "5.3.1"


PROJECT_NAME = PACKAGE_DIRECTORY_NAME
DEFAULT_EXCLUDE_DIRS = {
    ".cursor",
    ".idea",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".trae",
    ".vscode",
    "__pycache__",
    "checkpoints",
    "lightning_logs",
    "logs",
    "outputs",
    "wandb",
}
DEFAULT_EXCLUDE_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".onnx",
    ".h5",
    ".hdf5",
    ".pkl",
    ".pickle",
    ".joblib",
}
DEFAULT_EXCLUDE_PATTERNS = {
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.zip",
    "*.7z",
    "*.rar",
    "*.tmp",
    "*.temp",
    "*.log",
    ".DS_Store",
}


@dataclass(frozen=True)
class PackagePlan:
    source_root: Path
    archive_path: Path
    included_files: List[Path]
    skipped_files: Dict[str, List[str]]
    total_bytes: int


def parse_csv_values(values: Sequence[str]) -> List[str]:
    """解析可重复传入的逗号分隔参数。"""

    parsed: List[str] = []
    for raw in values:
        parsed.extend(item.strip() for item in str(raw).split(",") if item.strip())
    return parsed


def normalize_suffix(value: str) -> str:
    """统一后缀格式。"""

    item = str(value).strip().lower()
    if not item:
        return item
    return item if item.startswith(".") else f".{item}"


def should_skip_dir(rel_dir: Path, exclude_dirs: set[str]) -> Tuple[bool, str]:
    """判断目录是否应跳过。"""

    parts = set(rel_dir.parts)
    matched = sorted(parts & exclude_dirs)
    if matched:
        return True, f"excluded_dir:{matched[0]}"
    return False, ""


def should_skip_file(
    rel_path: Path,
    *,
    exclude_dirs: set[str],
    exclude_suffixes: set[str],
    exclude_patterns: set[str],
) -> Tuple[bool, str]:
    """判断文件是否应从包中排除。"""

    for part in rel_path.parts[:-1]:
        if part in exclude_dirs:
            return True, f"excluded_dir:{part}"

    suffixes = [suffix.lower() for suffix in rel_path.suffixes]
    for suffix in suffixes:
        if suffix in exclude_suffixes:
            return True, f"excluded_suffix:{suffix}"

    name = rel_path.name
    rel_posix = rel_path.as_posix()
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_posix, pattern):
            return True, f"excluded_pattern:{pattern}"

    return False, ""


def collect_files(
    source_root: Path,
    *,
    archive_path: Path,
    exclude_dirs: set[str],
    exclude_suffixes: set[str],
    exclude_patterns: set[str],
) -> Tuple[List[Path], Dict[str, List[str]], int]:
    """生成打包文件清单。"""

    included: List[Path] = []
    skipped: Dict[str, List[str]] = {}
    total_bytes = 0
    archive_resolved = archive_path.resolve()

    for path in sorted(source_root.rglob("*")):
        rel_path = path.relative_to(source_root)

        if path.is_dir():
            skip, reason = should_skip_dir(rel_path, exclude_dirs)
            if skip:
                skipped.setdefault(reason, []).append(rel_path.as_posix() + "/")
            continue

        if not path.is_file():
            skipped.setdefault("not_regular_file", []).append(rel_path.as_posix())
            continue

        if path.resolve() == archive_resolved:
            skipped.setdefault("archive_self", []).append(rel_path.as_posix())
            continue

        skip, reason = should_skip_file(
            rel_path,
            exclude_dirs=exclude_dirs,
            exclude_suffixes=exclude_suffixes,
            exclude_patterns=exclude_patterns,
        )
        if skip:
            skipped.setdefault(reason, []).append(rel_path.as_posix())
            continue

        included.append(path)
        total_bytes += path.stat().st_size

    return included, skipped, total_bytes


def build_archive_name(source_root: Path, fmt: str) -> str:
    """生成默认包名。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "tar.gz" if fmt == "tar.gz" else "zip"
    return f"{RELEASE_SLUG}_{timestamp}.{suffix}"


def build_package_plan(
    source_root: Path,
    *,
    output_dir: Path,
    archive_name: str | None,
    fmt: str,
    exclude_dirs: Iterable[str],
    exclude_suffixes: Iterable[str],
    exclude_patterns: Iterable[str],
) -> PackagePlan:
    """构建打包计划。"""

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    name = archive_name or build_archive_name(source_root, fmt)
    archive_path = output_dir / name

    included, skipped, total_bytes = collect_files(
        source_root,
        archive_path=archive_path,
        exclude_dirs={str(item) for item in exclude_dirs},
        exclude_suffixes={normalize_suffix(item) for item in exclude_suffixes},
        exclude_patterns={str(item) for item in exclude_patterns},
    )
    return PackagePlan(
        source_root=source_root,
        archive_path=archive_path,
        included_files=included,
        skipped_files=skipped,
        total_bytes=total_bytes,
    )


def write_zip(plan: PackagePlan) -> None:
    """写入 zip 包。"""

    plan.archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(plan.archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in plan.included_files:
            rel_path = path.relative_to(plan.source_root)
            archive.write(path, arcname=f"{plan.source_root.name}/{rel_path.as_posix()}")


def write_tar_gz(plan: PackagePlan) -> None:
    """写入 tar.gz 包。"""

    plan.archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(plan.archive_path, mode="w:gz") as archive:
        for path in plan.included_files:
            rel_path = path.relative_to(plan.source_root)
            archive.add(path, arcname=f"{plan.source_root.name}/{rel_path.as_posix()}")


def write_manifest(plan: PackagePlan, manifest_path: Path) -> None:
    """写入打包清单。"""

    payload = {
        "project_name": RELEASE_NAME,
        "version": VERSION,
        "release_slug": RELEASE_SLUG,
        "source_root": str(plan.source_root),
        "archive_path": str(plan.archive_path),
        "included_count": len(plan.included_files),
        "total_bytes": plan.total_bytes,
        "included_files": [path.relative_to(plan.source_root).as_posix() for path in plan.included_files],
        "skipped_files": plan.skipped_files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_plan(plan: PackagePlan, *, max_list: int) -> None:
    """打印打包计划摘要。"""

    print(f"source_root: {plan.source_root}")
    print(f"archive_path: {plan.archive_path}")
    print(f"included_files: {len(plan.included_files)}")
    print(f"total_bytes: {plan.total_bytes}")
    print(f"skipped_groups: {len(plan.skipped_files)}")

    if max_list <= 0:
        return

    print("\ninclude preview:")
    for path in plan.included_files[:max_list]:
        print(f"  + {path.relative_to(plan.source_root).as_posix()}")
    remaining = len(plan.included_files) - max_list
    if remaining > 0:
        print(f"  ... {remaining} more")

    print("\nskip preview:")
    shown = 0
    for reason, paths in sorted(plan.skipped_files.items()):
        for rel_path in paths[: max(1, max_list - shown)]:
            print(f"  - [{reason}] {rel_path}")
            shown += 1
            if shown >= max_list:
                return


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=f"Package {RELEASE_NAME} without model weights/checkpoints")
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default="tar.gz",
        help="archive format; default: tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "release_packages",
        help="output directory; default: ../release_packages",
    )
    parser.add_argument("--name", type=str, default=None, help="archive file name; default uses timestamp")
    parser.add_argument("--dry-run", action="store_true", help="only print package plan, do not write archive")
    parser.add_argument("--manifest", type=Path, default=None, help="optional JSON manifest output path")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="extra directory name to exclude; can be repeated or comma-separated",
    )
    parser.add_argument(
        "--exclude-suffix",
        action="append",
        default=[],
        help="extra file suffix to exclude, e.g. .npy; can be repeated or comma-separated",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="extra fnmatch pattern to exclude; can be repeated or comma-separated",
    )
    parser.add_argument("--preview", type=int, default=20, help="number of included/skipped items to preview")
    return parser.parse_args()


def main() -> None:
    """执行打包流程。"""

    args = parse_args()
    source_root = Path(__file__).resolve().parent
    if source_root.name != PROJECT_NAME:
        raise RuntimeError(f"This script must live inside {PROJECT_NAME}, got {source_root}")

    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(parse_csv_values(args.exclude_dir))
    exclude_suffixes = DEFAULT_EXCLUDE_SUFFIXES | {normalize_suffix(item) for item in parse_csv_values(args.exclude_suffix)}
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS | set(parse_csv_values(args.exclude_pattern))

    plan = build_package_plan(
        source_root,
        output_dir=args.output_dir,
        archive_name=args.name,
        fmt=args.format,
        exclude_dirs=exclude_dirs,
        exclude_suffixes=exclude_suffixes,
        exclude_patterns=exclude_patterns,
    )
    print_plan(plan, max_list=int(args.preview))

    if args.manifest is not None:
        write_manifest(plan, args.manifest)
        print(f"manifest_written: {args.manifest.resolve()}")

    if args.dry_run:
        print("dry_run: archive not written")
        return

    if args.format == "zip":
        write_zip(plan)
    else:
        write_tar_gz(plan)
    print(f"archive_written: {plan.archive_path}")


if __name__ == "__main__":
    main()