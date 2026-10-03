"""Opaque whole-owned-root file pins; selected validated artifacts remain mandatory."""
from __future__ import annotations

import os
from pathlib import Path

from .r7_v2_protocol import safe_output, sha256_file, write_path

EXCLUDED_RECURSIVE_OR_FUTURE = ("artifact_manifest.json", "attempt.json", "execution_attempt.json")


def _require_publication_intact(output):
    marker = output / "publication_failure.json"
    if marker.exists() or marker.is_symlink():
        raise ValueError("authoritative publication failure; acceptance refused")


def _regular_inventory(output):
    """Never follow links or open special files, including otherwise excluded names."""
    output = safe_output(output)
    _require_publication_intact(output)
    if not output.is_dir():
        raise ValueError("owned inventory root must be an existing directory")
    files, pending = set(), [output]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                relative = Path(entry.path).relative_to(output).as_posix()
                if entry.is_symlink():
                    raise ValueError(f"owned inventory symlink forbidden: {relative}")
                if entry.is_dir(follow_symlinks=False):
                    if relative in EXCLUDED_RECURSIVE_OR_FUTURE:
                        raise ValueError(f"excluded receipt must not be a directory: {relative}")
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if relative not in EXCLUDED_RECURSIVE_OR_FUTURE:
                        files.add(relative)
                else:
                    raise ValueError(f"owned inventory special file forbidden: {relative}")
    _require_publication_intact(output)
    return files


def pin_inventory(output, required_paths):
    """Hash every regular file, requiring the already validated selected-path subset."""
    output = safe_output(output)
    actual = _regular_inventory(output)
    required = {write_path(path, output).relative_to(output).as_posix() for path in required_paths}
    if not required <= actual:
        raise ValueError(f"required validated artifacts missing from inventory: {sorted(required - actual)}")
    pins = {name: sha256_file(write_path(output / name, output)) for name in sorted(actual)}
    if _regular_inventory(output) != actual:
        raise ValueError("owned inventory changed while pinning")
    return pins


def verify_inventory(output, expected):
    """After manifest publication, require exact unchanged file names AND opaque bytes."""
    output = safe_output(output)
    actual = _regular_inventory(output)
    if actual != set(expected):
        raise ValueError("owned inventory changed after manifest publication")
    for name in sorted(actual):
        if sha256_file(write_path(output / name, output)) != expected[name]:
            raise ValueError(f"owned inventory bytes changed after manifest publication: {name}")
    if _regular_inventory(output) != actual:
        raise ValueError("owned inventory changed during final verification")
    return expected
