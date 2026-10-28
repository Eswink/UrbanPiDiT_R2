"""Build a round's execution inventory from its import closure instead of a directory glob.

Decision 0067: a driver must bind the code it actually executes into ``protocol.json``,
not "whatever ``.py`` files happen to sit under ``training/``, ``model/``, ``data/`` right
now". The distinction is not stylistic. R1 froze 195 such entries, 182 of them produced by
a live ``rglob``; when an unrelated round added two modules to ``data/``, a third seed that
had not started yet failed ``validate_protocol`` in 20.76 seconds and the round could only
be finished by replaying its archived ``code.zip`` (decision 0065). Nine hours of GPU time
were spent before the problem was even discoverable.

This module computes the closure a new driver should declare:

    inventory = explicit_execution_files("scripts/study_r7_next_round.py", extra=ARCHIVED)
    # -> sorted ["data/...", "model/...", "scripts/...", "training/..."]

``extra`` names files the closure cannot see -- an archived ``protocol.json`` read at run
time, or a module loaded through ``importlib``. Decision 0067 requires those to be written
out literally, so the parameter exists rather than a silent gap.

**Known limit, stated rather than hidden:** only static ``import`` / ``from ... import``
statements are followed. A module reached through ``importlib.import_module``, a string, or
a plugin registry is invisible here and MUST be listed in ``extra``. That gap is the reason
decision 0067 keeps the per-file SHA256 comparison as well: the closure says *which* files,
the digests say *that they have not changed*, and neither substitutes for the other.

Nothing here touches an existing round. R1's driver keeps its glob; retrofitting it would
change its frozen ``execution_files_sha256`` and, under decision 0045's reuse whitelist,
invalidate the seeds that already finished.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Package directories whose ``.py`` files can be reached from a driver. Kept narrow on
#: purpose: ``tools/`` and ``tests/`` never enter a round's identity surface, and the
#: archived snapshots are read-only references that must never be imported.
PACKAGES = ("data", "model", "training", "scripts")

#: Directories whose contents are archived references, never executed.
EXCLUDED_PARTS = ("legacy_v531", "legacy_v531_full", "legacy_v6")


def module_index(repo: Path) -> dict[str, Path]:
    """Map dotted module names to files for every importable ``.py`` under ``repo``."""
    index: dict[str, Path] = {}
    for package in PACKAGES:
        root = Path(repo) / package
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            relative = path.relative_to(repo)
            parts = list(relative.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            index[".".join(parts)] = path
    return index


def _statement_imports(tree: ast.AST) -> Iterable[tuple[int, str, tuple[str, ...]]]:
    """Yield ``(level, module, names)`` for every import in the file, in source order."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            yield node.level or 0, node.module or "", tuple(alias.name for alias in node.names)


def _resolve(name: str, index: dict[str, Path]) -> list[str]:
    """A dotted import pulls in the module and each of its parent packages."""
    parts = name.split(".")
    return [".".join(parts[:count]) for count in range(len(parts), 0, -1)
            if ".".join(parts[:count]) in index]


def _relative_base(module: str, level: int) -> str:
    """The package a ``level``-deep relative import resolves against."""
    parts = module.split(".")
    if level <= 1:
        return ".".join(parts[:-1])
    keep = len(parts) - level
    return ".".join(parts[:keep]) if keep > 0 else ""


def import_closure(entry: str, *, repo: Path | None = None) -> list[str]:
    """Every repository module reachable from ``entry`` by static imports, sorted.

    ``entry`` is a repository-relative path to a ``.py`` file. The entry itself is always
    included. Raises ``FileNotFoundError`` for an entry that is not on disk, because a
    silently empty closure would declare an inventory that covers nothing.
    """
    base = Path(repo or REPO_ROOT)
    entry_path = base / entry
    if not entry_path.is_file():
        raise FileNotFoundError(f"entry module is not on disk: {entry_path}")
    index = module_index(base)
    by_path = {path.resolve(): name for name, path in index.items()}

    entry_name = by_path.get(entry_path.resolve())
    if entry_name is None:
        raise FileNotFoundError(
            f"{entry} is not under any of {PACKAGES}; a closure cannot be rooted outside them")

    seen: set[str] = set()
    queue = [entry_name]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        path = index.get(module)
        if path is None:
            continue
        seen.add(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for level, name, names in _statement_imports(tree):
            if level:
                prefix = _relative_base(module, level)
                if not prefix:
                    continue
                candidates = [f"{prefix}.{name}"] if name else []
                if names:
                    candidates += [f"{prefix}.{name}.{alias}" for alias in names]
                if not name and not names:
                    candidates = [prefix]
            else:
                candidates = _resolve(name, index)
                if names:
                    candidates += [f"{name}.{alias}" for alias in names
                                   if f"{name}.{alias}" in index]
            for candidate in candidates:
                if candidate in index and candidate not in seen:
                    queue.append(candidate)
    return sorted(str(index[name].relative_to(base)) for name in seen)


def explicit_execution_files(entry: str, *, extra: Iterable[str] = (),
                            repo: Path | None = None) -> list[str]:
    """The declared inventory for a new round: closure plus named extras, sorted.

    Everything must be a repository-relative path that exists; ``extra`` is not a way to
    smuggle in a file that is missing, it is how a deliberately declared non-imported file
    (an archived ``protocol.json``) gets in.
    """
    base = Path(repo or REPO_ROOT)
    names = set(import_closure(entry, repo=base))
    for name in extra:
        if not (base / name).is_file():
            raise FileNotFoundError(f"declared extra is not on disk: {name}")
        names.add(str(name))
    return sorted(names)


def inventory_faults(declared: Iterable[str], entry: str, *,
                     extra: Iterable[str] = (), repo: Path | None = None) -> list[str]:
    """Both halves of decision 0067's invariant, as a fault list.

    Missing entries mean the declared inventory does not cover code the round executes --
    an identity gap, which is worse than the over-wide glob it replaces. Extra entries mean
    the declaration names code the closure cannot reach; that is legal only when the name
    was passed in ``extra``, otherwise it is a stale hand-written line.
    """
    base = Path(repo or REPO_ROOT)
    observed = set(explicit_execution_files(entry, extra=extra, repo=base))
    stated = {str(name) for name in declared}
    faults: list[str] = []
    for name in sorted(observed - stated):
        faults.append(f"closure module not declared: {name}")
    for name in sorted(stated - observed):
        faults.append(f"declared module is not in the closure: {name}")
    return faults
