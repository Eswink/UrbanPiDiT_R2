"""Resolve every ``alias.attr`` a script reads off the modules it imports; report the missing ones.

Why this exists. A driver that writes ``curve.EVALUATION_LEADS`` compiles, imports and passes any
test that does not execute the line -- the name ``curve`` is a perfectly good global and only the
*attribute* is absent. The repository already had a guard for the sibling failure (a bare global
that does not exist, ``test_every_driver_global_resolves_before_a_gpu_stage_sees_it``), and that
guard did not catch this one: in the bytecode it is ``LOAD_GLOBAL curve`` followed by
``LOAD_ATTR EVALUATION_LEADS``, so the global resolves and the check passes. Stage C2's driver
carried two such attributes for weeks and both would have fired only after ten hours of training
on each of two arms (decision 0098).

What it reads. The file's own top-level ``from X import Y`` statements give the alias map, so no
configuration is needed and no list can go stale. Only reads of the form ``alias.attr`` where
``alias`` is one of those imports are checked; attribute writes, method calls on locals, and
anything reached through a parameter are out of scope by construction.

What it does not prove. A present attribute can still fail at call time (wrong arity, wrong
types); this tool answers exactly one question -- does the name exist -- because that is the
failure mode that survives import and unit tests alike.

Read-only, standard library only, no network, no imports of the target's own dependencies beyond
what importing that module requires. Exit status is 1 when any attribute is missing.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module_aliases(tree: ast.Module) -> dict[str, str]:
    """Map each local name to the fully qualified object it binds, from the top-level imports.

    Both spellings are covered, because the driver uses both and they resolve differently:
    ``import x.y as z`` binds the module ``x.y``, so ``z.attr`` is a module attribute;
    ``from x import y`` binds whatever ``y`` is, so ``y.attr`` is an attribute of that object.
    Nested ``from x import y`` (a package submodule) needs the submodule import to have happened,
    which importing the target normally does.
    """
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for imported in node.names:
                aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.asname:
                    aliases[imported.asname] = imported.name
                else:
                    root = imported.name.split(".")[0]
                    aliases[root] = root
    return aliases


def attribute_reads(tree: ast.Module, aliases: dict[str, str]) -> list[tuple[str, str, int]]:
    """Every ``alias.attr`` read, as ``(alias, attr, lineno)``; duplicates kept for line numbers."""
    reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id in aliases:
            reads.append((base.id, node.attr, node.lineno))
    return reads


def _resolve(target: str, load):
    """Resolve a dotted target to ``(owner, leaf)``: the object an attribute would be read off.

    ``from x import y`` binds whatever ``y`` is -- a module, a function, a constant -- so
    ``y.attr`` must be checked against that object, not against an importable module named
    ``x.y``. Walking the dotted path and stopping at the first non-module is what makes the
    difference: ``importlib.import_module("x.y")`` raises when ``y`` is a function, which would
    be reported as a failed import rather than as the attribute check it is.
    """
    parts = target.split(".")
    module = load(parts[0])
    walked = parts[0]
    for index, part in enumerate(parts[1:], start=1):
        candidate = f"{walked}.{part}"
        try:
            module = load(candidate)
        except Exception:
            # Not a module: the remaining path is attribute access on what we have so far.
            for rest in parts[index:]:
                if not hasattr(module, rest):
                    return module, rest, False
                module = getattr(module, rest)
            return module, None, True
        walked = candidate
    return module, None, True


def missing_attributes(path: Path, *, importer=None) -> list[dict[str, object]]:
    """Resolve every alias attribute; return one record per missing name, deduplicated by pair."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    load = importer or importlib.import_module
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    aliases = module_aliases(tree)
    seen: set[tuple[str, str]] = set()
    problems: list[dict[str, object]] = []
    for alias, attribute, lineno in attribute_reads(tree, aliases):
        target = aliases[alias]
        if (target, attribute) in seen:
            continue
        seen.add((target, attribute))
        try:
            owner, missing_leaf, ok = _resolve(target, load)
        except Exception as error:  # a target that cannot even import is reported, not skipped
            problems.append({"alias": alias, "attribute": attribute, "module": target,
                             "lineno": lineno, "reason": f"import failed: {error}"})
            continue
        if not ok and missing_leaf is not None:
            problems.append({"alias": alias, "attribute": missing_leaf, "module": target,
                             "lineno": lineno, "reason": "attribute is absent from the object"})
            continue
        if not hasattr(owner, attribute):
            problems.append({"alias": alias, "attribute": attribute, "module": target,
                             "lineno": lineno, "reason": "attribute is absent from the module"})
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="script(s) whose alias reads to resolve")
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        problems = missing_attributes(path)
        if problems:
            failed = True
            for problem in problems:
                print(f"{path}:{problem['lineno']}: {problem['alias']}.{problem['attribute']} "
                      f"({problem['module']}): {problem['reason']}")
        else:
            print(f"{path}: every imported-module attribute resolves")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
