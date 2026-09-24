#!/usr/bin/env python3
"""Read-only convention checker for UrbanPiDiT-R2.

Checks the machine-verifiable subset of docs/rules/*.md. Never writes to the
repository. Standard library only, so it runs in a bare CI job.

Run:
    python tools/check_conventions.py                  # blocking rules only
    python tools/check_conventions.py --report         # include C-class reporting
    python tools/check_conventions.py --rule R-013     # one rule
    python tools/check_conventions.py --root /tmp/x    # scan another tree

Exit codes: 0 = no blocking violation, 1 = blocking violation or UNKNOWN,
2 = usage error.

Design notes:
  * BLOCKING rules are A/B class: measured at 0 violations, or a handful with
    an explicitly listed exception path each. Only these gate a commit.
  * REPORT rules are C class: the project violates them widely and deliberately
    (dense numeric one-liners, large model constructors). They are printed for
    visibility and must NOT block, or the gate would be red on day one and then
    ignored. See docs/rules/MIGRATION.md.
  * A check that cannot decide (no git metadata) reports UNKNOWN and exits
    non-zero rather than silently passing.

Known limitations:
  * Text/AST heuristics, not a type checker. Rules that cannot be decided
    mechanically are absent; see RULES_NOT_MECHANISED.
  * Exceptions are fixed path lists, not inference.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

PRUNE_DIRS = {
    ".git", ".mimosa", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".ipynb_checkpoints", "node_modules", "outputs", ".venv",
}
ARCHIVAL_PREFIXES = (
    "legacy_v531_full/", "data/legacy_v531/", "model/legacy_v531/", "legacy_v6/",
)# R-002: path tokens that identify an archived snapshot inside a string operand.
# Kept separate from ARCHIVAL_PREFIXES because a write can target an archive via
# any of these spellings, and new archives must be added in exactly one place.
ARCHIVE_PATH_TOKENS = ("legacy_v531", "legacy_v6")

# R-013/R-014/R-015/R-018/R-019/R-020/R-021/R-022/R-023 scope.
CODE_SCOPES = ("data/", "model/", "training/", "scripts/", "tests/")
EXEMPT_FROM_ANNOTATIONS = {
    "data/__init__.py", "data/download/__init__.py", "model/__init__.py",
    "model/layers/__init__.py", "training/__init__.py", "train.py",
}
# R-016: outbound clients are only allowed in the download layer.
NETWORK_MODULES = {
    "requests", "urllib", "urllib.request", "urllib.error", "urllib.parse",
    "socket", "http.client", "fsspec", "s3fs", "gcsfs", "aiohttp", "httpx",
}
NETWORK_SCOPE_PREFIXES = ("data/", "model/", "training/")
NETWORK_ALLOWED_PREFIXES = ("data/download/",)

# R-006 scope: bounded study and pilot entry points.
STUDY_GLOBS = ("training/r7_*study.py", "training/r7_*control.py", "scripts/real_r7_*.py")
TRAINING_CALL_NAMES = {
    "run_local_updates", "evaluate_local", "run_study", "run_continuous_control",
    "run_extended_control", "run_spatial_study", "run_baseline_study",
}
PROTOCOL_WRITE = re.compile(r"protocol")

# R-010 scope.
CI_WORKFLOW_DIR = ".github/workflows"
CI_STUDY_EVIDENCE = re.compile(r"r7_\w*(study|control)|study_r7_|real_r7_\w*pilot")

ABSOLUTE_PATH = re.compile(r"""['"](/(?:home|data|mnt|media|opt|srv|scratch)/|[A-Za-z]:\\\\)""")
OS_PATH_ALLOWED = {"os.path.relpath"}
LINE_LEN_HARD = 200
LINE_LEN_TARGET = 120
FUNC_BODY_MAX = 100
FILE_LOC_MAX = 400
NESTING_MAX = 5
PARAM_MAX = 8

# R-009: recorded strength of the test suite. Update deliberately when tests are
# intentionally added or restructured, and note it in docs/rules/CHANGELOG.md.
TEST_FUNCTION_BASELINE = 320
ASSERT_BASELINE = 613
# R-027: how many recent commits to sample for message convention.
COMMIT_SAMPLE_SIZE = 30
CONVENTIONAL_COMMIT = re.compile(r"^[a-z]+(\([^)]*\))?!?:\s")

# ---------------------------------------------------------------------------
# Declared exception lists. Every entry is an exact path; no globs, no
# directory-level amnesty. Each one is explained in docs/rules/.
# ---------------------------------------------------------------------------
# R-006: pilots written before the protocol-first convention (#43/#44/#45).
PROTOCOL_EXCEPTIONS = {
    "scripts/real_r7_bounded_smoke.py",
    "scripts/real_r7_pressure_pilot.py",
    "scripts/real_r7_surface_pilot.py",
}
# R-010: same three pilot workflows, plus the offline replay that reuses an
# already-archived artifact instead of producing a new one.
CI_ARCHIVE_EXCEPTIONS = {
    ".github/workflows/r7-pressure-pilot.yml",
    ".github/workflows/r7-pressure-replay.yml",
    ".github/workflows/r7-surface-pilot.yml",
}
# R-030: studies that predate the internal wall-clock deadline convention.
DEADLINE_EXCEPTIONS = {
    "training/r7_baseline_study.py",
    "training/r7_cpu_study.py",
}

BLOCKING_RULES = (
    "R-001", "R-002", "R-004", "R-005", "R-006", "R-007", "R-008", "R-010",
    "R-012", "R-013", "R-014", "R-015", "R-016", "R-017", "R-018",
    "R-024", "R-025", "R-028", "R-029", "R-031",
    "R-032", "R-033", "R-036", "R-037",
    "R-038", "R-039", "R-040", "R-041", "R-042", "R-043", "R-044",
    "R-045", "R-046", "R-047",
)
REPORT_RULES = (
    "R-009", "R-019", "R-019b", "R-020", "R-021", "R-022", "R-023",
    "R-027", "R-030", "R-035", "R-048",
)

RULES_NOT_MECHANISED = (
    "R-003", "R-011", "R-026",
)


class Finding:
    __slots__ = ("rule", "path", "line", "detail", "tolerated")

    def __init__(self, rule: str, path: str, line: int, detail: str, tolerated: bool = False):
        self.rule, self.path, self.line = rule, path, line
        self.detail, self.tolerated = detail, tolerated


def iter_py(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            yield rel, path


def is_archival(rel: str) -> bool:
    return rel.startswith(ARCHIVAL_PREFIXES)


def read(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def parse(text: str):
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def scoped(rel: str, prefixes=CODE_SCOPES) -> bool:
    return rel.startswith(prefixes)


def is_training_call(func) -> bool:
    if isinstance(func, ast.Name):
        return func.id in TRAINING_CALL_NAMES
    if isinstance(func, ast.Attribute):
        return func.attr in TRAINING_CALL_NAMES or func.attr == "fit"
    return False


WRITE_CALL_NAMES = {
    "open", "to_netcdf", "to_zarr", "write_text", "write_bytes", "mkdir",
    "save", "savez", "dump", "unlink", "remove", "rmtree", "copy", "move",
}
WRITE_MODES = ("w", "a", "x", "w+", "a+", "wb", "ab", "xb")


def string_literals(node) -> list:
    """Every string constant appearing under `node` (arguments, not names)."""
    return [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)]


def write_targets(tree):
    """Yield (lineno, [string operands]) for calls that may write to the filesystem.

    AST-based on purpose: a line-based scan would be fooled by a path that only
    appears inside a string literal, such as a test fixture describing bad code.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name not in WRITE_CALL_NAMES:
            continue
        if name == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and "b" not in mode and mode not in ("w", "a", "x", "w+", "a+"):
                continue
            if not isinstance(mode, str):
                continue
        yield node.lineno, string_literals(node)


def r_002_archival_readonly(root: Path, allowed):
    """Active code must never write into an archived source-snapshot path.

    The snapshots themselves are historical sources that may contain writes;
    what matters is that current code treats them as read-only reference.
    Matched against ARCHIVE_PATH_TOKENS so every archive (legacy_v531, legacy_v6)
    is covered without editing this function.
    """
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for lineno, operands in write_targets(tree):
            for value in operands:
                token = next((t for t in ARCHIVE_PATH_TOKENS if t in value), None)
                if token:
                    out.append(Finding("R-002", rel, lineno,
                                       f"write to an archived snapshot path ({token}): {value!r}"))
                    break
    return out


# --------------------------------------------------------------------------- rules

def r_001_exclusive_outputs(root: Path, allowed):
    """New data outputs must be created exclusively ('x'), never truncated ('w')."""
    out = []
    target = re.compile(r"(raw|interim|processed|manifests|store|dataset|cache)", re.I)
    for rel, path in iter_py(root):
        if is_archival(rel) or not rel.startswith(("data/", "training/", "scripts/")):
            continue
        text = read(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not re.search(r"\.open\(|open\(", line):
                continue
            if not re.search(r"['\"]w['\"]|mode\s*=\s*['\"]w", line):
                continue
            if not target.search(line):
                continue
            out.append(Finding("R-001", rel, lineno, line.strip()[:110]))
    return out


def r_006_protocol_first(root: Path, allowed):
    """protocol.json must be written before the first training/eval call.

    Compared per top-level function, so a helper that merely mentions a study
    call is not mistaken for the study body.
    """
    out = []
    paths = []
    for pattern in STUDY_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue

        def write_line(node):
            """Line of the first statement that persists the frozen protocol.

            Only individual statements are inspected (never the whole function),
            otherwise unparsing the enclosing def would swallow every later call
            and make a wrong-order write look compliant.
            """
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(sub, ast.Call):
                    target = None
                    if isinstance(sub.func, ast.Name):
                        target = sub.func.id
                    elif isinstance(sub.func, ast.Attribute):
                        target = sub.func.attr
                    if target in ("_json", "_write", "dump", "write_text", "to_json"):
                        return sub.lineno
                if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call):
                    continue
            return None

        checked = False
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [sub.lineno for sub in ast.walk(node)
                     if isinstance(sub, ast.Call) and is_training_call(sub.func)]
            if not calls:
                continue
            checked = True
            wline = write_line(node)
            first = min(calls)
            tolerated = rel in PROTOCOL_EXCEPTIONS
            if wline is None:
                out.append(Finding("R-006", rel, first,
                                   "bounded study calls training/eval without a protocol.json write",
                                   tolerated))
            elif first < wline:
                out.append(Finding("R-006", rel, first,
                                   f"training/eval call precedes protocol write (line {wline})",
                                   tolerated))
        if not checked:
            out.append(Finding("R-006", rel, 0,
                               "study-scoped module has no bounded study function (check the glob scope)"))
    return out


def r_007_claim_flag(root: Path, allowed):
    """Bounded study/pilot results must carry an explicit non-claim flag."""
    out = []
    flag = re.compile(r"scientific_(claim|forecast_claim|training_ready|training_certified)|not_claimed")
    paths = []
    for pattern in STUDY_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        text = read(path)
        if text and not flag.search(text):
            out.append(Finding("R-007", rel, 0, "no scientific claim / not-claimed flag in result payload"))
    return out


def r_010_ci_code_archive(root: Path, allowed):
    """A workflow that runs a bounded study must archive code commit and zip."""
    out = []
    wf_dir = root / CI_WORKFLOW_DIR
    if not wf_dir.is_dir():
        return out
    for path in sorted(wf_dir.glob("*.yml")):
        rel = path.relative_to(root).as_posix()
        text = read(path) or ""
        if not CI_STUDY_EVIDENCE.search(text):
            continue
        has_commit = "rev-parse HEAD" in text
        has_archive = "git archive" in text
        if not (has_commit and has_archive):
            missing = ", ".join(n for n, ok in
                                (("code_commit.txt", has_commit), ("code.zip", has_archive)) if not ok)
            out.append(Finding("R-010", rel, 0, f"bounded study workflow lacks {missing}",
                               rel in CI_ARCHIVE_EXCEPTIONS))
    return out


def r_012_no_artifacts_tracked(root: Path, allowed):
    """Tracked files must exclude raw data, outputs and checkpoints."""
    if not (root / ".git").exists():
        return [Finding("R-012", ".", 0, "UNKNOWN: no git metadata, tracked set cannot be verified")]
    proc = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    if proc.returncode != 0:
        return [Finding("R-012", ".", 0, "UNKNOWN: git ls-files failed")]
    out = []
    for tracked in proc.stdout.splitlines():
        if tracked in ("data/raw/.gitkeep", "data/interim/.gitkeep", "data/processed/.gitkeep"):
            continue
        if tracked.startswith(("data/raw/", "data/interim/", "data/processed/", "outputs/", "logs/")):
            out.append(Finding("R-012", tracked, 0, "raw/interim/processed/outputs content is tracked"))
        elif tracked.endswith((".ckpt", ".pt", ".pth")):
            out.append(Finding("R-012", tracked, 0, "model checkpoint is tracked"))
    return out


def r_013_explicit_encoding(root: Path, allowed):
    """Text-mode open() must pass encoding='utf-8'."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "open"):
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and "b" in mode:
                continue
            out.append(Finding("R-013", rel, node.lineno, ast.unparse(node)[:100]))
    return out


def r_014_no_absolute_paths(root: Path, allowed):
    """Active code must not embed host absolute paths."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if ABSOLUTE_PATH.search(line):
                out.append(Finding("R-014", rel, lineno, line.strip()[:110]))
    return out


def r_015_no_bare_except(root: Path, allowed):
    """`except:` without a type is forbidden."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                out.append(Finding("R-015", rel, node.lineno, "bare except:"))
    return out


def r_016_network_imports(root: Path, allowed):
    """Outbound network clients only inside data/download/."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not rel.startswith(NETWORK_SCOPE_PREFIXES):
            continue
        if rel.startswith(NETWORK_ALLOWED_PREFIXES):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in NETWORK_MODULES:
                    out.append(Finding("R-016", rel, node.lineno,
                                       f"import {name} outside data/download/"))
    return out


def r_017_future_annotations(root: Path, allowed):
    """Library/script modules must enable postponed annotations."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or rel in EXEMPT_FROM_ANNOTATIONS:
            continue
        if not rel.startswith(("data/", "model/", "training/", "scripts/")):
            continue
        text = read(path)
        if text and "from __future__ import annotations" not in text:
            out.append(Finding("R-017", rel, 0, "missing 'from __future__ import annotations'"))
    return out


def r_019_line_length(root: Path, allowed):
    """Line-length hard cap (C class: report only)."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if len(line) > LINE_LEN_HARD:
                out.append(Finding("R-019", rel, lineno, f"{len(line)} chars (hard cap {LINE_LEN_HARD})"))
    return out


def r_019b_line_length_target(root: Path, allowed):
    """Line-length target cap (C class: report only)."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if LINE_LEN_TARGET < len(line) <= LINE_LEN_HARD:
                out.append(Finding("R-019b", rel, lineno, f"{len(line)} chars (target {LINE_LEN_TARGET})"))
    return out


def r_020_function_length(root: Path, allowed):
    """Function body length (C class: report only)."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno)
                if end - node.lineno + 1 > FUNC_BODY_MAX:
                    out.append(Finding("R-020", rel, node.lineno,
                                       f"{node.name}: {end - node.lineno + 1} lines (max {FUNC_BODY_MAX})"))
    return out


def r_021_file_length(root: Path, allowed):
    """File length (C class: report only)."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        if text and len(text.splitlines()) > FILE_LOC_MAX:
            out.append(Finding("R-021", rel, 0, f"{len(text.splitlines())} LOC (max {FILE_LOC_MAX})"))
    return out


def r_022_nesting_depth(root: Path, allowed):
    """Nesting depth (C class: report only)."""
    out = []

    def depth(node, level=0):
        best = level
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try,
                                  ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                best = max(best, depth(child, level + 1))
            else:
                best = max(best, depth(child, level))
        return best

    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                value = depth(node)
                if value > NESTING_MAX:
                    out.append(Finding("R-022", rel, node.lineno,
                                       f"{node.name}: nesting {value} (max {NESTING_MAX})"))
    return out


def r_023_parameter_count(root: Path, allowed):
    """Function parameter count (C class: report only); constructors excluded."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "__init__":
                args = [a for a in list(node.args.args) + list(node.args.kwonlyargs)
                        + list(node.args.posonlyargs) if a.arg not in ("self", "cls")]
                if len(args) > PARAM_MAX:
                    out.append(Finding("R-023", rel, node.lineno,
                                       f"{node.name}: {len(args)} parameters (max {PARAM_MAX})"))
    return out


def r_031_no_archival_imports(root: Path, allowed):
    """Active modules must not import the archived V5.3.1 snapshot."""
    out = []
    pattern = re.compile(r"^\s*(?:from|import)\s+\S*legacy_v531")
    for rel, path in iter_py(root):
        if is_archival(rel):
            continue
        text = read(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.match(line):
                out.append(Finding("R-031", rel, lineno, line.strip()[:110]))
    return out


def r_004_raw_readonly(root: Path, allowed):
    """data/raw, data/interim and data/processed are never written in place."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for lineno, operands in write_targets(tree):
            for value in operands:
                if re.search(r"\bdata/(raw|interim|processed)\b", value):
                    out.append(Finding("R-004", rel, lineno,
                                       f"write into an immutable data directory: {value!r}"))
                    break
    return out


def r_005_build_complete(root: Path, allowed):
    """Local dataset publication must go through the BUILD_COMPLETE contract."""
    marker = root / "data" / "preprocess" / "contracts.py"
    out = []
    if not marker.is_file():
        return [Finding("R-005", "data/preprocess/contracts.py", 0,
                        "UNKNOWN: publication contract module is missing")]
    text = read(marker) or ""
    if "BUILD_COMPLETE.json" not in text or "def fresh_outputs" not in text:
        out.append(Finding("R-005", "data/preprocess/contracts.py", 0,
                           "fresh_outputs/BUILD_COMPLETE publication contract not found"))
    readers = (root / "data" / "r7_store.py")
    if readers.is_file():
        reader_text = read(readers) or ""
        if "build_complete" not in reader_text:
            out.append(Finding("R-005", "data/r7_store.py", 0,
                               "store reader does not require build_complete"))
    return out


def r_008_no_synthetic_fallback(root: Path, allowed):
    """A failed real acquisition must never silently substitute synthetic data.

    The violation is *swallowing* a failure — an except handler that neither
    re-raises nor records a failure status, letting the caller believe the real
    download succeeded. Handlers that re-raise (optionally after annotating the
    message) are conforming, and so are handlers that write an auditable failure
    record before re-raising.

    Scope boundary: handlers for ImportError/ModuleNotFoundError are dependency
    API compatibility shims (e.g. `icechunk.storage.s3_storage` falling back to
    `icechunk.s3_storage`), not data-acquisition failures, and are out of scope.
    """
    out = []
    import_errors = {"ImportError", "ModuleNotFoundError"}
    for rel, path in iter_py(root):
        if is_archival(rel) or not rel.startswith("data/download/"):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                caught = ast.unparse(handler.type) if handler.type else ""
                if any(name in caught for name in import_errors):
                    continue
                reraises = any(isinstance(stmt, ast.Raise) for stmt in ast.walk(handler))
                if reraises:
                    continue
                # A handler that returns/continues without raising must leave an
                # auditable trace, otherwise the failure is silent.
                records = any(
                    isinstance(stmt, ast.Call) and re.search(
                        r"dump|write_text|_json|_write|receipt|journal|log",
                        ast.unparse(stmt.func))
                    for stmt in ast.walk(handler) if isinstance(stmt, ast.Call)
                )
                annotates = any(
                    isinstance(stmt, ast.Call) and re.search(
                        r"append|append_|setdefault|update|add",
                        ast.unparse(stmt.func))
                    for stmt in ast.walk(handler) if isinstance(stmt, ast.Call)
                )
                if not (records or annotates):
                    out.append(Finding("R-008", rel, handler.lineno,
                                       "except handler swallows a download failure without "
                                       "re-raising or recording it"))
    return out


def r_018_os_path(root: Path, allowed):
    """Use pathlib; the relative-path idiom os.path.relpath is the only exception."""
    out = []
    for rel, path in iter_py(root):
        if is_archival(rel) or not scoped(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if (isinstance(value, ast.Attribute) and value.attr == "path"
                    and isinstance(value.value, ast.Name) and value.value.id == "os"):
                name = f"os.path.{node.attr}"
                if name in OS_PATH_ALLOWED:
                    continue
                out.append(Finding("R-018", rel, node.lineno, f"{name} (use pathlib instead)"))
    return out


def r_024_tests_use_tmp_path(root: Path, allowed):
    """Tests must write into tmp_path, never into the repository tree."""
    out = []
    for rel, path in iter_py(root):
        if not rel.startswith("tests/"):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        repo_ref = re.compile(r"^(data|outputs|logs|results|legacy_v531_full)/")
        for lineno, operands in write_targets(tree):
            for value in operands:
                if repo_ref.match(value):
                    out.append(Finding("R-024", rel, lineno,
                                       f"test writes to a repository path: {value!r}"))
                    break
    return out


def r_025_tests_offline(root: Path, allowed):
    """Tests must not import real network clients (socket is allowed for denial)."""
    out = []
    clients = NETWORK_MODULES - {"socket", "urllib.parse"}
    for rel, path in iter_py(root):
        if not rel.startswith("tests/"):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in clients or name in clients:
                    out.append(Finding("R-025", rel, node.lineno,
                                       f"test imports network client {name}"))
    return out


def r_009_no_test_weakening(root: Path, allowed):
    """Test strength must not silently shrink (C class: report only).

    Counts test functions and assertions and compares them with the baseline
    recorded in docs/rules/testing.md. A decrease is reported for review; it is
    never blocking, because a legitimate refactor can merge tests. The point is
    to make an unremarked loss of coverage visible instead of silent.
    """
    out = []
    tests = asserts = 0
    for rel, path in iter_py(root):
        if not rel.startswith("tests/") or is_archival(rel):
            continue
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                tests += 1
            if isinstance(node, ast.Assert):
                asserts += 1
    if tests < TEST_FUNCTION_BASELINE:
        out.append(Finding("R-009", "tests/", 0,
                           f"test functions={tests} below baseline {TEST_FUNCTION_BASELINE}"))
    if asserts < ASSERT_BASELINE:
        out.append(Finding("R-009", "tests/", 0,
                           f"assertions={asserts} below baseline {ASSERT_BASELINE}"))
    return out


def r_027_commit_message_convention(root: Path, allowed):
    """Recent commits should follow `type(r7): summary (#N) [tag]` (report only)."""
    if not (root / ".git").exists():
        return [Finding("R-027", ".", 0, "UNKNOWN: no git metadata, commit history unavailable")]
    proc = subprocess.run(["git", "-C", str(root), "log", "--format=%s", "-n", str(COMMIT_SAMPLE_SIZE)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return [Finding("R-027", ".", 0, "UNKNOWN: git log failed")]
    subjects = [s for s in proc.stdout.splitlines() if s.strip()]
    if not subjects:
        return [Finding("R-027", ".", 0, "UNKNOWN: no commits sampled")]
    bad = [s for s in subjects if not CONVENTIONAL_COMMIT.match(s)]
    if bad:
        out = [Finding("R-027", ".", 0,
                       f"{len(bad)}/{len(subjects)} recent commit subjects are non-conventional")]
        out[0].detail += "; first: " + bad[0][:70]
        return out
    return []


def r_028_offline_workflow_denies_network(root: Path, allowed):
    """A workflow consuming archived artifacts must actively deny outbound network.

    Scoped to workflows that run a bounded study AND download an artifact: those
    are supposed to work from pinned local bytes, so "it happens not to connect"
    is not good enough. Fails closed by raising inside the run step.
    """
    out = []
    wf_dir = root / CI_WORKFLOW_DIR
    if not wf_dir.is_dir():
        return out
    for path in sorted(wf_dir.glob("*.yml")):
        rel = path.relative_to(root).as_posix()
        text = read(path) or ""
        if "download-artifact" not in text or not CI_STUDY_EVIDENCE.search(text):
            continue
        if "socket.socket.connect" not in text or "socket.create_connection" not in text:
            missing = [n for n, ok in (("socket.socket.connect", "socket.socket.connect" in text),
                                       ("socket.create_connection", "socket.create_connection" in text))
                       if not ok]
            out.append(Finding("R-028", rel, 0,
                               f"offline study workflow lacks network denial: {', '.join(missing)}"))
    return out


def r_029_workflow_has_timeout(root: Path, allowed):
    """Every CI workflow job must set timeout-minutes."""
    out = []
    wf_dir = root / CI_WORKFLOW_DIR
    if not wf_dir.is_dir():
        return out
    for path in sorted(wf_dir.glob("*.yml")):
        rel = path.relative_to(root).as_posix()
        text = read(path) or ""
        if "jobs:" not in text:
            continue
        if "timeout-minutes" not in text:
            out.append(Finding("R-029", rel, 0, "workflow sets no timeout-minutes"))
    return out


def r_030_study_has_wall_deadline(root: Path, allowed):
    """Bounded studies should abort on their own wall clock (C class: report only).

    A study that only relies on the CI timeout burns the whole budget before
    failing. Reported rather than blocked because two modules predate the
    convention (see docs/rules/MIGRATION.md).
    """
    out = []
    paths = []
    for pattern in STUDY_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [sub for sub in ast.walk(node)
                     if isinstance(sub, ast.Call) and is_training_call(sub.func)]
            if not calls:
                continue
            source = ast.unparse(node)
            if "monotonic" not in source or "raise" not in source:
                out.append(Finding("R-030", rel, node.lineno,
                                   f"{node.name} runs bounded training without an internal wall-clock deadline",
                                   rel in DEADLINE_EXCEPTIONS))
    return out


# --------------------------------------------------------------- R-032 – R-037
# Artifact storage. See docs/rules/artifact-storage.md.

# R-037: assets that must be tracked because they are *executed* — CI runs the
# checker, skills are auto-loaded, the contract is injected every turn.
GOVERNANCE_ASSETS = (
    "AGENTS.md",
    "docs/skills/README.md",
    "tools/check_conventions.py",
    "tests/test_check_conventions.py",
    "tests/test_agent_hooks.py",
)
GOVERNANCE_DIRS = (
    "docs/rules/",
    ".agents/skills/",
    "tools/agent_hooks/",
    "docs/plans/",
    "docs/decisions/",
    "docs/goals/",
)

VALID_DECISION_STATUS = ("proposed", "accepted", "deprecated", "superseded")

# R-035: tool runtime state that must stay out of version control.
TOOL_STATE_PREFIXES = (
    ".mimosa/", ".zcode/plans/", ".zcode/workflow-drafts/", "outputs/", "logs/",
    "input_pilot/", "input_seasonal/", "input_study/", "input_continuous/",
    "input54/", "input56/",
)


def r_032_plan_two_tier(root: Path, allowed):
    """An archived plan under docs/plans/ must record its actual outcome.

    Only the archived tier is checkable: whether a working plan was ever
    archived is a human decision, so a never-archived plan cannot be flagged.
    """
    out = []
    plans = root / "docs" / "plans"
    if not plans.is_dir():
        return out
    for path in sorted(plans.glob("*.md")):
        if path.name == "README.md":
            continue
        text = read(path) or ""
        if "## 实际结果" not in text:
            out.append(Finding("R-032", path.relative_to(root).as_posix(), 0,
                               "archived plan has no '## 实际结果' section"))
    return out


def r_033_decision_records(root: Path, allowed):
    """Decision records: sequential unique numbering, required sections, valid status."""
    out = []
    decisions = root / "docs" / "decisions"
    if not decisions.is_dir():
        return out
    seen = {}
    for path in sorted(decisions.glob("*.md")):
        if path.name == "README.md":
            continue
        rel = path.relative_to(root).as_posix()
        match = re.match(r"^(\d{4})-([a-z0-9-]+)\.md$", path.name)
        if not match:
            out.append(Finding("R-033", rel, 0, "filename must be NNNN-<kebab-slug>.md"))
            continue
        number = int(match.group(1))
        if number in seen:
            out.append(Finding("R-033", rel, 0,
                               f"number {number:04d} already used by {seen[number]}"))
        else:
            seen[number] = rel

        text = read(path) or ""
        for section in ("## Context", "## Decision", "## Consequences"):
            if section not in text:
                out.append(Finding("R-033", rel, 0, f"missing required section {section}"))
        status = re.search(r"(?m)^-\s*\*\*状态\*\*：\s*(\S+)", text)
        if not status:
            out.append(Finding("R-033", rel, 0, "missing a '- **状态**：' line"))
        else:
            value = status.group(1).strip().rstrip("。")
            if value not in VALID_DECISION_STATUS and not value.startswith("superseded"):
                out.append(Finding("R-033", rel, 0,
                                   f"invalid status {value!r}; expected one of "
                                   f"{', '.join(VALID_DECISION_STATUS)}"))
        if "## Consequences" in text:
            tail = text.split("## Consequences", 1)[1]
            if not any(marker in tail for marker in ("变难", "代价", "Trade-off")):
                out.append(Finding("R-033", rel, 0,
                                   "Consequences must list the costs, not only benefits"))

    if seen:
        gaps = sorted(set(range(1, max(seen) + 1)) - set(seen))
        if gaps:
            out.append(Finding("R-033", "docs/decisions/", 0,
                               f"numbering has gaps at {[f'{g:04d}' for g in gaps]}"))
    return out


def r_036_superseded_chain(root: Path, allowed):
    """A record marked 'superseded by NNNN' must point at a record that exists."""
    out = []
    decisions = root / "docs" / "decisions"
    if not decisions.is_dir():
        return out
    existing = {m.group(1) for p in decisions.glob("*.md")
                if (m := re.match(r"^(\d{4})-", p.name))}
    for path in sorted(decisions.glob("*.md")):
        rel = path.relative_to(root).as_posix()
        text = read(path) or ""
        if re.search(r"(?m)^-\s*\*\*状态\*\*：\s*superseded", text) is None:
            continue
        targets = re.findall(r"superseded by\s+(\d{4})", text)
        if not targets:
            out.append(Finding("R-036", rel, 0,
                               "status says 'superseded' but no NNNN target is given"))
            continue
        for target in targets:
            if target not in existing:
                out.append(Finding("R-036", rel, 0,
                                   f"superseded by {target}, but decision {target} does not exist"))
    return out


def r_035_tool_state_untracked(root: Path, allowed):
    """Tool runtime state must not be tracked (C class: report only).

    Being git-ignored is the desired state, so this only reports a path that is
    tracked *by mistake* — the actual violation.
    """
    if not (root / ".git").exists():
        return [Finding("R-035", ".", 0, "UNKNOWN: no git metadata, tracked set unavailable")]
    proc = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    if proc.returncode != 0:
        return [Finding("R-035", ".", 0, "UNKNOWN: git ls-files failed")]
    prefixes = tuple(p.rstrip("/") + "/" for p in TOOL_STATE_PREFIXES)
    out = []
    for tracked in proc.stdout.splitlines():
        for prefix in prefixes:
            if tracked.startswith(prefix):
                out.append(Finding("R-035", tracked, 0,
                                   f"tool runtime state under {prefix!r} is tracked"))
                break
    return out


def r_037_governance_tracked(root: Path, allowed):
    """Governance assets must be tracked; a clean clone is the test.

    Before this rule, the whole governance layer was untracked while ci.yml
    already ran tools/check_conventions.py — so CI failed on any fresh clone
    while every local check passed.
    """
    if not (root / ".git").exists():
        return [Finding("R-037", ".", 0, "UNKNOWN: no git metadata, tracked set unavailable")]
    proc = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    if proc.returncode != 0:
        return [Finding("R-037", ".", 0, "UNKNOWN: git ls-files failed")]
    tracked = set(proc.stdout.splitlines())
    out = []
    for asset in GOVERNANCE_ASSETS:
        if (root / asset).exists() and asset not in tracked:
            out.append(Finding("R-037", asset, 0, "governance asset is not tracked by git"))
    for directory in GOVERNANCE_DIRS:
        local = root / directory
        if not local.is_dir():
            continue
        present = [p.relative_to(root).as_posix() for p in sorted(local.rglob("*")) if p.is_file()]
        if present and not any(name in tracked for name in present):
            out.append(Finding("R-037", directory, 0,
                               f"governance directory has {len(present)} file(s), none tracked"))
    return out


# --------------------------------------------------------------- R-038 – R-048
# Naming. See docs/rules/naming.md. Scopes are measured against the active tree;
# archival dirs are excluded because they are read-only method records.

# tools/ is active code and must obey naming too, even though CODE_SCOPES
# (used by the size rules) does not list it.
NAMING_PY_SCOPES = ("data/", "model/", "training/", "scripts/", "tests/", "tools/")
# audit/ holds frozen evidence filenames (final_scorecard.md, final_*.log);
# renaming them would break references, so the banned-token rule exempts it.
NAMING_EXEMPT_PREFIXES = ("audit/",) + ARCHIVAL_PREFIXES

BANNED_NAME_TOKENS = (
    "final", "old", "new", "tmp", "temp", "copy", "backup", "bak", "draft",
    "deprecated", "misc",
)
JUNK_DIR_NAMES = (
    "utils", "util", "misc", "common", "commons", "helpers", "helper",
    "manager", "base", "shared", "lib", "libs", "core",
)
_BANNED_TOKEN_RE = re.compile(
    r"(?:^|[_\-.])(?:" + "|".join(BANNED_NAME_TOKENS) + r")(?:[_\-.]|$)", re.I)
_VERSION_SUFFIX_RE = re.compile(r"_v\d+$", re.I)

# R-042 default allow-list of idiomatically short parameter names. Measured in
# E-179: every flagged name is a 2-char domain abbreviation (xarray Dataset,
# learning rate, tensor dims, meteorological variables), not an unclear name.
SHORT_PARAM_ALLOWED = frozenset({
    "x", "y", "z", "i", "j", "k", "t", "n", "m", "p", "b", "c", "h", "w",
    "s", "d", "e", "f", "g", "r", "v", "a", "u", "q",
    "ds", "da", "lr", "kv", "kw", "fn", "hw", "hp", "wp", "p0", "nc", "id",
    "ok", "up", "to", "at", "by", "on", "in", "of", "or", "is", "as", "no",
})


def _naming_py(root: Path):
    """Active .py files subject to naming rules (archival + audit excluded)."""
    for rel, path in iter_py(root):
        if is_archival(rel) or rel.startswith(NAMING_EXEMPT_PREFIXES):
            continue
        if not rel.startswith(NAMING_PY_SCOPES):
            continue
        yield rel, path


def _non_ascii(value: str) -> bool:
    return any(ord(c) > 127 for c in value)


def r_038_module_filenames(root: Path, allowed):
    """Module and package filenames are snake_case (__init__.py excepted)."""
    out = []
    for rel, path in _naming_py(root):
        stem = path.stem
        if stem == "__init__":
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", stem):
            out.append(Finding("R-038", rel, 0, f"module name {stem!r} is not snake_case"))
    return out


def r_039_class_names(root: Path, allowed):
    """Class names are PascalCase, optionally with one leading underscore."""
    out = []
    for rel, path in _naming_py(root):
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", node.name):
                out.append(Finding("R-039", rel, node.lineno,
                                   f"class {node.name!r} is not PascalCase"))
    return out


def r_040_function_names(root: Path, allowed):
    """Function and method names are snake_case (dunder and _private allowed)."""
    out = []
    for rel, path in _naming_py(root):
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            if re.fullmatch(r"_{1,2}[a-z][a-z0-9_]*", name) or re.fullmatch(r"[a-z][a-z0-9_]*", name):
                continue
            out.append(Finding("R-040", rel, node.lineno,
                               f"function {name!r} is not snake_case"))
    return out


def r_041_constant_names(root: Path, allowed):
    """Module-level constants are UPPER_SNAKE.

    A constant is recognised by its VALUE being a literal or container (or by an
    all-caps name) — never by case alone, so lowercase pytest markers and
    argparse locals are not misread as constants.
    """
    out = []
    for rel, path in _naming_py(root):
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        # Module level only. Walking every node would flag ordinary local
        # variables that happen to be assigned a literal (`drafts = []`).
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            for target in targets:
                name = target.id
                if name == "__all__":
                    continue
                value = getattr(node, "value", None)
                is_const = (
                    name.isupper()
                    or isinstance(value, (ast.Tuple, ast.List, ast.Dict, ast.Set))
                    or (isinstance(value, ast.Constant) and isinstance(value.value, (str, int, float)))
                )
                if not is_const:
                    continue
                # A literal value with a mixed-case name is a mis-cased constant.
                # (`__all__` handled above; lowercase pytest markers are not
                # constants because their value is a call, not a literal.)
                if re.fullmatch(r"_?[A-Z][A-Z0-9_]*", name):
                    continue
                if name.startswith("__") and name.endswith("__"):
                    continue
                out.append(Finding("R-041", rel, getattr(node, "lineno", 0),
                                   f"constant {name!r} is not UPPER_SNAKE"))
    return out


def r_042_ascii_identifiers(root: Path, allowed):
    """Identifiers are pure ASCII; comments and docstrings are not constrained.

    Only AST identifier nodes are inspected. A line-based scan would flag the 43
    files that carry Chinese comments, which are intentional in this project.
    """
    out = []
    for rel, path in _naming_py(root):
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates.append(node.name)
            elif isinstance(node, ast.Name):
                candidates.append(node.id)
            elif isinstance(node, ast.arg):
                candidates.append(node.arg)
            elif isinstance(node, ast.Attribute):
                candidates.append(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                candidates.append(node.arg)
            for name in candidates:
                if _non_ascii(name):
                    out.append(Finding("R-042", rel, getattr(node, "lineno", 0),
                                       f"identifier {name!r} contains non-ASCII characters"))
    return out


def r_043_banned_names(root: Path, allowed):
    """No banned tokens, `_vN` suffixes or junk-drawer directory names in code paths."""
    out = []
    for rel, path in iter_py(root):
        if rel.startswith(NAMING_EXEMPT_PREFIXES):
            continue
        if not rel.startswith(NAMING_PY_SCOPES):
            continue
        for part in Path(rel).parts:
            stem = Path(part).stem if "." in part else part
            if _BANNED_TOKEN_RE.search(stem) or _VERSION_SUFFIX_RE.search(stem):
                out.append(Finding("R-043", rel, 0, f"banned token in name {part!r}"))
                break
    # Junk-drawer directory names, active tree only.
    for dirpath, dirnames, _ in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in list(dirnames):
            directory = Path(dirpath) / name
            rel = directory.relative_to(root).as_posix() + "/"
            if rel.startswith(NAMING_EXEMPT_PREFIXES):
                continue
            if name.lower() in JUNK_DIR_NAMES:
                out.append(Finding("R-043", rel, 0, f"junk-drawer directory name {name!r}"))
    return out


def r_044_test_names(root: Path, allowed):
    """Test files and functions are named test_*; scope is tests/ only.

    Required scope: outside tests/ there are six PyTorch-Lightning protocol
    methods named test_step / test_dataloader that must not be renamed.
    """
    out = []
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return out
    for path in sorted(tests_dir.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if path.name != "conftest.py" and not re.fullmatch(r"test_[a-z0-9_]+\.py", path.name):
            out.append(Finding("R-044", rel, 0, "test file must be named test_<subject>.py"))
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                if not re.fullmatch(r"test_[a-z0-9_]+", node.name):
                    out.append(Finding("R-044", rel, node.lineno,
                                       f"test {node.name!r} must be test_snake_case"))
    return out


def r_045_topic_marker_delimited(root: Path, allowed):
    """Where the `r7` topic marker appears it must sit in a delimited position.

    Both `r7_` prefix and `_r7` suffix are established practice, so a prefix is
    NOT required — measured against 129 modules, all of which match one of these.
    """
    out = []
    for rel, path in _naming_py(root):
        stem = path.stem
        if "r7" not in stem:
            continue
        # Delimited means `r7` sits between boundaries: `r7_helper`, `thing_r7`,
        # `x_r7_y`, or a bare `r7`. `r7x_helper` does NOT qualify — a letter
        # directly after `r7` makes it part of a longer word.
        if re.fullmatch(r"r7(?:_[a-z0-9]+)*", stem) or re.search(r"(?:^|_)r7(?:_|$)", stem):
            continue
        out.append(Finding("R-045", rel, 0,
                           f"topic marker 'r7' in {stem!r} is not in a delimited position"))
    return out


def r_046_config_and_workflow_names(root: Path, allowed):
    """Configs are snake_case.yaml; workflows are kebab-case.yml."""
    out = []
    for path in sorted((root / "configs").glob("*")):
        if path.suffix not in (".yaml", ".yml"):
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.ya?ml", path.name):
            out.append(Finding("R-046", path.relative_to(root).as_posix(), 0,
                               "config name must be snake_case.yaml"))
    for path in sorted((root / ".github" / "workflows").glob("*.yml")):
        if not re.fullmatch(r"[a-z0-9-]+\.yml", path.name):
            out.append(Finding("R-046", path.relative_to(root).as_posix(), 0,
                               "workflow name must be kebab-case.yml"))
    return out


def r_047_doc_names(root: Path, allowed):
    """docs/ doc names are layered by directory.

    Top level and the four ledger files under docs/rules/ use UPPER_SNAKE; the
    category rule files under docs/rules/ use kebab-case; docs/plans/ and
    docs/decisions/ use NNNN-kebab-case. README.md is exempt everywhere.
    """
    KEBAB = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.md")
    NUMBERED_KEBAB = re.compile(r"\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md")
    docs = root / "docs"
    if not docs.is_dir():
        return []
    out = []
    for path in sorted(docs.glob("*.md")):
        if not re.fullmatch(r"[A-Z0-9_]+\.md", path.name):
            out.append(Finding("R-047", path.relative_to(root).as_posix(), 0,
                               "top-level doc must be UPPER_SNAKE.md"))
    rules_dir = docs / "rules"
    if rules_dir.is_dir():
        for path in sorted(rules_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            ok = re.fullmatch(r"[A-Z0-9_]+\.md", path.name) or KEBAB.fullmatch(path.name)
            if not ok:
                out.append(Finding("R-047", path.relative_to(root).as_posix(), 0,
                                   "docs/rules/ name must be UPPER_SNAKE.md (ledger) "
                                   "or kebab-case.md (category)"))
    for sub in ("plans", "decisions"):
        directory = docs / sub
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            if path.name == "README.md":
                continue
            if not NUMBERED_KEBAB.fullmatch(path.name):
                out.append(Finding("R-047", path.relative_to(root).as_posix(), 0,
                                   f"docs/{sub}/ name must be NNNN-kebab-case.md"))
    # Any other docs/ subdirectory falls back to plain kebab-case.
    for path in sorted(docs.rglob("*.md")):
        parent = path.parent
        if parent == docs or parent in (rules_dir, docs / "plans", docs / "decisions"):
            continue
        if path.name == "README.md":
            continue
        if not KEBAB.fullmatch(path.name):
            out.append(Finding("R-047", path.relative_to(root).as_posix(), 0,
                               "doc under a docs/ subdirectory must be kebab-case.md"))
    return out


def r_048_parameter_abbreviations(root: Path, allowed):
    """Surface only parameters whose name carries no information (C class, report only).

    Deliberately narrow. A first attempt flagged any consonant run inside a
    name, which lit up 113 sites including clear names like `mlp_ratio` and
    `model_cfg` — a noisy report gets ignored, so the predicate was withdrawn.

    What remains is the case that is unambiguous and cheap to check: a
    parameter named with a single meaningless token (no underscore, not a known
    domain abbreviation, and not a recognised word). Those are the ones a
    reader actually has to go and look up.
    """
    out = []
    for rel, path in _naming_py(root):
        text = read(path)
        tree = parse(text) if text else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Lambda):
                args = list(node.args.args) + list(node.args.kwonlyargs)
            else:
                args = [a for a in list(node.args.args) + list(node.args.kwonlyargs)
                        + list(node.args.posonlyargs) if a.arg not in ("self", "cls")]
            for arg in args:
                name = arg.arg
                if len(name) < 4 or "_" in name:
                    continue
                if name.casefold() in SHORT_PARAM_ALLOWED:
                    continue
                # A 4+ char single-token name that is neither a dictionary word
                # nor a domain term is the readable-abbreviation case we want.
                if re.fullmatch(r"[bcdfghjklmnpqrstvwxz]{4,}", name.casefold()):
                    out.append(Finding("R-048", rel, arg.lineno,
                                       f"parameter {name!r} is an opaque abbreviation"))
    return out


RULES = {
    "R-001": ("新数据输出必须排他创建，禁止截断写入", r_001_exclusive_outputs),
    "R-002": ("归档快照只读", r_002_archival_readonly),
    "R-004": ("原始/中间/派生数据目录禁止原地写入", r_004_raw_readonly),
    "R-005": ("本地数据集必须经 BUILD_COMPLETE 发布契约", r_005_build_complete),
    "R-006": ("批量实验必须先冻结 protocol.json 再训练", r_006_protocol_first),
    "R-007": ("实验产物必须带显式非科学声明标志", r_007_claim_flag),
    "R-008": ("真实下载失败必须留审计，禁止 synthetic fallback", r_008_no_synthetic_fallback),
    "R-009": (f"测试不得静默减弱（基线 {TEST_FUNCTION_BASELINE}/{ASSERT_BASELINE}，报告）", r_009_no_test_weakening),
    "R-010": ("跑实验的 workflow 必须归档代码身份", r_010_ci_code_archive),
    "R-012": ("原始数据/产物不得进入版本控制", r_012_no_artifacts_tracked),
    "R-013": ("文本 IO 必须显式 encoding='utf-8'", r_013_explicit_encoding),
    "R-014": ("禁止硬编码宿主绝对路径", r_014_no_absolute_paths),
    "R-015": ("禁止裸 except", r_015_no_bare_except),
    "R-016": ("出网客户端只允许在 data/download/ 内", r_016_network_imports),
    "R-017": ("库与脚本模块必须启用 postponed annotations", r_017_future_annotations),
    "R-018": ("路径操作使用 pathlib", r_018_os_path),
    "R-019": (f"单行长度硬上限 {LINE_LEN_HARD}（C 类目标态，报告）", r_019_line_length),
    "R-019b": (f"单行长度目标上限 {LINE_LEN_TARGET}（C 类目标态，报告）", r_019b_line_length_target),
    "R-020": (f"函数体长度上限 {FUNC_BODY_MAX}（C 类目标态，报告）", r_020_function_length),
    "R-021": (f"单文件行数上限 {FILE_LOC_MAX}（C 类目标态，报告）", r_021_file_length),
    "R-022": (f"嵌套深度上限 {NESTING_MAX}（C 类目标态，报告）", r_022_nesting_depth),
    "R-023": (f"函数参数上限 {PARAM_MAX}（C 类目标态，报告）", r_023_parameter_count),
    "R-024": ("测试只写 tmp_path，不写仓库树", r_024_tests_use_tmp_path),
    "R-025": ("测试禁止导入真实网络客户端", r_025_tests_offline),
    "R-027": (f"最近 {COMMIT_SAMPLE_SIZE} 条提交遵循 Conventional Commits（报告）", r_027_commit_message_convention),
    "R-028": ("离线实验 workflow 必须真正禁网", r_028_offline_workflow_denies_network),
    "R-029": ("workflow 必须设置 timeout-minutes", r_029_workflow_has_timeout),
    "R-030": ("有界实验必须有内部墙钟截止（C 类目标态，报告）", r_030_study_has_wall_deadline),
    "R-031": ("活跃代码禁止 import 归档快照", r_031_no_archival_imports),
    "R-032": ("归档的计划必须记录实际结果", r_032_plan_two_tier),
    "R-033": ("决策记录编号唯一且含必需段落", r_033_decision_records),
    "R-035": ("工具运行态不得进版本控制（报告）", r_035_tool_state_untracked),
    "R-036": ("superseded 决策的取代目标必须存在", r_036_superseded_chain),
    "R-037": ("治理层必须在版本控制内", r_037_governance_tracked),
    "R-038": ("模块与包文件名 snake_case", r_038_module_filenames),
    "R-039": ("类名 PascalCase（允许前导 _）", r_039_class_names),
    "R-040": ("函数与方法名 snake_case", r_040_function_names),
    "R-041": ("模块级常量 UPPER_SNAKE（__all__ 豁免）", r_041_constant_names),
    "R-042": ("标识符纯 ASCII（注释不受限）", r_042_ascii_identifiers),
    "R-043": ("代码路径禁用词与杂物桶名（audit/ 豁免）", r_043_banned_names),
    "R-044": ("测试命名 test_*（范围限 tests/）", r_044_test_names),
    "R-045": ("r7 主题标记出现在定界位置", r_045_topic_marker_delimited),
    "R-046": ("配置 snake_case.yaml / workflow kebab-case.yml", r_046_config_and_workflow_names),
    "R-047": ("文档顶层 UPPER_SNAKE、子目录 kebab-case", r_047_doc_names),
    "R-048": ("参数名缩写（C 类目标态，报告）", r_048_parameter_abbreviations),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only UrbanPiDiT-R2 convention checker.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--rule", action="append", help="limit to rule id(s), e.g. --rule R-013")
    ap.add_argument("--report", action="store_true", help="also run C-class reporting rules")
    ap.add_argument("--quiet", action="store_true", help="print rule status only")
    ap.add_argument("--max-detail", type=int, default=5, help="detail lines per rule (0 = all)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 2

    default_group = list(BLOCKING_RULES) + (list(REPORT_RULES) if args.report else [])
    selected = args.rule or default_group
    unknown = [r for r in selected if r not in RULES]
    if unknown:
        print(f"error: unknown rule id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    blocking_failures = 0
    report_hits = 0
    for rule in selected:
        statement, fn = RULES[rule]
        findings = fn(root, set(selected))
        blocking = rule in BLOCKING_RULES
        tolerated = [f for f in findings if f.tolerated]
        real = [f for f in findings if not f.tolerated]
        unknown_findings = [f for f in real if "UNKNOWN:" in f.detail]

        if blocking:
            status = "PASS" if not real else ("UNKNOWN" if unknown_findings and len(unknown_findings) == len(real) else "FAIL")
            if real:
                blocking_failures += 1
        else:
            status = "INFO" if findings else "PASS"
            report_hits += len(findings)
        note = f" tolerated={len(tolerated)}" if tolerated else ""
        print(f"{rule:7s} {status:7s} hits={len(real):4d}{note}  {statement}  [{'blocking' if blocking else 'report'}]")
        if not args.quiet:
            limit = len(real) if args.max_detail == 0 else min(args.max_detail, len(real))
            for finding in real[:limit]:
                location = f"{finding.path}:{finding.line}" if finding.line else finding.path
                print(f"            {location}  {finding.detail}")
            if limit < len(real):
                print(f"            ... {len(real) - limit} more")
            if tolerated:
                for finding in tolerated:
                    print(f"            (tolerated) {finding.path}  {finding.detail}")

    print()
    print(f"blocking rules={len(BLOCKING_RULES)} failing={blocking_failures}  "
          f"report-only hits={report_hits}")
    print("C-class target rules are never blocking; see docs/rules/MIGRATION.md.")
    print("not mechanised here: " + ", ".join(RULES_NOT_MECHANISED))
    return 1 if blocking_failures else 0


if __name__ == "__main__":
    sys.exit(main())

