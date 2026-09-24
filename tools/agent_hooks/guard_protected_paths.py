"""PreToolUse guard: refuse writes into immutable or archived paths.

Rationale (project rules, not invented here):
  * R-004 / AGENTS.md — `data/raw`, `data/interim`, `data/processed` are read-only;
    new data is written to a NEW destination.
  * R-002 / R-031 — `legacy_v531_full/`, `data/legacy_v531/`, `model/legacy_v531/`
    and `legacy_v6/` are read-only archival snapshots.

CI only notices a violation after it is committed (R-004 is enforced by
`tools/check_conventions.py`). This hook rejects it at the moment of the write,
and the rejection names the supported alternative instead of just saying "no".

The protected list is derived from `tools/check_conventions.py`, which stays the
single source of truth; `tests/test_agent_hooks.py` asserts the two cannot drift.

Fail-open: any internal error allows the operation and explains itself on stderr.
A broken guard must not wedge the session.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# tools/agent_hooks/ -> tools/
_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

try:
    import check_conventions as _cc
    _ARCHIVES = tuple(_cc.ARCHIVAL_PREFIXES)
except Exception:  # pragma: no cover - import must never block the guard
    _ARCHIVES = ("legacy_v531_full/", "data/legacy_v531/", "model/legacy_v531/", "legacy_v6/")

# R-004 targets. Kept as a literal list rather than imported because
# check_conventions encodes them inside a regex; the sync test covers both.
_IMMUTABLE = ("data/raw/", "data/interim/", "data/processed/")

PROTECTED_PREFIXES = _ARCHIVES + _IMMUTABLE

# Deletion-only protection for the test suite.
#
# R-009 already counts test functions and assertions, but it is aggregate and
# report-only, so deleting a whole test file slips past it until the totals drop
# below a baseline that is easy to bump by hand. Deleting tests is the exact
# failure mode the hard constraints forbid, so removal is blocked here while
# *adding* or *editing* tests stays completely unrestricted.
DELETION_ONLY_PREFIXES = ("tests/",)
# Verbs that remove their operands outright.
_DELETING_VERBS = {"rm", "rmdir", "unlink", "shred"}

# Bare tokens used for a cheap "does this command touch a protected area?" test.
PROTECTED_TOKENS = tuple(sorted({p.rstrip("/") for p in PROTECTED_PREFIXES}))
DELETION_TOKENS = tuple(p.rstrip("/") for p in DELETION_ONLY_PREFIXES)

# Write tools whose target path is a single explicit field.
PATH_FIELD_TOOLS = {
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}

# Shell segments split on the usual separators so `grep -rn "rm" archive/`
# is not mistaken for an `rm` invocation.
_SEGMENT = re.compile(r"&&|\|\||;|\||\n")

# Verbs that modify every path argument they are given.
_MUTATING_VERBS = {
    "rm", "rmdir", "truncate", "dd", "tee", "touch", "mkdir",
    "chmod", "chown", "chgrp", "shred", "unlink", "install", "patch",
}
# Verbs where only the destination matters for a protected path.
_DEST_ONLY_VERBS = {"cp", "rsync", "ln"}
# Verbs where both directions destroy something.
_BOTH_WAYS_VERBS = {"mv"}
# Programs that write to a path named by an output flag rather than an operand.
_EXTERNAL_FETCHERS = {"curl", "wget", "sort", "unzip", "install"}
_OUTPUT_FLAGS = ("-o", "--output", "-O", "--output-document", "--target-directory")
# git subcommands that can delete or overwrite tracked content.
_GIT_WRITE_SUBCMDS = {"rm", "mv", "clean", "checkout", "restore", "reset"}
# In-place editing flags. `-i` alone missed `--in-place`, and perl has its own.
_INPLACE_EDIT = re.compile(r"(?:^|\s)(?:-i|--in-place)(?:\s|$|=)")
_INPLACE_CAPABLE = {"sed", "perl"}
# Write primitives that make a `python -c` payload a modification.
_WRITE_PRIMITIVE = re.compile(
    r"open\([^)]*['\"][wax]|write_text\(|write_bytes\(|mkdir\(|unlink\(|remove\(|"
    r"rmtree\(|shutil\.(?:copy|move)|to_csv\(|to_netcdf\(|to_zarr\(|savez?\(|\.write\("
)
# `find ... -delete` or `find ... -exec rm|shred|unlink ...`
_FIND_DESTRUCTIVE = re.compile(r"(?:^|\s)(?:-delete|-exec\s+(?:rm|shred|unlink)\b)")

# Redirections, including fd-prefixed and noclobber forms: `>`, `>>`, `2>`, `1>>`, `>|`.
#
# The negative lookbehind matters: without it the arrow in ordinary prose
# (`a -> legacy_v6/x`, a `git commit -F - <<MSG` body, a README example) is read
# as a redirect into the archive. `-` before `>` is never a shell redirect.
_REDIRECT = re.compile(r"(?<![-<])\d*(?:>>?\|?|>\|)\s*(\S+)")
# Heredoc introducers (`<<EOF`, `<<-EOF`) are not redirections at all.
_HEREDOC = re.compile(r"<<-?\s*(?:\w+|'[^']*'|\"[^\"]*\")")
# A bare `>` inside a quoted string is a mention, not a redirect.
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


_PROTECTED_RE = re.compile(
    "|".join(re.escape(p.rstrip("/")) for p in PROTECTED_PREFIXES),
    re.IGNORECASE,
)


def _mentions_protected(text: str) -> bool:
    """Cheap pre-filter.

    Tolerant on purpose: this only decides whether the detailed check runs, and
    it also matches spellings the plain prefix compare would miss (`data//raw`,
    `data/./raw`, `data/RAW`, and a glob like `data/ra*`). The authoritative
    decision is still `_path_is_protected`, so a false positive here costs one
    extra parse — never a wrong verdict.
    """
    if _PROTECTED_RE.search(text):
        return True
    # Separator/casing variants: fold them away and retry.
    folded = re.sub(r"/{2,}", "/", text.replace("/./", "/")).casefold()
    if any(token.casefold() in folded for token in PROTECTED_TOKENS):
        return True
    # A glob that only partially spells a protected tree (`data/ra*`).
    for chunk in re.split(r"[\s'\"]+", text):
        if chunk and any(ch in chunk for ch in "*?[") and _path_is_protected(chunk):
            return True
    return False


def _which_prefix(text: str) -> str | None:
    for prefix in PROTECTED_PREFIXES:
        if prefix in text:
            return prefix
    for token in PROTECTED_TOKENS:
        if token in text:
            return token
    return None


def _normalize(raw: str) -> str:
    """Reduce a path token to a repo-relative, lexically clean form.

    Handles the spellings that would otherwise slip past a prefix compare:
      * surrounding quotes and a leading `./`
      * interior `.` and duplicate `/` components (`data/./raw`, `data//raw`)
      * absolute paths inside this checkout (`/repo/data/raw/...`)
    Case is folded separately in `_path_is_protected` so a `DATA/RAW` spelling
    is treated the same as `data/raw` on a case-insensitive filesystem.
    """
    value = raw.strip().strip("'\"")
    if not value:
        return ""
    # An absolute path that lives under an expected prefix is normalized to that
    # prefix (rstrip'ing the root first, so /data/esw/... can not match `data/`).
    absolute = value.startswith("/")
    probe = value.lstrip("/") if absolute else value
    if absolute and probe not in PROTECTED_PREFIXES:
        for prefix in PROTECTED_PREFIXES:
            marker = prefix.rstrip("/")
            idx = probe.find("/" + marker + "/")
            if idx >= 0:
                probe = probe[idx + 1:]
                break
            if probe.endswith("/" + marker):
                probe = probe[-len(marker):]
                break
        else:
            return ""  # unrelated absolute path
        value = probe
    # Collapse interior `.` and `//` components without touching `..`
    # (a `..` token cannot be resolved lexically without the cwd, so it is
    # rejected instead of silently passing).
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def _path_is_protected(raw: str) -> str | None:
    value = _normalize(raw)
    if not value:
        return None
    folded = value.casefold()
    for prefix in PROTECTED_PREFIXES:
        bare = prefix.rstrip("/")
        if folded == bare.casefold() or folded.startswith(bare.casefold() + "/"):
            return bare
    # Glob form: `data/ra*` names a protected tree without spelling it in full.
    # Compare only the literal part, so a wildcard cannot be used to reach it.
    if any(ch in value for ch in "*?["):
        literal = re.split(r"[*?\[]", value, maxsplit=1)[0].casefold()
        for prefix in PROTECTED_PREFIXES:
            bare = prefix.rstrip("/").casefold()
            if literal and (bare.startswith(literal) or literal.startswith(bare)):
                return prefix.rstrip("/")
    return None


def _reason(prefix: str, what: str) -> str:
    """Name the rule and the supported alternative."""
    if what == "removal of test files":
        return (
            f"BLOCKED (AGENTS.md 硬约束): {what}\n"
            f"Removing tests is the failure mode the project forbids ('禁止为了让报告好看而删除、\n"
            f"跳过或弱化测试与断言'). Rule R-009 only counts test functions and assertions in\n"
            f"aggregate and is report-only, so a whole-file deletion would not have stopped it.\n"
            f"\n"
            f"Adding, editing and refactoring tests are NOT blocked. If a test is genuinely\n"
            f"obsolete, that is a decision to record and make deliberately (with the reason in\n"
            f"the commit and CHANGELOG), not something to do silently mid-task. To retire a\n"
            f"test, state the intent first and adjust R-009's baseline in the same change."
        )
    if prefix.startswith(("legacy_v531", "legacy_v6")) or "legacy" in prefix:
        return (
            f"BLOCKED (R-002/R-031): {what} targets the read-only archive `{prefix}`.\n"
            f"Archives are historical method records and must not be modified or imported.\n"
            f"If you need this code path back, move it out of the archive in a reviewed change "
            f"(see docs/rules/OPEN_QUESTIONS.md Q-001 for the precedent)."
        )
    return (
        f"BLOCKED (R-004): {what} is a shell-level modification of `{prefix}`.\n"
        f"Rule R-004 keeps original and derived inputs reproducible: their content is not\n"
        f"edited, moved or deleted in place by shell commands or by hand edits.\n"
        f"\n"
        f"Note this does NOT block the project's own downloaders — they write these paths\n"
        f"deliberately and stay allowed, e.g.\n"
        f"  python scripts/try_real_downloads.py\n"
        f"  python -m data.download.arco_era5\n"
        f"  python scripts/prepare_real_smoke.py\n"
        f"Only use a new destination when you are deviating from a script's own default,\n"
        f"and never delete an old directory to make a retry succeed."
    )


def _split_args(tokens: list) -> list:
    """Drop flags, keeping operands. Handles `--opt=value` by yielding the value."""
    out = []
    for tok in tokens:
        if tok.startswith("-") and tok != "-":
            if "=" in tok:
                out.append(tok.split("=", 1)[1])
            continue
        out.append(tok)
    return out


def _under_tests(value: str) -> str | None:
    """Return the normalized path when it names something under tests/."""
    norm = _normalize(value)
    if not norm:
        return None
    folded = norm.casefold()
    for prefix in DELETION_ONLY_PREFIXES:
        bare = prefix.rstrip("/")
        if folded == bare or folded.startswith(bare + "/"):
            return norm
    return None


def _deletion_verdict(verb: str, operands: list, segment: str) -> str | None:
    """Return a protected path when the command REMOVES something under tests/.

    Only removal is guarded. Authoring, editing and refactoring tests stay
    unrestricted, which is why this is a separate, narrowly-scoped check rather
    than an entry in PROTECTED_PREFIXES.
    """
    candidates = [op for op in operands if _under_tests(op)]
    if not candidates:
        return None

    if verb in _DELETING_VERBS:
        return candidates[0]
    if verb == "find" and _FIND_DESTRUCTIVE.search(segment):
        return candidates[0]
    if verb == "git":
        # `git rm` / `git mv` out of the tree / `git clean` over it.
        sub = operands[0] if operands else ""
        if sub in ("rm",) or (sub == "clean" and _FIND_DESTRUCTIVE.search(segment)):
            return candidates[0]
        if sub == "mv":
            remaining = [op for op in operands[1:] if _under_tests(op)]
            if not remaining or remaining[0] == candidates[0]:
                return candidates[0]
    return None


def _blank_heredocs(command: str) -> str:
    """Replace heredoc bodies with spaces, preserving length and line structure.

    A heredoc body is stdin data, not shell syntax: `git commit -F - <<MSG` with
    prose mentioning `-> legacy_v6/x` must not be read as a redirect into the
    archive. Returns the command unchanged when no heredoc is present.
    """
    lines = command.splitlines(keepends=True)
    out = []
    pending = None
    for line in lines:
        if pending is not None:
            # Inside a heredoc body: blank it until the terminator line.
            if line.strip() == pending:
                pending = None
            out.append(" " * len(line) if line.endswith("\n") else " " * len(line))
            continue
        match = _HEREDOC.search(line)
        if match:
            pending = match.group(0).split("<<", 1)[1].lstrip("-").strip().strip("'\"")
        out.append(line)
    return "".join(out)


def _check_bash(command: str) -> tuple[str, str] | None:
    # The pre-filter is only an optimisation for the protected-path rules; the
    # tests/ deletion check must also run for commands that never name a
    # protected path (e.g. `git rm tests/test_x.py`).
    if not _mentions_protected(command) and not any(
        token in command.casefold() for token in DELETION_TOKENS
    ):
        return None

    # Redirections: `>`, `>>`, `N>`, `N>>`, `>|`. Before scanning, two kinds of
    # text are removed because they are DATA, not shell syntax:
    #   1. heredoc bodies — a `git commit -F - <<MSG` message quotes paths in prose
    #   2. quoted spans — `echo "see > data/raw/x"` only mentions a redirect
    # Both would otherwise be read as writes into a protected path.
    masked = _blank_heredocs(command)
    outside_quotes = _QUOTED.sub(lambda m: " " * len(m.group(0)), masked)
    for match in _REDIRECT.finditer(outside_quotes):
        hit = _path_is_protected(match.group(1))
        if hit:
            return hit, "a shell redirection"

    into_protected = None
    for segment in _SEGMENT.split(command):
        tokens = segment.split()
        if not tokens:
            continue
        # Skip leading env assignments and `sudo`-style wrappers.
        idx = 0
        while idx < len(tokens) and ("=" in tokens[idx].split("-")[0]
                                     or tokens[idx] in ("sudo", "env", "command", "time", "nohup")):
            idx += 1
        if idx >= len(tokens):
            continue
        verb = tokens[idx].rsplit("/", 1)[-1]
        args = tokens[idx + 1:]
        operands = _split_args(args)

        # Removal of tests is blocked; adding/editing tests is not.
        deleted = _deletion_verdict(verb, operands, segment)
        if deleted:
            return deleted, "removal of test files"

        # `cd <protected> && rm file` — relative operands then resolve inside the
        # protected directory. Only entered when the cd target itself is
        # protected, so ordinary `cd <dir> && ...` is unaffected.
        if verb == "cd":
            target = next((a for a in args if not a.startswith("-")), None)
            if target and _path_is_protected(target):
                into_protected = _normalize(target)
            continue

        if into_protected:
            for arg in operands:
                if arg.startswith("/") or arg.startswith(".."):
                    continue
                hit = _path_is_protected(into_protected + "/" + arg)
                if hit:
                    return hit, f"`{verb}` on a path relative to `{into_protected}`"
                if verb in _INPLACE_CAPABLE and _INPLACE_EDIT.search(segment):
                    return hit or into_protected, f"`{verb}` editing inside `{into_protected}`"

        # `python -c "<payload>"`: only denied when the payload combines a write
        # primitive with a protected path. Searching the raw payload (rather than
        # extracting quoted substrings) is deliberate — the outermost quote pair
        # would otherwise swallow the nested path literal. Plain `python -c
        # "import torch;print(...)"` (documented in AGENTS.md) is unaffected.
        if verb in ("python", "python3") and any(a == "-c" for a in args):
            if _WRITE_PRIMITIVE.search(segment):
                match = _PROTECTED_RE.search(segment)
                if match:
                    hit = _path_is_protected(match.group(0))
                    if hit:
                        return hit, "a `python -c` write"

        # `sed -i` / `sed --in-place` / `perl -i` rewrite their target in place.
        if verb in _INPLACE_CAPABLE:
            if _INPLACE_EDIT.search(segment):
                for arg in operands:
                    hit = _path_is_protected(arg)
                    if hit:
                        return hit, f"`{verb} -i` editing in place"
            continue

        # `find <dir> -delete` / `-exec rm` destroys what it walks.
        if verb == "find":
            if _FIND_DESTRUCTIVE.search(segment):
                for arg in operands:
                    hit = _path_is_protected(arg)
                    if hit:
                        return hit, "`find -delete/-exec` removing files"
            continue

        # `git -C <path> <verb>` and `git --git-dir=... <verb>`: strip globals
        # before reading the subcommand, otherwise the subcommand check misses.
        if verb == "git":
            offset = 0
            while offset < len(args):
                token = args[offset]
                if token in ("-C", "--git-dir", "--work-tree", "-c"):
                    offset += 2
                elif token.startswith("--git-dir=") or token.startswith("--work-tree="):
                    offset += 1
                else:
                    break
            if offset >= len(args):
                continue
            sub = args[offset]
            rest = args[offset + 1:]
            if sub in _GIT_WRITE_SUBCMDS or sub == "worktree":
                for arg in _split_args(rest):
                    hit = _path_is_protected(arg)
                    if hit:
                        return hit, f"`git {sub}`"
            continue

        if verb in _MUTATING_VERBS:
            # `dd of=<path>` names its target inside an operand, not as a bare arg.
            for arg in operands:
                candidate = arg.split("=", 1)[1] if arg.startswith("of=") else arg
                hit = _path_is_protected(candidate)
                if hit:
                    return hit, f"`{verb}`"
        elif verb in _BOTH_WAYS_VERBS:
            for arg in operands:
                hit = _path_is_protected(arg)
                if hit:
                    return hit, f"`{verb}` (moves/removes the original)"
        elif verb in _DEST_ONLY_VERBS:
            # For `cp`/`rsync`/`ln` the destination is the LAST operand; writing
            # a new file into a protected tree (`cp x legacy_v6/f.py`) is what
            # R-002/R-004 forbid. Reading a protected tree as the source is fine,
            # so earlier operands are deliberately not checked.
            if operands:
                hit = _path_is_protected(operands[-1])
                if hit:
                    return hit, f"`{verb}` writing into"
        elif verb in _EXTERNAL_FETCHERS:
            # `curl -o PATH`, `wget -O PATH`, `sort -o PATH`, `tee PATH`.
            for flag in _OUTPUT_FLAGS:
                for match in re.finditer(re.escape(flag) + r"\s*(\S+)", segment):
                    hit = _path_is_protected(match.group(1))
                    if hit:
                        return hit, f"`{verb} {flag}` writing into"
    return None


def _check_path_tool(tool_name: str, tool_input: dict) -> tuple[str, str] | None:
    field = PATH_FIELD_TOOLS.get(tool_name)
    if not field:
        return None
    raw = tool_input.get(field)
    if not isinstance(raw, str):
        return None
    hit = _path_is_protected(raw)
    if hit:
        return hit, f"`{tool_name}`"
    return None


def evaluate(input_data: dict) -> tuple[str, str] | None:
    """Return (protected_prefix, description) when the call must be denied."""
    tool_name = input_data.get("tool_name") or ""
    tool_input = input_data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return None

    if tool_name in PATH_FIELD_TOOLS:
        return _check_path_tool(tool_name, tool_input)

    if tool_name == "Bash":
        command = tool_input.get("command")
        if isinstance(command, str):
            return _check_bash(command)

    return None


def main() -> int:
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # fail open
        print(f"guard_protected_paths: unreadable hook input ({exc}); allowing", file=sys.stderr)
        return 0

    try:
        verdict = evaluate(input_data)
    except Exception as exc:  # fail open
        print(f"guard_protected_paths: check failed ({exc}); allowing", file=sys.stderr)
        return 0

    if verdict is None:
        return 0

    prefix, what = verdict
    print(_reason(prefix, what), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
