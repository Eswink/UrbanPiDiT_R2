"""PreToolUse guard: a commit that would turn CI red is refused at the commit.

Why this exists. The convention gate ran only in two places before this hook: CI
(after the push) and the Stop hook (at the end of a turn, advisory and fail-open).
Neither one sits between `git add` and `git commit`, so a blocking violation could
be committed while the local gate was red - and the gate was red for a reason CI
could never reproduce, because the checker used to scan the working tree including
untracked debris. Decision 0017 fixed the second half by scoping blocking verdicts
to the index; this hook closes the first.

What it judges: exactly the content the commit will contain.

  * files already staged (``git diff --cached --name-only``);
  * the operands of ``git add`` / ``git rm`` / ``git mv`` in the same command line,
    because ``git add X && git commit`` is how this project commits;
  * every modified tracked file when the commit stages them itself (``-a``/``--all``).

An operand that names a directory, or a whole-tree form (``git add -A``, ``git add
.``), falls back to judging the whole tree: a superset is the honest answer when the
stage set cannot be enumerated. An operand that cannot be represented safely as a
path (a shell metacharacter, or a leading ``-`` that would read as a flag) also falls
back to the whole tree.

How it runs: the gate is *imported* and called in-process, so no path from a command
line is ever handed to a shell or to another program. The only subprocess calls are
git queries, whose arguments are literals from this module; the commit message is
never executed, and ``_safe_path_operand`` is the single door a path takes to reach
the checker.

Fail-open, like the other hooks: any internal error allows the commit and explains
itself on stderr. A broken guard must not wedge a session - and unlike a bad write
into data/raw, a missed check here has CI behind it.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# tools/agent_hooks/ -> tools/
_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

_SEGMENT = re.compile(r"&&|\|\||;|\n")
_HEREDOC = re.compile(r"<<-?\s*(?:\w+|'[^']*'|\"[^\"]*\")")
# git global options that take a separate value, e.g. `git -C <dir> commit`.
_GIT_VALUE_FLAGS = {"-C", "--git-dir", "--work-tree", "-c"}
_GIT_STAGE_SUBCOMMANDS = {"add", "rm", "mv"}
_WHOLE_TREE_OPERANDS = {".", "./", "-A", "--all", "*"}
_STAGE_EVERYTHING = {"-a", "--all", "-A"}
# Characters a repo-relative path may contain. Deliberately narrow: it excludes
# whitespace, quotes, backslashes and every shell metacharacter.
_SAFE_PATH = re.compile(r"[A-Za-z0-9_.@+/-]+")


def _safe_path_operand(raw: str) -> "str | None":
    """A path token this guard is willing to judge, or None when it is not a path.

    A leading ``-`` is refused rather than passed on: it would be read as an option
    by whatever consumes it next, which is the same class of mistake as quoting a
    path into a command.
    """
    candidate = raw.strip().strip("'\"")
    if not candidate or candidate.startswith("-"):
        return None
    if not _SAFE_PATH.fullmatch(candidate):
        return None
    return candidate


def _blank_heredocs(command: str) -> str:
    """Replace heredoc bodies with spaces: a commit message is data, not a command."""
    out = []
    pending = None
    for line in command.splitlines(keepends=True):
        if pending is not None:
            if line.strip() == pending:
                pending = None
            out.append(" " * len(line))
            continue
        match = _HEREDOC.search(line)
        if match:
            pending = match.group(0).split("<<", 1)[1].lstrip("-").strip().strip("'\"")
        out.append(line)
    return "".join(out)


def repo_root(declared: "str | None" = None) -> Path:
    """The checkout under judgement: an explicit -C style directory, or this repo."""
    if declared:
        candidate = Path(declared)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
        if (candidate / ".git").exists():
            return candidate.resolve()
    env = os.environ.get("ZCODE_PROJECT_DIR")
    if env and (Path(env) / ".git").exists():
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def git(root: Path, *args: str) -> list[str]:
    """A git query whose answer is a list of repo-relative paths.

    Every argument is a literal written here, and the checkout is selected with a
    working directory rather than an interpolated option value.
    """
    completed = subprocess.run(["git", *args], cwd=str(root),
                               capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return [name for name in completed.stdout.split("\0") if name]


def staged_paths(root: Path) -> list[str]:
    return git(root, "diff", "--cached", "--name-only", "-z")


def modified_paths(root: Path) -> list[str]:
    return git(root, "diff", "--name-only", "-z")


def commit_invocation(command: str) -> "tuple[Path, list[str], list[str]] | None":
    """Find a ``git ... commit`` in the command line.

    Returns ``(root, stage_operands, commit_flags)`` or None when the command does
    not commit anything. The root follows an explicit ``git -C <dir>`` (or a leading
    ``cd <dir>``), so a commit run inside a worktree is judged in that worktree.
    """
    root = repo_root()
    stage_operands: list[str] = []
    commit_flags: list[str] = []
    seen_commit = False
    for segment in _SEGMENT.split(_blank_heredocs(command)):
        tokens = segment.split()
        if not tokens:
            continue
        index = 0
        while index < len(tokens) and ("=" in tokens[index].split("-")[0]
                                       or tokens[index] in ("sudo", "env", "command",
                                                            "time", "nohup")):
            index += 1
        if index >= len(tokens):
            continue
        verb = tokens[index].rsplit("/", 1)[-1]
        args = tokens[index + 1:]
        if verb == "cd" and not seen_commit:
            target = next((a for a in args if not a.startswith("-")), None)
            if target:
                root = repo_root(target)
            continue
        if verb != "git":
            continue
        offset = 0
        while offset < len(args):
            token = args[offset]
            if token in _GIT_VALUE_FLAGS:
                if token == "-C" and offset + 1 < len(args):
                    root = repo_root(args[offset + 1])
                offset += 2
            elif token.startswith("--git-dir=") or token.startswith("--work-tree="):
                offset += 1
            else:
                break
        if offset >= len(args):
            continue
        sub = args[offset]
        rest = args[offset + 1:]
        if sub in _GIT_STAGE_SUBCOMMANDS:
            stage_operands.extend(token for token in rest if not token.startswith("-"))
        elif sub == "commit":
            seen_commit = True
            commit_flags.extend(token for token in rest if token.startswith("-"))
    if not seen_commit:
        return None
    return root, stage_operands, commit_flags


def paths_under_judgement(root: Path, operands: list[str],
                          flags: list[str]) -> "list[str] | None":
    """The paths this commit will contain, or None when the whole tree must be judged."""
    try:
        candidates = list(staged_paths(root))
    except Exception:
        return None  # cannot enumerate: judge the tree rather than nothing
    if set(flags) & _STAGE_EVERYTHING:
        try:
            candidates.extend(modified_paths(root))
        except Exception:
            return None
    for operand in operands:
        cleaned = _safe_path_operand(operand)
        if cleaned is None or cleaned in _WHOLE_TREE_OPERANDS:
            return None  # not a path this guard will carry, or it is the whole tree
        candidates.append(cleaned)
    return sorted({name for name in candidates if name})


def run_checker(root: Path, paths: "list[str] | None") -> "tuple[int, str]":
    """Run the blocking gate in-process over the given paths (None = whole tree).

    In-process rather than as a child command on purpose: the paths come from a
    shell command line, and this way they are Python arguments all the way down -
    there is no command string for anything to be injected into.
    """
    checker = importlib.import_module("check_conventions")
    argv = ["--root", str(root), "--max-detail", "3"]
    if paths is not None:
        argv += ["--paths", *paths]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            code = checker.main(argv)
        except SystemExit as exc:  # argparse usage errors
            code = int(exc.code or 2)
    return int(code), captured.getvalue()


def evaluate(input_data: dict) -> "tuple[Path, str, list[str]] | None":
    """Return (root, checker output, paths) when the commit must be refused."""
    if (input_data.get("tool_name") or "") != "Bash":
        return None
    tool_input = input_data.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or "commit" not in command:
        return None
    invocation = commit_invocation(command)
    if invocation is None:
        return None
    root, operands, flags = invocation
    paths = paths_under_judgement(root, operands, flags)
    if paths == []:
        return None  # nothing to commit on this machine
    try:
        code, output = run_checker(root, paths)
    except Exception as exc:  # fail open
        print(f"guard_conventions_before_commit: could not run the gate ({exc}); "
              f"allowing the commit", file=sys.stderr)
        return None
    if code == 0:
        return None
    return root, output, paths if paths is not None else ["<whole tree>"]


def _reason(root: Path, output: str, paths: list[str]) -> str:
    failing = [line.strip() for line in output.splitlines() if "FAIL" in line]
    detail = "\n".join(f"  {line}" for line in failing[:10])
    return (
        "BLOCKED (pre-commit gate, decision 0017): the blocking convention rules fail\n"
        f"for the content this commit would contain ({root}):\n"
        f"{detail or '  see the checker output below'}\n"
        f"\nJudged paths: {', '.join(paths[:12])}"
        f"{' …' if len(paths) > 12 else ''}\n"
        "\nThe same rules run in CI (ci.yml -> 'Check repository conventions'), so this\n"
        "commit would have turned CI red. Fix the named files, or commit without them.\n"
        "If the rule itself is wrong, that is a decision to record and make deliberately\n"
        "(docs/rules/CHANGELOG.md + docs/decisions/), not something to commit past."
    )


def main() -> int:
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # fail open
        print(f"guard_conventions_before_commit: unreadable hook input ({exc}); allowing",
              file=sys.stderr)
        return 0

    try:
        verdict = evaluate(input_data)
    except Exception as exc:  # fail open
        print(f"guard_conventions_before_commit: check failed ({exc}); allowing the commit",
              file=sys.stderr)
        return 0

    if verdict is None:
        return 0

    root, output, paths = verdict
    print(_reason(root, output, paths), file=sys.stderr)
    print(output[-4000:], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
