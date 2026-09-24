"""PreToolUse guard: refuse destructive git operations.

Rationale (project rules, not invented here):
  * AGENTS.md hard constraints — no `main` merge, no force push, no destructive
    data operation without explicit authorization.
  * `docs/R7_MANUAL_ITERATION.md:15` — "no force push"; the branch must stay
    `r7/weather-reasoning` on Draft PR #12.

Scope discipline: only genuinely destructive forms are denied. Read-only git
(`status`, `log`, `diff`, `show`, `rev-parse`, `grep`, `ls-files`, `blame`) must
always pass — they are a primary evidence source for this project, and R-027
even samples `git log` output.

Fail-open: any internal error allows the operation and explains itself on stderr.
"""
from __future__ import annotations

import json
import re
import shlex
import sys

# Global options that may sit between `git` and the subcommand.
_GIT_GLOBS = r"(?:\s+(?:-[cC]\s+\S+|--[A-Za-z-]+(?:=\S+)?))*"

# Destructive remote rewrites: a pushed branch's history is rewritten for everyone.
_FORCE_PUSH = re.compile(
    r"\bgit" + _GIT_GLOBS + r"\s+push\b[^|;&]*(?:--force(?!-with-lease)\b|--force-with-lease\b|\s-f\b)",
)
_FORCE_PUSH_PLUS = re.compile(r"\bgit" + _GIT_GLOBS + r"\s+push\b[^|;&]*\+[A-Za-z0-9_./-]+:")
# `git reset --hard` discards uncommitted work irrecoverably.
_RESET_HARD = re.compile(r"\bgit" + _GIT_GLOBS + r"\s+reset\b[^|;&]*--hard\b")
# `git clean -fd` deletes untracked files; -x also removes ignored ones.
_CLEAN_FORCE = re.compile(r"\bgit" + _GIT_GLOBS + r"\s+clean\b[^|;&]*(?:-[A-Za-z]*f[A-Za-z]*\b|--force\b)")
# `git branch -D` force-deletes unmerged work;
# `git push --delete` / `:branch` removes a remote branch.
_BRANCH_FORCE_DELETE = re.compile(r"\bgit" + _GIT_GLOBS + r"\s+branch\b[^|;&]*(?:-D\b|--delete\s+--force\b)")
# `git checkout .` / `git restore .` / `git checkout -- .` wipe working-tree edits.
_DISCARD_WORKTREE = re.compile(
    r"\bgit" + _GIT_GLOBS + r"\s+(?:checkout|restore)\b[^|;&]*(?:\s--\s+|\s)(?:\.|\./)\s*(?:$|[|;&])",
)

# Writes aimed at the default/integration branch. This project works on
# r7/weather-reasoning and merges to main only with explicit authorization.
_MAIN_BRANCH_WRITES = (
    re.compile(r"\bgit" + _GIT_GLOBS + r"\s+push\b[^|;&]*\s(?:origin\s+)?(?:main|master)\b"),
    re.compile(r"\bgit" + _GIT_GLOBS + r"\s+merge\b[^|;&]*\s(?:main|master)\b"),
    re.compile(r"\bgh\s+pr\s+merge\b"),
)

_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (_FORCE_PUSH, "force-push"),
    (_FORCE_PUSH_PLUS, "force-push (via +refspec)"),
    (_RESET_HARD, "git reset --hard"),
    (_CLEAN_FORCE, "git clean -f"),
    (_BRANCH_FORCE_DELETE, "git branch -D"),
    (_DISCARD_WORKTREE, "discarding working-tree changes"),
    (_MAIN_BRANCH_WRITES[0], "pushing to main"),
    (_MAIN_BRANCH_WRITES[1], "merging main locally"),
    (_MAIN_BRANCH_WRITES[2], "gh pr merge"),
)

_ADVICE = {
    "force-push": (
        "Force-pushing rewrites public history and is prohibited on this branch.\n"
        "If the remote needs a correction, publish a NEW commit on top instead."
    ),
    "force-push (via +refspec)": (
        "`+refspec` is a force-push in disguise and is prohibited.\n"
        "Publish a new commit on top instead of rewriting the remote ref."
    ),
    "git reset --hard": (
        "`reset --hard` discards uncommitted work irrecoverably.\n"
        "To inspect or set aside changes use `git stash` (recoverable) or `git diff` first.\n"
        "If you truly need to drop work, do it in an explicitly authorized step."
    ),
    "git clean -f": (
        "`git clean -f` deletes untracked files with no recovery.\n"
        "List what would go first with `git clean -nd`, then remove specific paths "
        "deliberately instead of bulk-deleting."
    ),
    "git branch -D": (
        "`branch -D` deletes unmerged work. Use `git branch -d` (safe delete) so git "
        "refuses when commits would be lost, or inspect with `git log <branch>` first."
    ),
    "discarding working-tree changes": (
        "This wipes uncommitted edits in the working tree.\n"
        "Commit or `git stash` them first so the work stays recoverable."
    ),
    "pushing to main": (
        "Work stays on `r7/weather-reasoning` (Draft PR #12); pushing to main is not authorized.\n"
        "Push your branch and update the Draft PR instead."
    ),
    "merging main locally": (
        "Merging into the integration branch requires explicit authorization "
        "(see AGENTS.md hard constraints)."
    ),
    "gh pr merge": (
        "Merging the PR is a release action requiring explicit authorization.\n"
        "Prepare the branch and let the user decide when to merge."
    ),
}


def evaluate(input_data: dict) -> tuple[str, str] | None:
    """Return (rule_label, advice) when the command must be denied."""
    if (input_data.get("tool_name") or "") != "Bash":
        return None
    tool_input = input_data.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return None

    # Strip quoted strings so `git log --grep="reset --hard"` is not a hit.
    try:
        text = " ".join(shlex.split(command))
    except ValueError:
        text = command

    for pattern, label in _RULES:
        if pattern.search(text):
            return label, _ADVICE.get(label, "")
    return None


def main() -> int:
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # fail open
        print(f"guard_destructive_git: unreadable hook input ({exc}); allowing", file=sys.stderr)
        return 0

    try:
        verdict = evaluate(input_data)
    except Exception as exc:  # fail open
        print(f"guard_destructive_git: check failed ({exc}); allowing", file=sys.stderr)
        return 0

    if verdict is None:
        return 0

    label, advice = verdict
    print(f"BLOCKED (AGENTS.md hard constraints / R7_MANUAL_ITERATION.md): {label}\n{advice}",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
