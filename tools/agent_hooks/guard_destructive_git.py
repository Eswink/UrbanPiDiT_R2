"""PreToolUse guard: refuse destructive git operations.

Rationale (project rules, not invented here):
  * AGENTS.md hard constraints — no `main` merge, no force push, no destructive
    data operation without explicit authorization.
  * `docs/R7_MANUAL_ITERATION.md:15` — "no force push".
  * Decision `docs/decisions/0003-push-vs-merge-hook-policy.md` (2026-09-25,
    explicit user authorization): non-force PUSHES to main are allowed; MERGES
    (local `git merge` involving main, `gh pr merge`) stay gated and require
    the user's explicit authorization. GitHub rejects non-fast-forward pushes,
    and every force variant is denied below, so an allowed main push can only
    advance main linearly over working-branch commits that already passed CI.

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
# Hardened 2026-09-25 (decision 0003): the destination after ':' is optional so
# a bare `+main` — force-moving the default branch with no refspec colon — is
# caught too; the original pattern required `+src:dst` and missed it.
_FORCE_PUSH_PLUS = re.compile(
    r"\bgit" + _GIT_GLOBS + r"\s+push\b[^|;&]*\+[A-Za-z0-9_./-]+(?::[A-Za-z0-9_./-]+)?",
)
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
# `--mirror` force-updates every ref (main included) and deletes unmatched
# remote branches. Never matched by any rule before 2026-09-25; lethal once
# pushes to main are allowed, so it is explicitly denied.
_MIRROR_PUSH = re.compile(r"\bgit" + _GIT_GLOBS + r"\s+push\b[^|;&]*--mirror\b")
# Deleting the default branch (`git push origin :main`, `git push origin
# --delete main`) is unrecoverable for the published repo. The empty-source
# refspec form requires whitespace before the ':' so a normal ff refspec like
# `HEAD:main` is not a hit; `--delete main` is spelled out because
# _BRANCH_FORCE_DELETE only covers `--delete --force`.
_DEFAULT_BRANCH_DELETE = re.compile(
    r"\bgit" + _GIT_GLOBS + r"\s+push\b[^|;&]*"
    r"(?:\s(?:origin\s+)?:\s*(?:main|master)\b|\s--delete\b[^|;&]*\s(?:origin\s+)?(?:main|master)\b)",
)

# Merges into the integration branch stay gated: they are release-adjacent
# actions requiring the user's explicit authorization (AGENTS.md hard
# constraints; decision 0003). Non-force pushes to main are NOT gated — see the
# module docstring. Hardened 2026-09-25: `origin/main` (and `origin/master`)
# are merges involving the integration branch too, so the `origin/` prefix is
# accepted alongside the bare-branch form; the trailing `\b` keeps words like
# `maintain` from matching.
_MAIN_MERGE = (
    re.compile(r"\bgit" + _GIT_GLOBS + r"\s+merge\b[^|;&]*\s(?:origin[/\s]+)?(?:main|master)\b"),
    re.compile(r"\bgh\s+pr\s+merge\b"),
)

_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (_FORCE_PUSH, "force-push"),
    (_FORCE_PUSH_PLUS, "force-push (via +refspec)"),
    (_RESET_HARD, "git reset --hard"),
    (_CLEAN_FORCE, "git clean -f"),
    (_BRANCH_FORCE_DELETE, "git branch -D"),
    (_DISCARD_WORKTREE, "discarding working-tree changes"),
    (_MIRROR_PUSH, "mirror push"),
    (_DEFAULT_BRANCH_DELETE, "deleting the default branch"),
    (_MAIN_MERGE[0], "merging main locally"),
    (_MAIN_MERGE[1], "gh pr merge"),
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
    "mirror push": (
        "`--mirror` force-updates every ref (main included) and deletes remote "
        "branches that have no local counterpart.\n"
        "Push explicit refspecs instead; mirror pushes are not authorized."
    ),
    "deleting the default branch": (
        "Deleting the default branch removes the published repo's integration "
        "history and is not authorized.\n"
        "If a branch must be removed, name it explicitly and never the default."
    ),
    "merging main locally": (
        "Merges require the user's explicit authorization and cannot be verified "
        "in-conversation.\n"
        "Execute the merge outside ZCode, or remove this hook entry BEFORE the "
        "session starts (hook config is read at session start; see AGENTS.md)."
    ),
    "gh pr merge": (
        "Merging the PR is a release action requiring explicit user authorization.\n"
        "Let the user decide when to merge; the hook cannot verify in-conversation "
        "authorization."
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
