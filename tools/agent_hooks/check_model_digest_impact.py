"""PostToolUse notice: flag edits that change the model source digest.

Rationale (project facts, not invented here):
  * `model_code_digest()` (`training/r7_experiment.py`) hashes every non-legacy
    `.py` under `model/`. Any byte change — including pure reformatting — moves it.
  * `load_checkpoint` compares that digest and fails closed on mismatch, so
    pre-existing checkpoints can no longer be loaded by current code.
  * Three workflows load archived checkpoints with current HEAD code:
    `r7-restored-diagnostic.yml`, `r7-correction-audit.yml`,
    `r7-extended-control.yml` (see docs/rules/OPEN_QUESTIONS.md Q-009 / Q-012).

This hook does NOT block. Reformatting `model/` is a legitimate action; the
point is that the consequence is stated at the moment it happens rather than
discovered later when a replay fails. Precedent for the disposition is recorded
in `docs/R7_CPU_REFINEMENT_RESULTS.md`: use the artifact's archived `code.zip`
for older checkpoints, and never bypass the integrity check.

Emits `systemMessage` only when it has something to say.
"""
from __future__ import annotations

import json
import sys

# Only these tools can modify source bytes.
_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

# Digest-covered scope: model/**.py, excluding anything under a legacy/ path
# (mirrors model_code_digest's own skip).  tests/ and training/ are NOT covered,
# so edits there must not trigger this notice.
_DIGEST_ROOT = "model/"


def _is_digest_covered(raw_path: str) -> bool:
    if not isinstance(raw_path, str):
        return False
    path = raw_path.strip().strip("'\"")
    while path.startswith("./"):
        path = path[2:]
    if "legacy" in path:
        return False  # model/legacy_v531/** is excluded by model_code_digest
    if not path.startswith(_DIGEST_ROOT):
        return False
    return path.endswith(".py")


MESSAGE = (
    "model/ source changed -> `model_code_sha256` is no longer the value recorded "
    "in existing checkpoints. `load_checkpoint` fails closed on mismatch, so archived "
    "artifacts (r7-restored-diagnostic / r7-correction-audit / r7-extended-control) can "
    "only be replayed with their own archived code.zip.\n"
    "Required follow-ups for this change:\n"
    "  1. Mark the commit `[model-digest-change]` so the digest move is visible in history.\n"
    "  2. Record it in docs/rules/CHANGELOG.md next to the previous value "
    "(20196c64... -> d9fb07f2... is the existing entry).\n"
    "  3. Do NOT relax or bypass the digest comparison — see docs/rules/OPEN_QUESTIONS.md Q-009."
)


def evaluate(input_data: dict) -> str | None:
    """Return the notice text when this edit changed the model digest scope."""
    tool_name = input_data.get("tool_name") or ""
    if tool_name not in _WRITE_TOOLS:
        return None
    tool_input = input_data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return None

    if tool_name == "NotebookEdit":
        candidate = tool_input.get("notebook_path")
    else:
        candidate = tool_input.get("file_path")

    return MESSAGE if _is_digest_covered(candidate) else None


def main() -> int:
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # fail open
        print(f"check_model_digest_impact: unreadable input ({exc})", file=sys.stderr)
        return 0

    try:
        message = evaluate(input_data)
    except Exception as exc:  # fail open
        print(f"check_model_digest_impact: check failed ({exc})", file=sys.stderr)
        return 0

    if not message:
        return 0

    print(json.dumps({"systemMessage": message}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
