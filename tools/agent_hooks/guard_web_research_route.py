"""PreToolUse guard: route external lookups through the web-researcher subagent.

Rationale (project rules, not invented here):
  * `docs/rules/external-sources.md` R-049 — information from outside this
    repository must be fetched by the `web-researcher` subagent
    (`.zcode/agents/web-researcher.md`); the main link must not call
    `WebSearch` / `WebFetch` directly.
  * The sanctioned fallback, when that subagent is unavailable, is `curl` from
    Bash (R-049). This guard deliberately leaves Bash alone so the fallback can
    never be blocked by this hook.
  * Decision `docs/decisions/0018-web-research-route.md`.

Hook-payload limits measured in this client build (E-186 / E-187):
  * tool events carry no agent identity — the only subagent-correlated value is
    `session_id` (child sessions look like `sess_subagent_agent_<uuid>`);
  * child sessions appear not to run project hooks at all (code inference).
  Hence this guard targets the main link, and additionally allows any call whose
  `session_id` carries the child-session prefix: should a future build start
  running hooks inside subagents, the route guard must not lock
  `web-researcher` — the one agent whose entire toolset is exactly these two
  tools — out of its job.

Fail-open: any internal error allows the call and explains itself on stderr.
"""
from __future__ import annotations

import json
import sys

# The two tools that reach the open web. Kept in lockstep with the agent
# definition by tests/test_agent_hooks.py, which asserts this set equals
# `.zcode/agents/web-researcher.md`'s tool list and that no other project agent
# carries either tool.
WEB_TOOLS = frozenset({"WebSearch", "WebFetch"})

# Child sessions are created with this id shape by the client.
SUBAGENT_SESSION_PREFIX = "sess_subagent_"

_REASON = (
    "BLOCKED (R-049, docs/rules/external-sources.md): external lookups must be "
    "delegated to the web-researcher subagent.\n"
    "Do this instead: call the Agent tool with subagent_type: web-researcher and a "
    "self-contained prompt (it cannot see this repository), asking for per-claim URLs, "
    "primary sources first, and an explicit 'not confirmed' section.\n"
    "If that subagent is unavailable (dispatch error / no output / empty result), fall "
    "back to `curl` from Bash and record the URL, access date and failure reason (R-050). "
    "Do not fetch from the main link with WebFetch/WebSearch.\n"
    "Skill: .agents/skills/web-research/SKILL.md"
)


def evaluate(input_data: dict) -> str | None:
    """Return the deny reason when the main link tries to reach the open web."""
    tool_name = input_data.get("tool_name") or ""
    if tool_name not in WEB_TOOLS:
        return None
    session_id = input_data.get("session_id") or ""
    if isinstance(session_id, str) and session_id.startswith(SUBAGENT_SESSION_PREFIX):
        return None  # anti-self-lock: never gate the sanctioned agent's own tools
    return _REASON


def main() -> int:
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # fail open
        print(f"guard_web_research_route: unreadable hook input ({exc}); allowing",
              file=sys.stderr)
        return 0

    try:
        reason = evaluate(input_data)
    except Exception as exc:  # fail open
        print(f"guard_web_research_route: check failed ({exc}); allowing", file=sys.stderr)
        return 0

    if reason is None:
        return 0

    print(reason, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
