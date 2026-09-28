"""PostToolUse notice: an external fetch just happened from the shell.

Rationale (project rules, not invented here):
  * `docs/rules/external-sources.md` R-050 — external citations must carry a URL
    and an access date; "reachable / unreachable" must be measured before it is
    written; failures must be recorded; a search summary is not evidence
    (precedent E-184).
  * R-049's fallback path is `curl` from Bash, so this notice exists to keep
    that fallback auditable. It never blocks — the fetch already happened, and
    blocking fetches would kill the sanctioned fallback.

Emits `systemMessage` only when the command actually fetched an http(s) URL with
curl or wget.
"""
from __future__ import annotations

import json
import re
import sys

# Token-boundary match so words like "curly" or "wgetter" are not hits.
_INVOKES_FETCHER = re.compile(r"(?<![\w.-])(?:curl|wget)(?![\w.-])")

_URL = re.compile(r"https?://[^\s'\"<>|;)]+")

MESSAGE_HEAD = (
    "External fetch via the shell: {urls}\n"
    "Record it per R-050 (docs/rules/external-sources.md): URL + access date + how it was "
    "fetched. Keep unreachable results as failures with the reason (timeout / 403 / login / "
    "anti-bot). A search summary is not evidence — verify key numbers against the primary "
    "source. Raw captures go to outputs/web-research/ (untracked).\n"
    "If this was a fallback because the web-researcher subagent was unavailable, say so and "
    "why. Skill: .agents/skills/web-research/SKILL.md"
)

_MAX_SHOWN_URLS = 3


def fetched_urls(command: str) -> list[str]:
    """Return the distinct external URLs a curl/wget command line would touch."""
    if not isinstance(command, str) or not command:
        return []
    if not _INVOKES_FETCHER.search(command):
        return []
    urls: list[str] = []
    for match in _URL.finditer(command):
        url = match.group(0).rstrip(".,")
        if url not in urls:
            urls.append(url)
    return urls


def evaluate(input_data: dict) -> str | None:
    """Return the notice text when this Bash call fetched an external URL."""
    if (input_data.get("tool_name") or "") != "Bash":
        return None
    tool_input = input_data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return None
    urls = fetched_urls(tool_input.get("command") or "")
    if not urls:
        return None
    shown = ", ".join(urls[:_MAX_SHOWN_URLS])
    if len(urls) > _MAX_SHOWN_URLS:
        shown += f" (+{len(urls) - _MAX_SHOWN_URLS} more)"
    return MESSAGE_HEAD.format(urls=shown)


def main() -> int:
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # fail open
        print(f"note_external_fetch: unreadable input ({exc})", file=sys.stderr)
        return 0

    try:
        message = evaluate(input_data)
    except Exception as exc:  # fail open
        print(f"note_external_fetch: check failed ({exc})", file=sys.stderr)
        return 0

    if not message:
        return 0

    print(json.dumps({"systemMessage": message}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
