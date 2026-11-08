"""Syntax-check a workflow draft with Node's TypeScript parser, then prove the check can fail.

Why this exists. ``.zcode/workflow-drafts/`` is git-ignored, so CI never sees those
files and nothing in this repository parses them. A draft is only compiled when it is
submitted to ``CreateWorkflow``, which means a typo can sit in one for weeks and surface
the first time someone tries to run a 20-hour round. ``tools/check_conventions.py``
reads Python only.

How it works. Node 22 can strip TypeScript types (``--experimental-strip-types``), and a
dynamic ``import`` forces a parse. Workflow scripts are compiled *inside a function body*,
so a bare top-level ``return`` is legal there but not in a module; the draft is therefore
wrapped in an ``async function`` before parsing. Types are STRIPPED, not checked: this
verifies syntax only. A semantic error (wrong property name, bad call signature) still
passes here and is caught only by the real compiler.

The self-test matters more than the check. Measured during development: a "broken"
variant that merely dropped a semicolon PARSED, because automatic semicolon insertion
makes it legal -- so a single negative case can quietly prove nothing. This module
therefore tries three genuinely-broken variants (unclosed brace, unterminated string,
dangling operator) and reports how many were rejected.

    python tools/check_workflow_syntax.py .zcode/workflow-drafts/foo.dwf.ts
    python tools/check_workflow_syntax.py <draft> --quiet      # exit code only

Exit codes: 0 = the draft parses AND every negative control was rejected;
1 = the draft failed to parse, or the controls did not fail (the check is untrustworthy);
2 = usage error / node missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

#: Each control appends one genuinely broken statement to the wrapped draft. The breakage
#: must survive automatic semicolon insertion, or the control rejects nothing.
_NEGATIVE_CONTROLS = {
    "unclosed_brace": "if (true) {",
    "unterminated_string": 'const __probe = "unterminated;',
    "dangling_operator": "const __probe = 1 + ;",
}

_HARNESS = """\
const name = process.argv[2];
try {{
  await import(name);
  console.log("PARSED");
}} catch (error) {{
  console.log("REJECTED:" + error.constructor.name);
}}
"""


def _strip_metadata(text: str) -> str:
    """Drop the leading ``/* zcode-workflow ... */`` block, which is comment, not code."""
    return re.sub(r"^\s*/\* zcode-workflow[\s\S]*?\*/", "", text, count=1)


def _wrap(body: str) -> str:
    """Wrap in an async function: the real compiler puts this body inside one."""
    return "async function __draft(args: Readonly<Record<string, unknown>>) {\n" + body + "\n}\n"


def _parse(folder: Path, source: str, name: str, node: str) -> dict:
    """Write ``source`` as ``name`` and ask Node whether it parses."""
    target = folder / name
    target.write_text(source, encoding="utf-8")
    harness = folder / "harness.mjs"
    harness.write_text(_HARNESS, encoding="utf-8")
    done = subprocess.run([node, "--experimental-strip-types", str(harness), str(target)],
                          capture_output=True, text=True, timeout=120, cwd=folder)
    output = (done.stdout + done.stderr).strip()
    return {"case": name, "parsed": "PARSED" in output, "raw": output.splitlines()[-1][:300]}


def check(draft: Path, node: str | None = None) -> dict:
    """Parse the draft, then confirm the parser rejects three genuinely broken variants."""
    node = node or shutil.which("node")
    if not node:
        return {"passed": False, "reason": "node is not installed; cannot type-strip"}
    text = draft.read_text(encoding="utf-8")
    body = _strip_metadata(text)
    with tempfile.TemporaryDirectory(prefix="wf_syntax_") as raw:
        folder = Path(raw)
        draft_result = _parse(folder, _wrap(body), "draft.mts", node)
        controls = []
        for label, broken_statement in _NEGATIVE_CONTROLS.items():
            # Append inside the wrapper's body, so only the control's own breakage shows.
            broken = _wrap(body + "\n" + broken_statement)
            result = _parse(folder, broken, f"bad_{label}.mts", node)
            controls.append({"control": label, "rejected": not result["parsed"]})
    controls_ok = all(item["rejected"] for item in controls) and bool(controls)
    return {"draft": str(draft),
            "bytes": len(text.encode("utf-8")),
            "parsed": draft_result["parsed"],
            "parse_message": draft_result["raw"],
            "negative_controls": controls,
            "controls_all_rejected": controls_ok,
            "note": "types are STRIPPED, not checked: syntax only",
            "passed": bool(draft_result["parsed"] and controls_ok)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("draft", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.draft.is_file():
        print(json.dumps({"error": f"not a file: {args.draft}"}))
        return 2
    report = check(args.draft)
    if args.quiet:
        print(json.dumps({"passed": report["passed"], "parsed": report.get("parsed"),
                          "controls": report.get("controls_all_rejected"),
                          "reason": report.get("reason")}, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
