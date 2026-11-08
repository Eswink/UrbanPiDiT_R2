"""The workflow-draft syntax check, and the reason its negative controls matter.

``.zcode/workflow-drafts/`` is git-ignored, so nothing in CI parses those files and a typo
survives until someone submits the draft to ``CreateWorkflow``. These tests pin both
directions: the real draft parses, and deliberately broken variants are rejected.

The negative direction is the whole point. During development a "broken" variant that
merely dropped a semicolon still PARSED (automatic semicolon insertion makes it legal), so
a single negative case can prove nothing. The controls here are chosen to survive ASI.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / ".zcode/workflow-drafts/r7-throughput-physics-round.dwf.ts"


def _checker():
    spec = importlib.util.spec_from_file_location("check_workflow_syntax",
                                                  ROOT / "tools/check_workflow_syntax.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is required to strip TypeScript types")


def test_the_metadata_comment_is_stripped():
    """The leading zcode-workflow block is comment, so removing it must not change syntax."""
    checker = _checker()
    text = "/* zcode-workflow\ndescription: x\n*/\nconst a = 1;\n"
    assert _body_has_no_metadata(checker, text)


def _body_has_no_metadata(checker, text: str) -> bool:
    return "zcode-workflow" not in checker._strip_metadata(text) and "const a = 1;" in \
        checker._strip_metadata(text)


def test_the_draft_body_is_wrapped_in_a_function():
    """A workflow body has a top-level return, legal inside a function and not in a module."""
    checker = _checker()
    wrapped = checker._wrap("return 1;")
    assert wrapped.startswith("async function") and wrapped.rstrip().endswith("}")


def test_the_negative_controls_are_real_syntax_errors():
    """Each control must break parsing on its own, or the self-test proves nothing."""
    checker = _checker()
    assert set(checker._NEGATIVE_CONTROLS) == {"unclosed_brace", "unterminated_string",
                                               "dangling_operator"}
    # A missing semicolon is NOT such an error: automatic semicolon insertion repairs it.
    assert ";" not in checker._NEGATIVE_CONTROLS["unclosed_brace"]


def test_the_repository_draft_parses_and_its_controls_reject():
    """The actual draft -- the one that would burn a 20-hour round -- parses today."""
    if not DRAFT.is_file():
        pytest.skip(f"draft not present: {DRAFT}")
    report = _checker().check(DRAFT)
    assert report["passed"] is True, report
    assert report["parsed"] is True
    assert report["controls_all_rejected"] is True
    assert report["bytes"] > 1000
