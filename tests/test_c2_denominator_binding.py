"""C2's (a) divides by one file; the page and the record must name that same file.

Exit (a) is ``skill = 1 - (rmse / denominator)**2``, and the denominator comes from
``climatology_reference()``, which reads ``D2_REFERENCE`` in
``scripts/r7_s3_fy_year_coverage_climatology.py``. The result page and the evidence record both
print a digest of "the denominator" -- but neither of them runs ``readings()``. If either one
spelled the path out again, the two copies could drift apart, and then the page would print the
digest of a file the reading never divided by. That failure is silent and it *looks* traceable:
a digest is present, it verifies, and it names the wrong bytes.

These tests bind the three paths together. They are import-and-compare only -- no ``outputs/``
read, no torch -- so they run on CI's bare checkout, which is exactly where a drift would
otherwise go unnoticed until a registration months later.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    """Import a tool by dotted path with the repo on ``sys.path``."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return __import__(name, fromlist=["_"])


def test_the_page_and_the_record_name_the_canonical_denominator():
    """One Path object, three importers: the reading's, the page's, the record's."""
    canonical = _module("scripts.r7_s3_fy_year_coverage_climatology").D2_REFERENCE
    page = _module("tools.render_c2_result").D2_REFERENCE
    record = _module("tools.build_c2_registration").D2_REFERENCE
    assert page == canonical, f"the page would print a digest of {page}, not {canonical}"
    assert record == canonical, f"the record would pin {record}, not {canonical}"
    assert isinstance(page, Path) and isinstance(record, Path)


def test_neither_tool_spells_the_path_out_again():
    """A second literal would pass the equality test only while the two happen to agree.

    This reads the source instead of the imported value, so it still fails if someone re-adds a
    hardcoded path that today matches -- the point is that there must be one spelling, not two
    that are currently equal.
    """
    for name in ("tools/render_c2_result.py", "tools/build_c2_registration.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        assigned = {target.id for node in ast.walk(tree)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                    if isinstance(target, ast.Name)}
        assert "D2_REFERENCE" not in assigned, (
            f"{name} assigns D2_REFERENCE itself; it must import the canonical constant")


def test_the_canonical_module_names_the_attempt_directory_once():
    """The constant is the single source: the path appears in that module and nowhere else."""
    canonical_source = (ROOT / "scripts/r7_s3_fy_year_coverage_climatology.py").read_text(
        encoding="utf-8")
    assert "r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json" in canonical_source
    for name in ("tools/render_c2_result.py", "tools/build_c2_registration.py"):
        assert "r7_s3_fy_year_coverage_d2_20261023_attempt01" not in \
            (ROOT / name).read_text(encoding="utf-8"), f"{name} repeats the denominator path"
