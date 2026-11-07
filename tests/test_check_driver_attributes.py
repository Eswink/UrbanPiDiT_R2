"""Counterproofs for ``tools/check_driver_attributes.py``; no data, no GPU, no network.

The tool exists because a missing *attribute* survives import and every test that does not run the
line, while the repository's older guard only resolved bare *globals*. So the tests here are mostly
negative: each one builds a tiny module pair on ``tmp_path`` and asserts the tool reports the
missing attribute, with a positive control in the same file so a silently no-op tool fails too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import check_driver_attributes as guard  # noqa: E402


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def two_modules(tmp_path, monkeypatch):
    """A module pair on ``tmp_path`` that the importer below can resolve by bare name."""
    _write(tmp_path, "guard_supplier.py", "PRESENT = 1\n\ndef function():\n    return 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("guard_supplier",):
        sys.modules.pop(name, None)
    yield tmp_path
    sys.modules.pop("guard_supplier", None)


def test_a_present_attribute_is_not_reported(two_modules):
    """The positive control: without this, a tool that reports nothing always passes."""
    script = _write(two_modules, "guard_ok.py",
                    "from guard_supplier import PRESENT, function\n"
                    "value = PRESENT\nother = function()\n")
    assert guard.missing_attributes(script) == []


def test_a_missing_attribute_is_reported_with_its_line(two_modules):
    """Partial application of the real defect: the name imports, the attribute does not exist.

    ``from module import function`` binds the *object*, so ``function.NOT_THERE`` is resolved
    against ``guard_supplier.function``, not against the module. The tool records that fully
    qualified target, which is what makes the message useful when several modules export
    similarly named helpers.
    """
    script = _write(two_modules, "guard_bad.py",
                    "from guard_supplier import function\n"
                    "value = function\n"
                    "broken = function.NOT_THERE\n")
    problems = guard.missing_attributes(script)
    assert len(problems) == 1
    problem = problems[0]
    assert (problem["alias"], problem["attribute"]) == ("function", "NOT_THERE")
    assert problem["module"] == "guard_supplier.function"
    assert problem["lineno"] == 3
    assert "absent" in problem["reason"]


def test_a_missing_attribute_on_an_imported_module_is_reported(two_modules):
    """The shape the C2 driver used: ``import module as alias`` then ``alias.missing``."""
    script = _write(two_modules, "guard_modattr.py",
                    "import guard_supplier as supplier\n"
                    "value = supplier.PRESENT\n"
                    "broken = supplier.NOT_THERE\n")
    problems = guard.missing_attributes(script)
    assert len(problems) == 1
    assert (problems[0]["module"], problems[0]["attribute"]) == ("guard_supplier", "NOT_THERE")


def test_an_unimportable_target_is_reported_not_swallowed(two_modules):
    """A target module that cannot import must be a reported failure, never a quiet pass."""
    script = _write(two_modules, "guard_nomodule.py",
                    "from guard_absent_module import thing\nvalue = thing.attr\n")
    problems = guard.missing_attributes(script)
    assert len(problems) == 1
    assert "import failed" in problems[0]["reason"]


def test_the_same_pair_is_reported_once_even_when_read_twice(two_modules):
    """Two reads of one missing attribute are one defect; the tool deduplicates by module/pair."""
    script = _write(two_modules, "guard_twice.py",
                    "from guard_supplier import function\n"
                    "a = function.NOT_THERE\n"
                    "b = function.NOT_THERE\n")
    assert len(guard.missing_attributes(script)) == 1


def test_write_targets_and_locals_are_out_of_scope(two_modules):
    """Only ``alias.attr`` reads count: assigning a local attribute must not be reported."""
    script = _write(two_modules, "guard_locals.py",
                    "from guard_supplier import function\n"
                    "class Holder:\n"
                    "    def __init__(self):\n"
                    "        self.whatever = 1\n"
                    "holder = Holder()\n"
                    "holder.also_missing = 2\n")
    assert guard.missing_attributes(script) == []


def test_exit_status_is_nonzero_when_a_driver_attribute_is_missing(two_modules, capsys):
    """The CLI contract the eventual CI step would rely on: 1 on a problem, 0 when clean."""
    good = _write(two_modules, "guard_cli_ok.py", "from guard_supplier import PRESENT\nv = PRESENT\n")
    bad = _write(two_modules, "guard_cli_bad.py",
                 "from guard_supplier import function\nv = function.MISSING\n")
    assert guard.main([str(good)]) == 0
    assert guard.main([str(bad)]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_the_real_c2_driver_has_no_missing_attributes():
    """The regression that matters: the driver whose defects cost attempt04 must stay clean.

    This is the check that would have caught ``curve.EVALUATION_LEADS`` and
    ``curve.check_provenance`` before two arms each trained for ten hours (decision 0098).
    """
    driver = ROOT / "scripts/study_r7_s3_derivative_matching_c2.py"
    assert guard.missing_attributes(driver) == []
