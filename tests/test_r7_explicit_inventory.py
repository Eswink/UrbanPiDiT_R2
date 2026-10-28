"""Decision 0067: an execution inventory built from the import closure must not drift.

The defect this prevents is concrete and already cost a round: R1's ``execution_files()``
globbed ``training/``, ``model/`` and ``data/`` live, so when an unrelated round added two
modules to ``data/`` a seed that had not started failed ``validate_protocol`` in 20.76 s,
after nine GPU-hours had already been spent on the seeds that did finish. The closure
replacement is only worth adopting if it is *provably* immune to that, so the
counter-proof here adds a module to ``data/`` and asserts the inventory does not move --
while the old glob, measured on the same tree, does move.

Fixtures are synthetic trees in ``tmp_path``: these tests must run on CI, where the real
round's multi-gigabyte stores are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import r7_explicit_inventory as inventory  # noqa: E402

REPO = inventory.REPO_ROOT


def _write(root: Path, relative: str, body: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _tiny_round(root: Path) -> Path:
    """A miniature round: an entry script reaching one helper through one package hop."""
    for package in ("data", "model", "training", "scripts"):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    _write(root, "training/harness.py", "import math\n\nCONST = math.pi\n")
    _write(root, "data/loader.py", "from training.harness import CONST\n\nVALUE = CONST\n")
    return _write(root, "scripts/entry.py",
                  "from data.loader import VALUE\n\nprint(VALUE)\n")


def test_closure_follows_transitive_imports(tmp_path):
    _tiny_round(tmp_path)
    closure = inventory.import_closure("scripts/entry.py", repo=tmp_path)
    assert "scripts/entry.py" in closure
    assert "data/loader.py" in closure
    assert "training/harness.py" in closure


def test_closure_ignores_package_init_only(tmp_path):
    """``__init__.py`` files appear only if something actually imports the package."""
    _tiny_round(tmp_path)
    closure = inventory.import_closure("scripts/entry.py", repo=tmp_path)
    # the entry is inside scripts/, so scripts is reachable; nothing reaches model/
    assert "model/__init__.py" not in closure
    assert not [name for name in closure if name.startswith("model/")]


def test_adding_an_unrelated_module_does_not_change_the_inventory(tmp_path):
    """The counter-proof decision 0067 requires: this is the R1 failure, made harmless."""
    _tiny_round(tmp_path)
    before = inventory.explicit_execution_files("scripts/entry.py", repo=tmp_path)
    _write(tmp_path, "data/preprocess/urban_static_sources.py", "X = 1\n")
    _write(tmp_path, "training/unused_extra.py", "Y = 2\n")
    after = inventory.explicit_execution_files("scripts/entry.py", repo=tmp_path)
    assert after == before
    assert inventory.inventory_faults(before, "scripts/entry.py", repo=tmp_path) == []


def test_the_old_glob_would_have_drifted_on_the_same_tree(tmp_path):
    """Establishes the defect was real: the glob shape R1 used moves where the closure does not."""
    _tiny_round(tmp_path)

    def glob_inventory() -> list[str]:
        return sorted(str(path.relative_to(tmp_path)) for directory in ("training", "model", "data")
                      for path in (tmp_path / directory).rglob("*.py"))

    before = glob_inventory()
    _write(tmp_path, "data/preprocess/unrelated.py", "X = 1\n")
    assert glob_inventory() != before
    assert len(glob_inventory()) == len(before) + 1


def test_a_new_transitive_import_does_move_the_inventory(tmp_path):
    """The other direction: a real code change must be visible, or the inventory is blind."""
    _tiny_round(tmp_path)
    before = inventory.explicit_execution_files("scripts/entry.py", repo=tmp_path)
    _write(tmp_path, "training/extra.py", "Z = 3\n")
    _write(tmp_path, "data/loader.py",
           "from training.harness import CONST\nfrom training.extra import Z\n\nVALUE = CONST + Z\n")
    after = inventory.explicit_execution_files("scripts/entry.py", repo=tmp_path)
    assert "training/extra.py" in after
    assert after != before


def test_declaring_a_module_the_closure_cannot_reach_is_a_fault(tmp_path):
    """A hand-written stale line must be reported, not silently accepted."""
    _tiny_round(tmp_path)
    declared = inventory.explicit_execution_files("scripts/entry.py", repo=tmp_path)
    faults = inventory.inventory_faults(declared + ["training/ghost.py"],
                                        "scripts/entry.py", repo=tmp_path)
    assert any("not in the closure" in fault for fault in faults)


def test_omitting_a_closure_module_is_a_fault(tmp_path):
    """An identity gap is the dangerous direction: the declaration must cover everything."""
    _tiny_round(tmp_path)
    declared = [name for name in inventory.explicit_execution_files("scripts/entry.py", repo=tmp_path)
                if not name.startswith("training/")]
    faults = inventory.inventory_faults(declared, "scripts/entry.py", repo=tmp_path)
    assert any("closure module not declared" in fault for fault in faults)


def test_extra_names_a_file_the_closure_cannot_see(tmp_path):
    """Decision 0067: a runtime-read file is declared literally, and must exist."""
    _tiny_round(tmp_path)
    _write(tmp_path, "outputs/round/protocol.json", "{}\n")
    declared = inventory.explicit_execution_files(
        "scripts/entry.py", extra=["outputs/round/protocol.json"], repo=tmp_path)
    assert "outputs/round/protocol.json" in declared
    with pytest.raises(FileNotFoundError):
        inventory.explicit_execution_files(
            "scripts/entry.py", extra=["outputs/round/absent.json"], repo=tmp_path)


def test_a_missing_entry_is_an_error_not_an_empty_inventory(tmp_path):
    """An empty list would declare 'this round executes nothing', which is never true."""
    _tiny_round(tmp_path)
    with pytest.raises(FileNotFoundError):
        inventory.import_closure("scripts/absent.py", repo=tmp_path)


def test_an_entry_outside_the_packages_is_rejected(tmp_path):
    _tiny_round(tmp_path)
    _write(tmp_path, "tools/helper.py", "A = 1\n")
    with pytest.raises(FileNotFoundError):
        inventory.import_closure("tools/helper.py", repo=tmp_path)


def test_archived_snapshots_never_enter_a_closure(tmp_path):
    _tiny_round(tmp_path)
    _write(tmp_path, "training/legacy_v6/old.py", "OLD = 1\n")
    _write(tmp_path, "training/legacy_v531_full/older.py", "OLDER = 1\n")
    closure = inventory.import_closure("scripts/entry.py", repo=tmp_path)
    assert not [name for name in closure if "legacy" in name]


def test_the_real_round_closure_excludes_tools_and_tests():
    """On the real tree: the closure is strictly narrower than the glob and never leaks."""
    entry = "scripts/study_r7_s3_fy_year_coverage_matched.py"
    if not (REPO / entry).is_file():
        pytest.skip("real round driver is not present in this checkout")
    closure = inventory.import_closure(entry, repo=REPO)
    assert closure, "the real round must have a non-empty closure"
    assert not [name for name in closure if name.startswith(("tools/", "tests/"))]
    assert not [name for name in closure if "legacy" in name]
