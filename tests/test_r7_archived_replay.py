"""The archived-replay overlay must be reconstructible and must refuse unsound layouts.

``tools/r7_archived_replay.py`` is the sanctioned remedy from decision 0065: when an
unrelated commit breaks a frozen round's live-glob inventory, the remaining seed is
finished against the round's own ``code.zip`` at its frozen absolute paths. Two things
must hold for that to be safe, and both are disproved here rather than asserted:

* the overlay's mount order is load-bearing -- if the real ``outputs/`` or ``.venv/``
  is captured *after* the archive covers the root, the rebind binds the archive's own
  empty directory and every result quietly lands nowhere;
* ``prepare`` must refuse to rewrite a tree that a live worker has mounted, because
  overwriting files under a running replay corrupts it mid-run.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import r7_archived_replay as replay  # noqa: E402

REPO = replay.REPO_ROOT
TREE = Path("/tmp/r7_archived_replay_test/tree")


def test_mount_script_captures_before_overlaying():
    script = replay.mount_script(tree=TREE, repo=REPO)
    assert replay.layout_faults(script, REPO) == []
    assert script.index(str(REPO / "outputs")) < script.index('mount --bind "$ARCHIVE_TREE"')
    assert script.index(".venv") < script.index('mount --bind "$ARCHIVE_TREE"')


def test_outputs_captured_after_overlay_is_a_fault():
    """The counter-proof: a late capture binds the archive's empty dir, losing results."""
    script = replay.mount_script(tree=TREE, repo=REPO)
    capture = f'mount --bind "{REPO / "outputs"}" "$CAP_OUT"\n'
    overlay = 'mount --bind "$ARCHIVE_TREE" "$REAL_ROOT"\n'
    faulty = script.replace(capture, "").replace(overlay, overlay + capture, 1)
    assert faulty.index(capture) > faulty.index(overlay)   # the fault is really present
    faults = replay.layout_faults(faulty, REPO)
    assert any("after the overlay" in fault for fault in faults), faults


def test_rebind_before_overlay_is_a_fault():
    """A rebind placed before the overlay is hidden by it -- also silent."""
    script = replay.mount_script(tree=TREE, repo=REPO)
    rebind = 'mount --bind "$CAP_VENV" "$REAL_ROOT/.venv"\n'
    hidden = script.replace(rebind, "").replace(
        'mount --bind "$ARCHIVE_TREE" "$REAL_ROOT"\n',
        rebind + 'mount --bind "$ARCHIVE_TREE" "$REAL_ROOT"\n',
    )
    assert any("before the overlay" in fault
               for fault in replay.layout_faults(hidden, REPO))


def test_missing_exec_is_a_fault():
    script = replay.mount_script(tree=TREE, repo=REPO).replace('exec "$@"', "true")
    assert any("never execs" in fault for fault in replay.layout_faults(script, REPO))


def test_run_refuses_an_unsound_overlay(monkeypatch, capsys):
    """``run`` must not exec a command on top of a broken overlay."""
    def broken(*, tree, repo):
        return 'set -euo pipefail\nmount --bind "$ARCHIVE_TREE" "$REAL_ROOT"\n'
    monkeypatch.setattr(replay, "mount_script", broken)
    assert replay.run(TREE, REPO, ["true"]) == 2
    assert "refusing to run an unsound overlay" in capsys.readouterr().err


def test_extract_rebuilds_and_creates_bind_points(tmp_path):
    archive = tmp_path / "code.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("training/example.py", "x = 1\n")
        handle.writestr("scripts/study_example.py", "y = 2\n")
    tree = tmp_path / "tree"
    assert replay.extract(archive, tree) == 2
    assert (tree / "training/example.py").is_file()
    assert (tree / "outputs").is_dir() and (tree / ".venv").is_dir()
    # a stale file from an earlier extraction must not survive a rebuild
    (tree / "stale.py").write_text("stale\n")
    replay.extract(archive, tree)
    assert not (tree / "stale.py").exists()


def test_prepare_refuses_a_tree_a_live_process_holds(tmp_path, monkeypatch, capsys):
    """The counter-proof for the in-use guard: an idle tree passes, a held one does not."""
    archive = tmp_path / "code.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.py", "x = 1\n")
    tree = tmp_path / "tree"

    monkeypatch.setattr(replay, "tree_in_use", lambda path: [])
    assert replay.main(["prepare", "--code-zip", str(archive), "--tree", str(tree)]) == 0
    stamp = (tree / "a.py").read_text()

    monkeypatch.setattr(replay, "tree_in_use", lambda path: [424242])
    assert replay.main(["prepare", "--code-zip", str(archive), "--tree", str(tree)]) == 3
    assert "mounted by live pid(s) [424242]" in capsys.readouterr().err
    assert (tree / "a.py").read_text() == stamp


def test_prepare_refuses_a_missing_archive(tmp_path, capsys):
    assert replay.main(["prepare", "--code-zip", str(tmp_path / "absent.zip"),
                        "--tree", str(tmp_path / "tree")]) == 2
    assert "no such archive" in capsys.readouterr().err


def test_tree_in_use_sees_this_process_mount_root(tmp_path):
    """A tree nobody has mounted reports no holders (the honest empty case)."""
    idle = tmp_path / "idle"
    idle.mkdir()
    assert replay.tree_in_use(idle) == []
