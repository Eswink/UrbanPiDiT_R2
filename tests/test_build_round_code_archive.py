"""Tests for ``tools/build_round_code_archive.py`` -- the round ``code.zip`` builder.

The builder exists because R2's archive was measured to be ``git archive <commit>`` plus
exactly the three local ``.zcode/`` runtime prefixes. These tests pin that shape and, more
importantly, pin the two refusals that make an archive replayable at all: an existing
destination is never overwritten, and a dirty tree is never archived under a commit name.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from tools import build_round_code_archive as builder

ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def test_the_extra_prefixes_are_the_three_measured_on_r2():
    assert builder.EXTRA_PREFIXES == (".zcode/plans/", ".zcode/workflow-drafts/",
                                      ".zcode/workflow-runs/")


def test_it_refuses_to_overwrite_an_existing_archive(tmp_path):
    destination = tmp_path / "code.zip"
    destination.write_bytes(b"not really a zip")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        builder.build(_head(), destination)
    assert destination.read_bytes() == b"not really a zip"


def test_it_refuses_a_dirty_tree(tmp_path):
    """An archive that names a commit while the tree differs cannot be replayed."""
    probe = ROOT / "data" / "unrelated_archive_probe.txt"
    try:
        probe.write_text("dirty\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="working tree is dirty"):
            builder.build(_head(), tmp_path / "code.zip")
    finally:
        probe.unlink(missing_ok=True)
    assert not (tmp_path / "code.zip").exists()


def test_the_extras_skip_anything_already_tracked():
    tracked = set(builder._git("ls-files", "-z").decode().split("\0"))
    for path in builder.tracked_extras():
        assert str(path.relative_to(ROOT)) not in tracked
        assert path.is_file() and not path.is_symlink()
    assert all(str(p.relative_to(ROOT)).startswith(builder.EXTRA_PREFIXES)
               for p in builder.tracked_extras())


def test_a_built_archive_contains_the_commit_tree_verbatim(tmp_path):
    destination = tmp_path / "code.zip"
    result = builder.build(_head(), destination)
    assert destination.is_file()
    assert result["entries"] == result["git_archive_entries"] + result["extra_entries"]
    assert result["commit"] == _head()
    tree = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.split()
    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
        for name in tree:
            assert name in names, name
        # The round's own driver must be inside: an archive that omits the code it names is
        # the failure mode this builder exists to prevent.
        assert "scripts/study_r7_s3_position_retention_65x65.py" in names
        assert "scripts/r7_s3_position_retention_protocol.py" in names
        # Nothing outside the repository root may be archived.
        assert not [name for name in names if name.startswith(("/", "../"))]


def test_the_content_digest_changes_with_content_and_not_with_ordering(tmp_path):
    """The digest covers name+bytes, so a reordered zip with the same content still differs
    only if a file changed -- that is what makes it usable as an identity."""
    first = builder.build(_head(), tmp_path / "a.zip")
    second = builder.build(_head(), tmp_path / "b.zip")
    assert first["content_digest_sha256"] == second["content_digest_sha256"]
    assert first["code_zip_sha256"] is not None and len(first["code_zip_sha256"]) == 64


def test_the_archive_carries_the_commit_as_its_zip_comment(tmp_path):
    """Every round's archive stage compares the comment to the frozen code_commit.

    Without it the archive is unusable: the first R4 archive stage failed with
    "archive commit differs from freeze" because this builder wrote no comment.
    """
    destination = tmp_path / "code.zip"
    result = builder.build(_head(), destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.comment.decode() == result["commit"] == _head()
