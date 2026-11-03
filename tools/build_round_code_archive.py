"""Build a round's ``code.zip`` exactly the way R2's archived one is built.

R2's ``outputs/r7_s3_anchor_feedback_65x65_code.zip`` was measured to be
``git archive <commit>`` (1,570 entries) plus 99 local ``.zcode/`` runtime-state entries
(plans, workflow drafts, workflow runs) -- so the archive carries both the code identity a
replay needs and the planning state that explains how the round was steered. Reproducing that
shape here rather than inventing a new one keeps the two rounds' archives comparable.

Read-only with respect to the repository: it runs ``git archive`` and copies ``.zcode/``
entries into a new file under ``outputs/``. It refuses an existing destination and refuses a
dirty tree, because an archive that claims a commit while the tree differs cannot be replayed.

Usage::

    python tools/build_round_code_archive.py --commit HEAD \
        --out outputs/r7_s3_position_retention_65x65_code.zip
"""
from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The local runtime-state prefixes R2's archive carried on top of ``git archive``. They are
#: untracked by design (``docs/rules/artifact-storage.md``), and they explain the round's
#: steering, so the archive keeps them next to the code that produced the readings. The set was
#: measured on R2's archive (96 files, exactly these three prefixes) rather than assumed.
EXTRA_PREFIXES = (".zcode/plans/", ".zcode/workflow-drafts/", ".zcode/workflow-runs/")


def _git(*args: str) -> bytes:
    finished = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if finished.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: "
                           f"{finished.stderr.decode(errors='replace')[:500]}")
    return finished.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_extras() -> list[Path]:
    """Local ``.zcode/`` files that exist, are not tracked, and pass the exclude list."""
    tracked = set(_git("ls-files", "-z").decode().split("\0"))
    found = []
    for prefix in EXTRA_PREFIXES:
        base = ROOT / prefix
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = str(path.relative_to(ROOT))
            if relative in tracked:
                continue
            found.append(path)
    return found


def build(commit: str, destination: Path) -> dict:
    """Write the archive and return its identity. Refuses an existing file."""
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite an existing archive: {destination}")
    status = _git("status", "--porcelain").decode()
    if status.strip():
        raise RuntimeError("the working tree is dirty; archive a clean tree so the recorded "
                           f"commit matches its contents:\n{status[:2000]}")
    resolved = _git("rev-parse", commit).decode().strip()

    archive_bytes = _git("archive", "--format=zip", resolved)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        archive_names = source.namelist()
        contents = {name: source.read(name) for name in archive_names}

    extras = tracked_extras()
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as target:
        for name, content in contents.items():
            target.writestr(name, content)
            digest.update(name.encode() + b"\0" + content)
        for path in extras:
            name = str(path.relative_to(ROOT))
            content = path.read_bytes()
            target.writestr(name, content)
            digest.update(name.encode() + b"\0" + content)
        # The archive's own identity is the zip comment. Every round's archive stage compares
        # it against the frozen ``code_commit`` and refuses a mismatch, so an archive without
        # it cannot start a run at all -- the failure this builder itself hit once.
        target.comment = resolved.encode()

    with zipfile.ZipFile(destination) as written:
        entries = len(written.namelist())
    return {"path": str(destination), "commit": resolved,
            "commit_matches_head": resolved == _git("rev-parse", "HEAD").decode().strip(),
            "git_archive_entries": len(archive_names), "extra_entries": len(extras),
            "entries": entries, "code_zip_sha256": _sha256(destination),
            "content_digest_sha256": digest.hexdigest(),
            "extra_prefixes": list(EXTRA_PREFIXES)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.commit, args.out if args.out.is_absolute() else ROOT / args.out)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
