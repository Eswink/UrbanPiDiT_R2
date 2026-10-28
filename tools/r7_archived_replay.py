"""Replay an archived round's code at its frozen absolute paths (decision 0065).

A round freezes ``execution_files_sha256`` over a **live** inventory: the driver
re-globs ``training/``, ``model/``, ``data/`` on every stage. Any unrelated commit
that adds a ``.py`` there makes ``validate_protocol`` reject the frozen round before
it can finish a remaining seed. The sanctioned remedy is not to relax the comparison
(``AGENTS.md`` forbids editing run-identity comparison logic) and not to re-freeze
(``protocol_sha256`` and ``code_zip_sha256`` sit in decision 0045's reuse whitelist,
so a re-freeze would retrain every seed). It is to reconstruct the tree the round was
frozen against -- its own ``code.zip`` -- and run inside it at the same absolute paths.

That reconstruction cannot happen by extracting elsewhere: ``validate_protocol`` also
compares recorded **absolute** source paths (``source.nc`` and friends), so the code
must appear at ``/data/esw/UrbanPiDiT_R2``. An unprivileged user+mount namespace gives
exactly that: bind the extracted archive over the real repo root, then bind the real
``outputs/`` and ``.venv/`` back on top so results land where the round expects them
and the interpreter stays the project's. Nothing in the real tree is modified; the
overlay is private to the namespace and evaporates when it exits.

Layout order matters and is asserted rather than assumed: the real ``outputs/`` and
``.venv/`` are captured into temporary mounts **before** the archive covers the root,
because after the overlay those paths would resolve inside the archive.

    python tools/r7_archived_replay.py prepare --code-zip outputs/<round>/code.zip
    python tools/r7_archived_replay.py verify --out outputs/<round>
    python tools/r7_archived_replay.py run -- .venv/bin/python tools/r7_round.py stage reading

``prepare`` refuses to touch a tree that a live process has mounted (overwriting
files under a running worker would corrupt the very replay it is serving).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREE = Path("/tmp/r7_archived_replay/tree")
REAL_BINDINGS = ("outputs", ".venv")
CAPTURE_OUT = "/tmp/r7_replay_real_outputs"
CAPTURE_VENV = "/tmp/r7_replay_real_venv"


def _repo_root(repo: Path | None = None) -> Path:
    return Path(repo).resolve() if repo is not None else REPO_ROOT


def tree_in_use(tree: Path) -> list[int]:
    """PIDs holding a mount rooted at ``tree``; empty when the tree is idle."""
    target = str(Path(tree).resolve())
    holders: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "mountinfo").read_text()
        except OSError:
            continue
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 5 and target in (fields[3], fields[4]):
                holders.append(int(entry.name))
                break
    return sorted(set(holders))


def extract(code_zip: Path, tree: Path) -> int:
    """Unpack ``code_zip`` into a fresh ``tree``; returns the member count."""
    tree = Path(tree)
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    with zipfile.ZipFile(code_zip) as archive:
        members = archive.namelist()
        archive.extractall(tree)
    for name in REAL_BINDINGS:
        (tree / name).mkdir(parents=True, exist_ok=True)
    return len(members)


def mount_script(*, tree: Path, repo: Path) -> str:
    """The in-namespace bash that builds the overlay, in its load-bearing order."""
    lines = [
        "set -euo pipefail",
        f'ARCHIVE_TREE="{Path(tree)}"',
        f'REAL_ROOT="{Path(repo)}"',
        f'CAP_OUT="{CAPTURE_OUT}"',
        f'CAP_VENV="{CAPTURE_VENV}"',
        # 1. capture the real directories before the overlay hides them
        'mkdir -p "$CAP_OUT" "$CAP_VENV"',
        f'mount --bind "{Path(repo) / "outputs"}" "$CAP_OUT"',
        f'mount --bind "{Path(repo) / ".venv"}" "$CAP_VENV"',
        # 2. cover the frozen absolute path with the archived tree
        'mkdir -p "$ARCHIVE_TREE/outputs" "$ARCHIVE_TREE/.venv"',
        'mount --bind "$ARCHIVE_TREE" "$REAL_ROOT"',
        # 3. hand the real results and interpreter back through the overlay
        'mount --bind "$CAP_OUT" "$REAL_ROOT/outputs"',
        'mount --bind "$CAP_VENV" "$REAL_ROOT/.venv"',
        'cd "$REAL_ROOT"',
        'exec "$@"',
    ]
    return "\n".join(lines) + "\n"


def layout_faults(script: str, repo: Path) -> list[str]:
    """Ordering violations in a mount script; empty means the overlay is sound.

    This is the gate ``tests/test_r7_archived_replay.py`` disproves: capture the real
    ``outputs/`` *after* the overlay and the rebind becomes a bind of the archive's own
    empty directory, silently losing every result -- which the check must report.
    """
    repo = Path(repo)
    faults: list[str] = []
    overlay = script.find('mount --bind "$ARCHIVE_TREE" "$REAL_ROOT"')
    if overlay < 0:
        return ["the archive tree is never bind-mounted over the repo root"]

    landmarks = {
        "outputs capture": script.find(f'mount --bind "{repo / "outputs"}" "$CAP_OUT"'),
        ".venv capture": script.find(f'mount --bind "{repo / ".venv"}" "$CAP_VENV"'),
        "outputs rebind": script.find('mount --bind "$CAP_OUT" "$REAL_ROOT/outputs"'),
        ".venv rebind": script.find('mount --bind "$CAP_VENV" "$REAL_ROOT/.venv"'),
    }
    for label in ("outputs capture", ".venv capture"):
        if landmarks[label] < 0:
            faults.append(f"the real {label.split()[0]} is never captured")
        elif landmarks[label] > overlay:
            faults.append(f"the {label} happens after the overlay, so it captures "
                          f"the archive's empty directory instead of the real one")
    for label in ("outputs rebind", ".venv rebind"):
        if landmarks[label] < 0:
            faults.append(f"the real {label.split()[0]} is never rebound through the overlay")
        elif landmarks[label] < overlay:
            faults.append(f"the {label} happens before the overlay, so it is hidden by it")

    if 'cd "$REAL_ROOT"' not in script:
        faults.append("the script never enters the repo root")
    if 'exec "$@"' not in script:
        faults.append("the script never execs the requested command")
    return faults


def run(tree: Path, repo: Path, command: list[str]) -> int:
    script = mount_script(tree=tree, repo=repo)
    faults = layout_faults(script, repo)
    if faults:
        print("refusing to run an unsound overlay:", file=sys.stderr)
        for fault in faults:
            print(f"  - {fault}", file=sys.stderr)
        return 2
    argv = ["unshare", "--user", "--map-root-user", "--mount", "--propagation", "private",
            "--", "bash", "-c", script, "bash", *command]
    return subprocess.call(argv, cwd=repo)


def verify(tree: Path, repo: Path, out: Path) -> int:
    probe = (
        "import sys; sys.path.insert(0, {repo!r})\n"
        "from pathlib import Path\n"
        "from scripts.study_r7_s3_fy_year_coverage_matched import execution_files, validate_protocol\n"
        "print('execution_files():', len(execution_files()))\n"
        "body = validate_protocol(Path({out!r}))\n"
        "print('VALIDATE_PROTOCOL: PASS', body['protocol_sha256'][:12])\n"
    ).format(repo=str(repo), out=str(out))
    return run(tree, repo, [".venv/bin/python", "-c", probe])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    prep = sub.add_parser("prepare", help="extract an archived code.zip into a replay tree")
    prep.add_argument("--code-zip", required=True, type=Path)
    prep.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    prep.add_argument("--force", action="store_true",
                      help="replace a tree even though a live process is using it")

    run_p = sub.add_parser("run", help="run a command inside the replay overlay")
    run_p.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    run_p.add_argument("--repo", type=Path, default=None)
    run_p.add_argument("command", nargs=argparse.REMAINDER)

    ver = sub.add_parser("verify", help="check validate_protocol inside the replay")
    ver.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    ver.add_argument("--repo", type=Path, default=None)
    ver.add_argument("--out", required=True, type=Path)

    args = parser.parse_args(argv)
    tree = Path(args.tree)
    repo = _repo_root(getattr(args, "repo", None))

    if args.action == "prepare":
        if not args.code_zip.is_file():
            print(f"no such archive: {args.code_zip}", file=sys.stderr)
            return 2
        holders = tree_in_use(tree) if tree.exists() else []
        if holders and not args.force:
            print(f"refusing to replace {tree}: mounted by live pid(s) {holders}", file=sys.stderr)
            print("a running worker would be corrupted mid-replay; wait or pass --force",
                  file=sys.stderr)
            return 3
        count = extract(Path(args.code_zip), tree)
        print(f"extracted {count} members -> {tree}")
        print(f"mountpoints created: {', '.join(REAL_BINDINGS)}")
        return 0
    if args.action == "run":
        command = [item for item in args.command if item != "--"]
        if not command:
            print("no command given", file=sys.stderr)
            return 2
        return run(tree, repo, command)
    return verify(tree, repo, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
