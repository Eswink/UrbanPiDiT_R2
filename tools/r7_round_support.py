"""Deterministic support for the campaign round driver (decisions 0044 / 0045).

This module holds the parts of a campaign round that must **not** be left to a model:
reading the current round state, gating cross-attempt seed reuse, computing ledger
arithmetic, running the validators, probing GPU co-residency and the commit gate.
Every function that touches the outside world takes an injectable runner, so the whole
module is testable without a GPU, without a network and without ``outputs/``.

No network client is imported here; the CI helper shells out to ``curl`` instead
(``tools/r7_ci_wait.py``), which also keeps R-016 and R-025 untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MASTER_REL = Path("docs/goals/main-model-climatology-campaign.md")
INDEX_REL = Path("docs/R7_EVIDENCE_INDEX.jsonl")
BRIEF_REL = Path("docs/R7_CANDIDATE_BRIEF.md")
GIT_TIMEOUT_SECONDS = 20
PYTEST_TIMEOUT_SECONDS = 900.0

STATE_RE = re.compile(r"<!--\s*campaign-state:\s*(\{.*?\})\s*-->")
ROUND_SPEC_RE = re.compile(r"<!--\s*round-spec:\s*(\{.*?\})\s*-->")
LEDGER_ROW_RE = re.compile(
    r"^\|\s*(?P<name>[^|]+?)\s*\|\s*\*{0,2}(?P<gpu>\d+(?:\.\d+)?)\*{0,2}\s*\|"
    r"\s*\*{0,2}(?P<cum>[\d.]+|—|-)\*{0,2}\s*\|\s*(?P<evidence>[^|]+?)\s*\|\s*$",
    re.M,
)
RECORD_REF_RE = re.compile(r"record:([A-Za-z0-9_.-]+)")
OUTPUT_STEM_RE = re.compile(r"[a-z0-9_]+")
PROTECTED_BRANCHES = ("main", "master")

REQUIRED_STATE_KEYS = (
    "current_node",
    "previous_node",
    "current_round_goal",
    "previous_round_goal",
    "previous_round_evidence",
    "cap_gpu_h",
    "used_gpu_h",
    "remaining_gpu_h",
)
REQUIRED_SPEC_KEYS = ("arm", "driver", "output_stem", "control_record", "seeds")
NEXT_ACTION = "下一动作"

#: Fields that must be equal, one by one, before a finished seed may be reused across
#: attempts (decision 0045). A whitelist on purpose: a new protocol field does not
#: silently become "reusable".
REUSE_PROTOCOL_FIELDS = (
    "protocol_sha256",
    "code_zip_sha256",
    "code_commit",
    "train_data_identity",
    "val_data_identity",
    "source_inventory_sha256",
    "supervision",
    "updates",
    "reasoning_steps",
    "physical_weights",
    "lr",
    "warmup",
    "weight_decay",
    "batch_size",
    "clip",
    "per_seed_seconds",
    "planned_seconds",
    "hard_cap_seconds",
)
REUSE_RECEIPT_FIELDS = ("reasoning_steps", "updates")

VALIDATOR_COMMANDS: tuple[tuple[str, ...], ...] = (
    (PYTHON, "tools/check_conventions.py", "--quiet"),
    (
        PYTHON,
        "tools/check_campaign_state.py",
        "--campaign",
        "docs/goals/main-model-climatology-campaign.md",
    ),
    (
        PYTHON,
        "tools/verify_r7_evidence_index.py",
        "--index",
        "docs/R7_EVIDENCE_INDEX.jsonl",
        "--check-brief",
        "docs/R7_CANDIDATE_BRIEF.md",
    ),
)
GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=uuid,memory.free,memory.total",
    "--format=csv,noheader,nounits",
)


class RoundError(RuntimeError):
    """A round-level precondition failed; the caller must stop, not improvise."""


def default_runner(
    argv: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    if timeout is None and argv and argv[0] == "git":
        timeout = GIT_TIMEOUT_SECONDS
    try:
        completed = subprocess.run(  # fixed argv, no shell
            list(argv),
            cwd=str(cwd or ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        # A stage that runs past its frozen limit is a failed stage, not an orchestrator
        # crash: the caller records exit_code != 0 and stops the round, exactly as for any
        # other non-zero exit. Letting this escape killed a whole `stage-seeds` call on
        # 2026-10-22 with a traceback instead of a structured failure.
        def decode(value):
            if value is None:
                return ""
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else value

        return {
            "argv": list(argv),
            "exit_code": 124,
            "timed_out": True,
            "timeout_seconds": timeout,
            "stdout": decode(expired.stdout),
            "stderr": decode(expired.stderr) + f"\nstage exceeded its frozen limit of {timeout} s\n",
        }
    return {
        "argv": list(argv),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def campaign_state(master: Path | None = None) -> dict[str, Any]:
    """Parse the machine-readable ``<!-- campaign-state: ... -->`` block."""
    matches = STATE_RE.findall(_text(master or (ROOT / MASTER_REL)))
    if len(matches) != 1:
        raise RoundError(f"expected exactly one campaign-state block, found {len(matches)}")
    body = json.loads(matches[0])
    missing = [key for key in REQUIRED_STATE_KEYS if key not in body]
    if missing:
        raise RoundError(f"campaign-state is missing required keys: {missing}")
    return body


def round_spec(goal: Path) -> dict[str, Any]:
    """Parse the ``<!-- round-spec: ... -->`` block that carries the round's intent.

    The workflow cannot hand dynamic argv to a command, so every runtime value a round
    needs lives in this block and is read from disk instead.
    """
    matches = ROUND_SPEC_RE.findall(_text(goal))
    if len(matches) != 1:
        raise RoundError(f"{goal}: expected exactly one round-spec block, found {len(matches)}")
    body = json.loads(matches[0])
    missing = [key for key in REQUIRED_SPEC_KEYS if key not in body]
    if missing:
        raise RoundError(f"{goal}: round-spec is missing {missing}")
    return body


def ledger_rows(master: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    for match in LEDGER_ROW_RE.finditer(_text(master or (ROOT / MASTER_REL))):
        if match.group("cum").strip() in {"—", "-"}:
            continue
        rows.append(
            {
                "name": match.group("name").strip(),
                "gpu_hours": float(match.group("gpu")),
                "cumulative": float(match.group("cum")),
                "evidence": match.group("evidence").strip(),
                "record_ids": RECORD_REF_RE.findall(match.group("evidence")),
            }
        )
    return rows


def index_records(index: Path | None = None) -> list[dict[str, Any]]:
    path = index or (ROOT / INDEX_REL)
    return [json.loads(line) for line in _text(path).splitlines() if line.strip()]


def last_next_action(master: Path | None = None) -> str:
    """Return the tail of §8 after the last ``下一动作`` marker."""
    text = _text(master or (ROOT / MASTER_REL))
    position = text.rfind(NEXT_ACTION)
    if position < 0:
        raise RoundError("the master plan has no 下一动作 marker")
    return text[position:].strip()


def next_attempt_dir(stem: str, outputs: Path | None = None) -> Path:
    """Return the next free ``outputs/<stem>_attempt<NN>`` directory.

    Deterministic on purpose: the workflow never computes an output path itself.
    """
    root = Path(outputs or (ROOT / "outputs"))
    if not OUTPUT_STEM_RE.fullmatch(stem):
        raise RoundError(f"output_stem must match [a-z0-9_]+: {stem!r}")
    numbers = []
    for path in root.glob(f"{stem}_attempt*"):
        suffix = path.name.rsplit("attempt", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            numbers.append(int(suffix))
    return root / f"{stem}_attempt{max(numbers, default=0) + 1:02d}"


def _field_diff(left: dict[str, Any], right: dict[str, Any], fields: Iterable[str]) -> list[str]:
    return [field for field in fields if left.get(field) != right.get(field)]


def reuse_decision(
    previous_protocol: dict[str, Any] | None,
    new_protocol: dict[str, Any],
    previous_receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Decide which finished seeds may be reused across attempts (decision 0045).

    Reuse requires every field in :data:`REUSE_PROTOCOL_FIELDS` to be equal and each
    seed's receipt to agree on the seed itself. Anything else re-runs that seed.
    """
    if previous_protocol is None:
        return {"reusable": [], "refused": {}, "reason": "no previous attempt"}
    drift = _field_diff(previous_protocol, new_protocol, REUSE_PROTOCOL_FIELDS)
    if drift:
        return {"reusable": [], "refused": {}, "reason": f"protocol drift: {sorted(drift)}"}
    reusable, refused = [], {}
    for seed, receipt in sorted(previous_receipts.items(), key=lambda item: str(item[0])):
        bad = [] if str(receipt.get("seed")) == str(seed) else ["seed"]
        bad += _field_diff(receipt, new_protocol, REUSE_RECEIPT_FIELDS)
        if bad:
            refused[seed] = f"receipt drift: {sorted(bad)}"
        else:
            reusable.append(seed)
    return {"reusable": reusable, "refused": refused, "reason": "identity matched"}


def ledger_plan(rows: Sequence[dict[str, Any]], cap_gpu_h: float, new_gpu_h: float) -> dict[str, Any]:
    """Compute the ledger arithmetic that must hold after a new row is added (C-02)."""
    previous_total = round(sum(float(row["gpu_hours"]) for row in rows), 6)
    new_total = round(previous_total + float(new_gpu_h), 6)
    return {
        "rows_before": len(rows),
        "previous_total_gpu_h": previous_total,
        "new_row_gpu_h": float(new_gpu_h),
        "new_total_gpu_h": new_total,
        "cap_gpu_h": float(cap_gpu_h),
        "remaining_gpu_h": round(float(cap_gpu_h) - new_total, 6),
    }


def gpu_headroom(
    uuid: str,
    *,
    estimated_peak_bytes: int,
    owned_reserved_peak_bytes: int = 0,
    margin_bytes: int,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read-only co-residency probe; mirrors ``training.r7_s3_v3_screen.gpu_gate``.

    Never signals another process. When ``nvidia-smi`` is unavailable (CI, or a host
    without the declared card) the record comes back ``available: False``.
    """
    probe = runner or default_runner
    result = probe(list(GPU_QUERY))
    base = {"uuid": uuid, "source": "nvidia-smi read-only"}
    if result["exit_code"] != 0:
        return {**base, "available": False, "passed": False}
    rows = [line.split(", ") for line in result["stdout"].strip().splitlines() if line.strip()]
    match = next((row for row in rows if row[0].strip() == uuid), None)
    if match is None:
        return {**base, "available": False, "passed": False}
    free_mib, total_mib = int(match[1]), int(match[2])
    required_mib = max(estimated_peak_bytes, owned_reserved_peak_bytes) // 2**20 + margin_bytes // 2**20
    return {
        **base,
        "available": True,
        "free_bytes": free_mib * 2**20,
        "total_bytes": total_mib * 2**20,
        "required_bytes": required_mib * 2**20,
        "owned_reserved_peak_bytes": owned_reserved_peak_bytes,
        "passed": bool(free_mib >= required_mib),
    }


def run_validators(
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the deterministic validators; a non-zero exit means the round must stop."""
    base = Path(root or ROOT)
    run = runner or default_runner
    results = []
    for command in VALIDATOR_COMMANDS:
        result = run(list(command), base)
        tail = (result["stdout"] + result["stderr"]).strip().splitlines()[-3:]
        results.append({"command": " ".join(command), "exit_code": result["exit_code"], "tail": tail})
    return {"passed": all(item["exit_code"] == 0 for item in results), "results": results}


def run_pytest(
    paths: Sequence[str] | None,
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the round's own test module plus the convention self-test.

    The paths come from the round-spec so the workflow never has to build argv.
    """
    if not paths:
        return {"passed": True, "skipped": True, "reason": "round-spec declares no test_path"}
    base = Path(root or ROOT)
    run = runner or default_runner
    argv = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths, "tests/test_check_conventions.py"]
    result = run(argv, base, None, PYTEST_TIMEOUT_SECONDS)
    tail = (result["stdout"] + result["stderr"]).strip().splitlines()[-3:]
    return {"passed": result["exit_code"] == 0, "exit_code": result["exit_code"],
            "command": " ".join(argv), "tail": tail}


def commit_guard(
    paths: Sequence[str],
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The commit gate the workflow runs itself, since subagents may not run hooks.

    The paths are simulated as commit content via ``--paths``; a non-zero exit means
    the commit would turn CI red and must not be made.
    """
    base = Path(root or ROOT)
    run = runner or default_runner
    if not paths:
        raise RoundError("commit_guard needs at least one path")
    result = run([PYTHON, "tools/check_conventions.py", "--quiet", "--paths", *paths], base)
    tail = (result["stdout"] + result["stderr"]).strip().splitlines()[-5:]
    return {"paths": list(paths), "exit_code": result["exit_code"], "passed": result["exit_code"] == 0, "tail": tail}


def state_report(root: Path | None = None, runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Everything the reconnaissance step needs, in one read-only JSON document."""
    base = Path(root or ROOT)
    run = runner or default_runner
    rows = ledger_rows(base / MASTER_REL)
    records = index_records(base / INDEX_REL)
    return {
        "head": run(["git", "rev-parse", "HEAD"], base)["stdout"].strip(),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], base)["stdout"].strip(),
        "campaign_state": campaign_state(base / MASTER_REL),
        "ledger_rows": len(rows),
        "ledger_total_gpu_h": round(sum(row["gpu_hours"] for row in rows), 6),
        "last_ledger_row": rows[-1] if rows else None,
        "index_records": len(records),
        "last_record_ids": [record.get("record_id") for record in records[-3:]],
        "next_action": last_next_action(base / MASTER_REL),
        "goal_briefs": sorted(path.name for path in (base / "docs/goals").glob("*.md")),
    }
