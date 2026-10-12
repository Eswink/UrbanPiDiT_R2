"""Multi-card device selection for a co-resident GPU round (decision 0047).

The registered arms froze **one** physical UUID and their protocols still validate
against it, so ``training/r7_s3_v3_screen.py`` is left untouched: ``gpu_gate`` and
``pin_declared_gpu`` keep meaning exactly what they meant. This module adds only what a
round needs to spread **its own** seeds over both cards without an operator picking one,
and to leave a co-resident neighbour room to grow between the spawn-time probe and the
end of training.

Read-only throughout: it queries ``nvidia-smi`` and never signals another process. A
neighbour PID is recorded, never touched (R-054 / decision 0026).
"""
from __future__ import annotations

import os
import subprocess

MODE_SINGLE = "single"
MODE_DUAL = "dual"
MODES = (MODE_SINGLE, MODE_DUAL)

#: Declared reserve on top of the arm's own footprint and margin, for a co-resident
#: neighbour that grows *after* the spawn-time probe.
#:
#: Evidence (2026-10-12): the probe passed (free 15.29 GiB >= the 9 GiB requirement) and
#: the seed still died with CUDA OOM 32.5 min in, because the neighbour on the declared
#: card grew from ~8.1 GiB to 14.32 GiB (+~6.2 GiB) while our own process sat at 9.18 GiB.
#: A spawn-time read can never bound a neighbour's future growth, so the gate now asks
#: for a declared reserve as well. This is a bounded judgement, **not** a guarantee: a
#: larger growth still fails, and that failure is registered as a resource failure.
NEIGHBOUR_RESERVE_BYTES = 6 * 2**30

GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=uuid,memory.free,memory.total",
    "--format=csv,noheader,nounits",
)


def _default_probe(argv):
    result = subprocess.run(list(argv), capture_output=True, text=True, timeout=60)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def protocol_fields(mode, *, single_uuid, reserve_bytes=NEIGHBOUR_RESERVE_BYTES, runner=None):
    """The frozen device block: mode, the candidate card list, and the reserve."""
    if mode not in MODES:
        raise ValueError(f"device mode must be one of {MODES}, got {mode!r}")
    if mode == MODE_SINGLE:
        uuids = [single_uuid]
    else:
        uuids = card_order(single_uuid, runner=runner)
    return {"device_mode": mode, "gpu_uuids": uuids,
            "neighbour_reserve_bytes": int(reserve_bytes)}


def card_order(single_uuid, *, runner=None):
    """Every visible card, the round's declared one first. Read-only."""
    rows = read_cards(runner=runner)
    if rows is None:
        return [single_uuid]
    uuids = [row["uuid"] for row in rows]
    if single_uuid not in uuids:
        raise RuntimeError(f"declared GPU UUID {single_uuid} not visible to nvidia-smi")
    return [single_uuid] + [uuid for uuid in uuids if uuid != single_uuid]


def read_cards(*, runner=None):
    """``nvidia-smi`` inventory as a list of rows, or ``None`` when unavailable."""
    probe = runner or _default_probe
    result = probe(list(GPU_QUERY))
    if result["exit_code"] != 0:
        return None
    rows = []
    for line in result["stdout"].strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            raise ValueError("malformed GPU UUID/headroom inventory")
        rows.append({"uuid": fields[0], "free_bytes": int(fields[1]) * 2**20,
                     "total_bytes": int(fields[2]) * 2**20})
    return rows


def seed_order(uuids, position):
    """Deterministic card preference for the seed at ``position`` in the round's list.

    Rotation (not a fixed first card) is what stops every seed from piling onto card 0
    when the list is frozen once and three seeds have to share two cards.
    """
    uuids = list(uuids)
    if not uuids:
        raise ValueError("the frozen device list is empty")
    offset = int(position) % len(uuids)
    return uuids[offset:] + uuids[:offset]


def gate_candidates(
    uuids,
    *,
    estimated_peak_bytes,
    owned_reserved_peak_bytes=0,
    margin_bytes,
    reserve_bytes,
    runner=None,
):
    """Probe the candidate cards in order and report the first that clears the gate.

    ``required = max(estimate, measured peak) + margin + reserve``. Every card is probed
    read-only and every observation is kept, so a refusal says which card was short and
    by how much instead of just "blocked".
    """
    uuids = list(uuids)
    required = (max(int(estimated_peak_bytes), int(owned_reserved_peak_bytes))
                + int(margin_bytes) + int(reserve_bytes))
    rows = read_cards(runner=runner)
    if rows is None:
        return {"uuid": None, "passed": False, "available": False, "required_bytes": required,
                "candidates": uuids, "observations": [], "source": "nvidia-smi read-only"}
    by_uuid = {row["uuid"]: row for row in rows}
    observations, chosen = [], None
    for uuid in uuids:
        row = by_uuid.get(uuid)
        if row is None:
            observations.append({"uuid": uuid, "available": False, "passed": False})
            continue
        observation = {**row, "available": True, "required_bytes": required,
                       "passed": bool(row["free_bytes"] >= required),
                       "source": "nvidia-smi read-only"}
        observations.append(observation)
        if chosen is None and observation["passed"]:
            chosen = observation
    return {"uuid": None if chosen is None else chosen["uuid"],
            "passed": chosen is not None, "available": True, "required_bytes": required,
            "candidates": uuids, "observations": observations,
            "source": "nvidia-smi read-only"}


def frozen_uuids(body):
    """The device list a frozen protocol declares, falling back to the legacy single."""
    declared = list(body.get("gpu_uuids") or [])
    if declared:
        return declared
    single = body.get("gpu_uuid")
    return [single] if single else []


def assignment_is_valid(body, primary_uuid):
    """True when the frozen device block is self-consistent with the recorded card.

    A protocol without ``gpu_uuids`` keeps the historical rule byte for byte: the single
    recorded UUID must be the round's declared one.
    """
    declared = list(body.get("gpu_uuids") or [])
    if not declared:
        return body.get("gpu_uuid") == primary_uuid
    return body.get("gpu_uuid") == primary_uuid and declared[0] == primary_uuid \
        and primary_uuid in declared


def assigned_uuid(body):
    """The card the parent pinned this worker to, validated against the freeze.

    The orchestrator exports ``CUDA_VISIBLE_DEVICES``. A worker must not silently fall
    back to another card: the freeze is what the receipt is compared to, so an
    unexpected assignment is a hard error, not a fallback.
    """
    declared = frozen_uuids(body)
    if not declared:
        raise RuntimeError("the frozen protocol declares no device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    assigned = visible or declared[0]
    if assigned not in declared:
        raise RuntimeError(f"assigned GPU {assigned!r} is not in the frozen device list {declared}")
    return assigned
