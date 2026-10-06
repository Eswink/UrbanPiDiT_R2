"""Shared frozen support for the S3 v3 (five-train-year) screening rounds.

The v3 confirmation instance (train 2017-2021 / val 2022 / test 2023) is
scored with the same recipe as the v2 rounds: the archived actual-C
initialization reproduced bitwise on CPU, a frozen-endpoint fine-tune, per-lead
``evaluate_local`` cohorts, and a read-only co-residency gate. This module owns
the parts that must be identical between the v3 incumbent-control round and the
v3 budget-dose screen -- the fidelity check, the fine_tune contract assembly,
the paired-reading table parser, the GPU gate -- so the dose screen cannot
drift from the control it pins. Each round still freezes its own protocol
constants and writes its own protocol.json before any training step.

Instance identity constants here are facts about the published v3 store
(decision 0038/0039): the combined source SHA256, the derived data identity,
the mechanical train-window summary and the val per-lead cohorts. Loading this
module performs no model, data or network work.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCHIVED = ROOT / "outputs/r7_v2_comparison_20261004_attempt01"
ARCHIVED_PAIRING = {
    41: {"state": "a79ea47fa098bfc4dce651e2ea6c8fd18d996e4ff88a0e8503832645a29f1c47",
         "report": "b33c756723575681e0f7c57eb38ee8b2ba4715af708d00381c0914d803deb217"},
    42: {"state": "ec5bb5ef581e1cfe4923b938f60635fd5c79aaebd0f3f1fb37f01524ea3a26d0",
         "report": "8070b6d437d06dc94d21465fe3bed870b19ca7c773eaa882fbcdf976259e85f4"},
    43: {"state": "844bd23402cabb3f0cfb961efa4ba2e854207a06c51b51d32fba89e6e358efdf",
         "report": "8e2de147b25671b810a94d4c400f0ef567af189d1e634fb87734656556762c25"},
}
V3_SOURCE_SHA256 = "bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8"
V3_DATA_IDENTITY = "2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac"
V3_TRAIN_WINDOWS = {
    "input_windows": 2360, "usable_windows": 2340,
    "excluded_sample_ids": [
        "era5z_train_2017013012_p006h", "era5z_train_2017043012_p006h",
        "era5z_train_2017073012_p006h", "era5z_train_2017103012_p006h",
        "era5z_train_2018013012_p006h", "era5z_train_2018043012_p006h",
        "era5z_train_2018073012_p006h", "era5z_train_2018103012_p006h",
        "era5z_train_2019013012_p006h", "era5z_train_2019043012_p006h",
        "era5z_train_2019073012_p006h", "era5z_train_2019103012_p006h",
        "era5z_train_2020013012_p006h", "era5z_train_2020043012_p006h",
        "era5z_train_2020073012_p006h", "era5z_train_2020103012_p006h",
        "era5z_train_2021013012_p006h", "era5z_train_2021043012_p006h",
        "era5z_train_2021073012_p006h", "era5z_train_2021103012_p006h"],
    "window_sha256": "77a9945d2bf2952d589cfe8e522095bc3010c40e9680a303029b024b8736065c",
}
V3_CLIMATOLOGY_YEARS = [2017, 2018, 2019, 2020, 2021]
V3_VAL_COHORTS = {"6": 472, "12": 468, "24": 460, "48": 444, "72": 428}


def refused_test_manifest(manifest):
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this study is validation-only: the test split stays sealed")


def archived_process_spec():
    """The actual-C process spec, initialization and full configuration, verbatim."""
    protocol = json.loads((ARCHIVED / "protocol.json").read_text(encoding="utf-8"))
    configuration = protocol["configuration"]
    return (configuration["model_specs"]["process"]["model"],
            configuration["initialization"], configuration)


def cpu_fidelity(seed, configuration):
    """Reproduce the archived actual-C initialization for one seed, bitwise."""
    from training.r7_v2_profile import seeded_mapped_model, state_hash
    from training.r7_v2_protocol import digest
    model, report, spec = seeded_mapped_model(configuration, seed, "process")
    expected = ARCHIVED_PAIRING[seed]
    actual_state, actual_report = state_hash(model.state_dict()), digest(report)
    if actual_state != expected["state"] or actual_report != expected["report"]:
        raise RuntimeError(f"seed {seed} initialized state differs from archived actual C")
    if spec != configuration["model_specs"]["process"]["model"]:
        raise RuntimeError(f"seed {seed} spec differs from the archived process spec")
    return model, report, spec, actual_state, actual_report


def train_windows(train_manifest):
    """Metadata-only train-window discovery, asserted against the frozen v3 facts."""
    from data.r7_autoregressive_dataset import preflight_training_windows
    windows = preflight_training_windows(train_manifest)
    observed = {key: windows[key] for key in V3_TRAIN_WINDOWS}
    if observed != V3_TRAIN_WINDOWS:
        raise RuntimeError("v3 train windows differ from the frozen expectation")
    return windows


def contract_for(seed, spec, initialization, protocol_sha256, data_identity, windows, *,
                 source_sha256, mode, lambda12, arm):
    """Assemble the fine_tune contract the archived V2 worker would have passed."""
    from training.r7_v2_protocol import child_contract
    protocol = {"arm_configs": {arm: {"kind": "process", "mode": mode}},
                "data": {"data_identity": data_identity},
                "sources": {"source_sha256": source_sha256},
                "protocol_sha256": protocol_sha256, "windows": windows,
                "parents": {},
                "cpu_profile": {"pairing": {str(seed): {arm: {"seed": seed}}}},
                "shared_controls": {"lambda12": lambda12},
                "limitations": []}
    contract = child_contract(protocol, {"arm": arm, "seed": seed}, spec)
    contract["initialization"] = initialization
    contract["parent_provenance"] = None
    # The new child must not inherit parent process supervision (the archived
    # actual-C contract records process_supervision: None).
    contract.pop("process_supervision", None)
    return contract


def flop_probe(train_manifest, *, candidate_updates, control_updates, probe_seed):
    """Measure the objective's forward+backward cost on one frozen train sample."""
    import torch
    from torch.utils.data import default_collate
    from torch.utils.flop_counter import FlopCounterMode
    from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset
    from training.r7_autoregressive_rollout import training_one_step
    from training.r7_v2_profile import seeded_mapped_model
    torch.set_num_threads(4)
    windows = train_windows(train_manifest)
    dataset = ZarrAutoregressiveDataset(train_manifest,
                                        expected_exclusions=windows["excluded_sample_ids"])
    probe = default_collate([dataset[0]])
    _, initialization, configuration = archived_process_spec()
    model, _, _ = seeded_mapped_model(configuration, probe_seed, "process")
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        training_one_step(model, probe, reasoning_steps=4)
        forward = int(counter.get_total_flops())
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        output = training_one_step(model, probe, reasoning_steps=4)
        output.loss.backward()
        forward_backward = int(counter.get_total_flops())
    model.zero_grad(set_to_none=True)
    return {"probe_sample_id": probe["sample_id"][0],
            "l6_forward_flops": forward, "l6_forward_backward_flops": forward_backward,
            "candidate_updates": candidate_updates, "control_updates": control_updates,
            "candidate_training_flops": candidate_updates * forward_backward,
            "control_training_flops": control_updates * forward_backward,
            "updates_ratio": candidate_updates / control_updates,
            "flop_convention": ("supported aten operations under enable_grad; "
                                "elementwise/normalization omitted")}


def read_rmse_table(path):
    """{variable: rmse} for one single-lead rmse.csv (full-region pooled row set)."""
    import csv
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 17 or len({row["variable"] for row in rows}) != 17:
        raise RuntimeError(f"rmse table must carry all 17 variables: {path}")
    return {row["variable"]: float(row["rmse"]) for row in rows}


def paired_cells(candidate_dir, control_dir, evaluation_leads):
    """Per (lead, variable) RMSE and relative-MSE-change cells for one seed."""
    import math
    cells = {}
    for lead in evaluation_leads:
        candidate = read_rmse_table(Path(candidate_dir) / f"lead_{lead:03d}h" / "rmse.csv")
        control = read_rmse_table(Path(control_dir) / f"lead_{lead:03d}h" / "rmse.csv")
        if set(candidate) != set(control):
            raise RuntimeError(f"candidate/control variables differ at lead {lead}")
        for variable in sorted(candidate):
            delta = candidate[variable] - control[variable]
            relative = (candidate[variable] ** 2 - control[variable] ** 2) / (control[variable] ** 2)
            if not math.isfinite(delta) or not math.isfinite(relative):
                raise RuntimeError(f"nonfinite paired cell at lead {lead} variable {variable}")
            cells[f"{lead}|{variable}"] = {"lead_hours": lead, "variable": variable,
                                           "candidate_rmse": candidate[variable],
                                           "control_rmse": control[variable],
                                           "rmse_delta": delta, "relative_mse_change": relative}
    return cells


def loss_windows(report, size=100):
    """Per-segment training-loss means: the direct reading of 'still descending?'."""
    values = [row["loss"] for row in report["losses"]]
    windows = {}
    for start in range(0, len(values), size):
        segment = values[start:start + size]
        windows[f"{start + 1}-{start + len(segment)}"] = round(sum(segment) / len(segment), 6)
    return windows


def gpu_gate(uuid, *, owned_reserved_peak_bytes=0, estimated_peak_bytes, margin_bytes):
    """Read-only co-residency gate on the declared UUID; never signals a neighbor."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=60)
    rows = [line.split(", ") for line in query.stdout.strip().splitlines()]
    match = next((row for row in rows if row[0].strip() == uuid), None)
    if match is None:
        raise RuntimeError(f"declared GPU UUID {uuid} not visible to nvidia-smi")
    free_mib, total_mib = int(match[1]), int(match[2])
    required_mib = max(estimated_peak_bytes, owned_reserved_peak_bytes) // 2**20 \
        + margin_bytes // 2**20
    gate = {"uuid": uuid, "free_bytes": free_mib * 2**20, "total_bytes": total_mib * 2**20,
            "required_bytes": required_mib * 2**20,
            "owned_reserved_peak_bytes": owned_reserved_peak_bytes,
            "passed": bool(free_mib >= required_mib), "source": "nvidia-smi read-only"}
    if not gate["passed"]:
        raise RuntimeError(f"co-residency headroom gate failed: {gate}")
    return gate


def pin_declared_gpu(uuid):
    """Make the declared UUID the only visible CUDA device before any CUDA init."""
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", uuid):
        raise RuntimeError("CUDA_VISIBLE_DEVICES already set to a different device")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid


def code_state():
    """Working-tree identity at freeze time, recorded before any training step."""
    def run(*arguments):
        return subprocess.run(["git", *arguments], cwd=ROOT, check=True,
                              capture_output=True, text=True, timeout=60).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("status", "--porcelain")}


def seed_protocol_stamp(output_dir):
    """The frozen root protocol digest that nested child contracts must match."""
    frozen = json.loads((Path(output_dir) / "protocol.json").read_text(encoding="utf-8"))
    return frozen["protocol_sha256"]
