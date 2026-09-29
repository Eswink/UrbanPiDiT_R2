"""Append this round's evidence records to docs/R7_EVIDENCE_INDEX.jsonl.

Two records, because the round produced two independently checkable artifacts: the
0 GPU-h mechanism probe and the bounded leave-one-out contrast. The hashes of the
evidence documents are recomputed here rather than pasted, so the index cannot be
written with a stale digest - the same defect that failed the previous round's first
CI run. Nothing is recomputed about the measurements themselves.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "R7_EVIDENCE_INDEX.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def record(**fields) -> dict:
    body = {"format": "urbanpidit-r7-evidence-record-v1", **fields}
    body["evidence_sha256"] = sha256_file(ROOT / body["evidence_path"])
    return body


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
                          text=True).stdout.strip()
    if len(head) != 40:
        raise SystemExit(f"cannot resolve HEAD: {head!r}")

    data_identity = json.loads(
        (ROOT / "outputs" / "r7_72_rw_b_subtraction" / "merged_result.json")
        .read_text(encoding="utf-8"))["protocol"]["data"]["data_identity"]

    probe = record(
        record_id="rw-b-subtraction-probe",
        evidence_path="docs/R7_72_RW_B_SUBTRACTION.md",
        evidence_commit=head,
        experiment_commit=head,
        protocol_sha256=None,
        data_identity=None,
        source_sha256=None,
        outcome_class="audit",
        candidate_state="needs-review",
        priority=76,
        scientific_claim=False,
        reason=(
            "The 0 GPU-h leave-one-observation probe over the archived RW-B checkpoints: "
            "removing the gate+anchored proposal collapses the step-1 correction to 0.20-0.24x "
            "the RW-A magnitude in both seeds, while removing the cross-step recurrence of Z "
            "turns the error/correction cosine positive from step 1 in both seeds without "
            "reproducing the 1.8x magnitude. The same probe measures that the focus arm's "
            "correction_head never received a gradient, which is why the decision-0021 round "
            "retrains the leave-one-out arms from scratch rather than reusing this row."),
        limitations=[
            "Validation split only, 8 windows, one lead: a diagnostic, never a skill number.",
            "No threshold is introduced; the readings are the three the bounded round defined.",
            "Each row is an inference-time removal from a checkpoint trained with the piece on, "
            "so it is not the training-time contrast the bounded round measured.",
            "The gate+proposal-removed row routes the checkpoint through a correction head this "
            "arm never trained; the probe reports that measurement instead of leaving it implicit.",
            "The archived rows were measured on CUDA and this replay on CPU, so agreement is "
            "float32 reduction order (max |delta| 7e-06 on one seed and one flipped boolean on "
            "the other, reported).",
        ],
        metrics={
            "gpu_hours": 0.0, "cpu_seconds": 6.2, "checkpoints_hashed": 32,
            "checkpoints_replayed": 4, "seeds": 2, "validation_windows": 8,
            "step1_magnitude_ratio_gate_removed_mean": 0.22,
            "step1_magnitude_ratio_recurrence_removed_seed41": 1.48,
            "step1_magnitude_ratio_recurrence_removed_seed42": 3.04,
            "recurrence_removed_cosine_positive_both_seeds": True,
            "focus_arm_correction_head_moved_by_training": False,
            "test_read": False, "thresholds_added": 0,
        },
        ci_run_id=None,
        excluded_reason=(
            "Read-only diagnostic over archived checkpoints: it trains nothing and produces no "
            "forecast skill, so it cannot be indexed as a candidate model result."))

    bounded = record(
        record_id="rw-b-subtraction-round-cannot-attribute",
        evidence_path="docs/R7_72_RW_B_SUBTRACTION.md",
        evidence_commit=head,
        experiment_commit=head,
        protocol_sha256="58fc74b7a7aaa513197d85f836684cd55851013b3c7f8519f184649837b357d4",
        data_identity=data_identity,
        source_sha256=None,
        outcome_class="unresolved",
        candidate_state="needs-review",
        priority=79,
        scientific_claim=False,
        reason=(
            "The pre-declared subtraction round: the registered negative control RW-B-(a) is "
            "degenerate by construction - with identical weights it is bitwise identical to "
            "RW-A in the forward pass, the training loss and all 131 shared gradients, "
            "receiving zero gradient on the solver side - so its same-sign 'worsening' of "
            "3e-05..8e-05 K is float noise on an identity comparison, not a confound. The "
            "frozen branch rule fired as written (stop-confounded-control) and no attribution "
            "is reported. The primary arm's own reading (48 h +0.785 K, 72 h +1.174 K, both "
            "seeds, worsened) would have pointed at the gate+anchored proposal had the control "
            "been admissible, and the previous round's registered deltas reproduced to ~2e-04 K "
            "under a changed model-code digest."),
        limitations=[
            "The round's negative control is invalid by construction: this is a defect of the "
            "control arm's design, recorded rather than worked around.",
            "Two seeds: sign agreement is a consistency check, not significance; no threshold "
            "was introduced or relaxed.",
            "The arms are not capacity-matched, so no pair here is a compute-controlled contrast.",
            "The leave-one-out arms remove a piece at training time; the reference arm RW-A was "
            "retrained under this round's digests, so the two rounds' numbers are never pooled.",
            "The correction probe covers 8 validation windows per seed: proportions over those "
            "windows, not a distribution estimate.",
            "Validation split only; the test split stayed sealed and was never opened.",
            "One winter segment of one year in one region: no seasonal or cross-region claim.",
        ],
        metrics={
            "gpu_hours": 0.4128, "gpu_hours_training": 0.376, "gpu_hours_evaluation": 0.0368,
            "seeds": 2, "arms": 4, "updates_per_arm": 400, "test_read": False,
            "thresholds_added": 0, "branch": "stop-confounded-control",
            "negative_control_shared_weights_relative_diff": 1.6e-4,
            "rw_b_shared_weights_relative_diff": 1.74,
            "primary_no_recurrence_t2m_48h": 0.7854, "primary_no_recurrence_t2m_72h": 1.1739,
            "round_reference_t2m_48h": 1.0651, "round_reference_t2m_72h": 1.5805,
            "previous_round_reproduced_max_abs_delta": 1.8e-4,
            "parameters_rw_b": 3283157, "forward_flops_ratio_rw_b_vs_rw_a": 1.2097,
        },
        ci_run_id=None,
        excluded_reason=(
            "Cannot attribute at this budget: the control arm cannot distinguish anything, so "
            "no mechanism conclusion may be drawn from this round. The next admissible step is "
            "the pre-declared falsifiable hypothesis in the evidence document, or the turn to "
            "forecast state / training objective / data regime that stop condition 3 requires."))

    with INDEX.open("a", encoding="utf-8") as handle:
        for entry in (probe, bounded):
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"appended 2 records at HEAD {head[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
