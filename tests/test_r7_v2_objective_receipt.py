"""Native FP32 objective reconstruction counterexamples, no real outputs/GPU."""
from copy import deepcopy
import math
import struct

import pytest

from training.r7_v2_objective_receipt import float32, verify_training_objective
from training.r7_v2_tables import close


def inputs(*, bf16=False, mode="two_step"):
    return {"bf16": bf16, "lambda12": .5}, {
        "bf16": bf16, "mode": mode,
        "autoregression": {"objective": "deep_supervised_latitude_area_mse", "loss_space": "normalized",
                           "internal_deep_supervision": True}}


@pytest.mark.parametrize("l6,l12", [(.15507206320762634, .22569331526756287),
                                  (.08634218573570251, .17500606179237366),
                                  (float32(1e-20), float32(1e-20)),
                                  (float32(1e20), float32(3e20))])
@pytest.mark.parametrize("bf16", [False, True])
def test_native_components_fp32_sequential_rounding_is_accepted_without_metric_tolerance_change(l6, l12, bf16):
    controls, contract = inputs(bf16=bf16)
    expected = float32(l6 + float32(.5 * l12))
    row = {"l6": l6, "l12": l12, "loss": expected}
    verified = verify_training_objective(row, mode="two_step", controls=controls, contract=contract)
    assert verified["bits_match"] is True
    assert verified["expected_loss"] == expected
    assert verified["metric_tolerance_changed"] is False
    if expected != l6 + .5 * l12:
        with pytest.raises(ValueError): close(expected, l6 + .5 * l12, "original binary64 objective")


@pytest.mark.parametrize("change", ["loss-bit", "l6-bit", "negative", "non-native", "nan", "inf", "bool", "precision", "recipe", "lambda"])
def test_fp32_objective_receipt_rejects_tamper_unknown_precision_and_invalid_values(change):
    controls, contract = inputs()
    row = {"loss": .2679187059402466, "l6": .15507206320762634, "l12": .22569331526756287}
    if change in ("loss-bit", "l6-bit"):
        key = "loss" if change == "loss-bit" else "l6"
        bits = struct.unpack("!I", struct.pack("!f", row[key]))[0]
        row[key] = struct.unpack("!f", struct.pack("!I", bits + 2))[0]
    elif change == "negative": row["loss"] = -row["loss"]
    elif change == "non-native": row["loss"] += 1e-14
    elif change == "nan": row["l12"] = float("nan")
    elif change == "inf": row["l12"] = float("inf")
    elif change == "bool": row["l6"] = True
    elif change == "precision": contract["bf16"] = None
    elif change == "recipe": contract["autoregression"]["loss_space"] = "physical"
    elif change == "lambda": controls["lambda12"] = True
    with pytest.raises(ValueError): verify_training_objective(row, mode="two_step", controls=controls, contract=contract)


def test_l6_no_future_loss_is_strict_native_component_identity():
    controls, contract = inputs(mode="l6")
    row = {"loss": float32(.2), "l6": float32(.2), "l12": None}
    assert verify_training_objective(row, mode="l6", controls=controls, contract=contract)["bits_match"] is True
    row["loss"] = float32(.21)
    with pytest.raises(ValueError): verify_training_objective(row, mode="l6", controls=controls, contract=contract)
    row.update(loss=float32(.2), l12=0.)
    with pytest.raises(ValueError): verify_training_objective(row, mode="l6", controls=controls, contract=contract)


def test_reconstruction_matches_actual_torch_fp32_scalar_ops_and_keeps_metric_guard():
    import torch
    controls, contract = inputs()
    for l6, l12 in [(.15507206320762634, .22569331526756287), (1e-20, 1e-20), (1e20, 3e20)]:
        x, y = torch.tensor(l6, dtype=torch.float32), torch.tensor(l12, dtype=torch.float32)
        row = {"loss": float(x + .5 * y), "l6": float(x), "l12": float(y)}
        assert verify_training_objective(row, mode="two_step", controls=controls, contract=contract)["expected_loss"] == row["loss"]
    with pytest.raises(ValueError): close(1., 1. + 1e-6, "unchanged physical metric guard")


def test_corrected_future_training_validator_accepts_real_fp32_vectors_and_refuses_tamper(tmp_path):
    import json
    from test_r7_v2_results import full_protocol, training_receipt
    from training.r7_v2_protocol import sha256_file
    from training.r7_v2_results import _verify_training
    output, protocol = full_protocol(tmp_path)
    job = next(job for job in protocol["jobs"] if job["phase"] == "train" and job["arm"] == "rollout_l6_l12")
    entry = training_receipt(output, protocol, job)
    for row in entry["report"]["losses"]:
        row.update(loss=.2679187059402466, l6=.15507206320762634, l12=.22569331526756287)
    report_path = entry["training_report"]
    def bind_report():
        from pathlib import Path
        Path(report_path).write_text(json.dumps(entry["report"]), encoding="utf-8")
        entry["training_report_sha256"] = sha256_file(report_path)
    bind_report()
    assert _verify_training(entry, protocol, job, output)["updates_this_run"] == 200
    entry["report"]["losses"][0]["loss"] = float32(.27)
    bind_report()
    with pytest.raises(ValueError, match="FP32"): _verify_training(entry, protocol, job, output)
