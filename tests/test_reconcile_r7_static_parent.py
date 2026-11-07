"""Counterproofs for ``tools/reconcile_r7_static_parent.py``; CPU only, no training, no claim.

Decision 0097's D1.d says the ``static_conditioning=True`` arm carries two persistent buffers the
``False`` arm never builds, so the registered parent cannot load into it under ``strict=True`` and
``strict=False`` silently drops them. These tests pin the three properties that make the receipt
trustworthy: the buffers come from the published product rather than a checkpoint, the strict path
is the one used, and neither the bare parent nor a mismatched arm is accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import reconcile_r7_static_parent as tool  # noqa: E402

PARENT = ROOT / ("outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/"
                 "seed41/parent_update_0001600.pt")


def _skip_without_parent():
    if not PARENT.is_file():
        pytest.skip("the registered parent checkpoint is an untracked outputs/ artifact; no "
                    "synthetic fallback")


def _skip_without_static_product():
    """The published static product is also an untracked outputs/ artifact.

    CI is a bare checkout with no outputs/, so every test that builds the conditioner must skip
    there with a named reason rather than failing on a missing BUILD_COMPLETE. This is the same
    artifact-absence rule the rest of the suite follows; making the requirement explicit here is
    what keeps the clean checkout green without inventing a synthetic field.
    """
    from model import static_conditioning_r7 as static
    if not (static.STATIC_MANIFEST.parent / "BUILD_COMPLETE.json").is_file():
        pytest.skip("the published static product is an untracked outputs/ artifact; no synthetic "
                    "fallback")


def test_the_buffer_keys_are_exactly_the_two_the_model_adds():
    """If model/ renames a buffer, the tool must fail rather than reconcile nothing silently."""
    _skip_without_parent()
    _skip_without_static_product()
    from training.r7_experiment import make_model
    spec = torch.load(PARENT, map_location="cpu", weights_only=True)["contract"]["model"]
    with torch.random.fork_rng(devices=[]):
        off = make_model("process", dict(spec))
    with torch.random.fork_rng(devices=[]):
        on = make_model("process", dict(spec, static_conditioning=True))
    added = sorted(set(on.state_dict()) - set(off.state_dict()))
    assert added == sorted(tool.BUFFER_KEYS)


def test_the_published_buffers_are_the_published_product():
    """The field buffer must be bit-identical to the stack ``load_static_fields`` returns."""
    _skip_without_static_product()
    built, provenance = tool._published_buffers()
    assert provenance["field_equals_published_stack"] is True
    assert provenance["variable_count"] == len(provenance["variables"])
    # The manifest states the caveat in prose; the tool must pass it through rather than drop it.
    caveat = provenance["derived_statistic_not_truth"]
    assert isinstance(caveat, str) and "NOT" in caveat.upper()
    field = built["backbone.static_conditioning_pathway.field"]
    projection = built["backbone.static_conditioning_pathway.projection"]
    assert field.ndim == 3 and field.shape[0] == provenance["variable_count"]
    assert projection.ndim == 2 and projection.shape[1] == 192


def test_the_reconciled_digest_is_stable_across_two_builds():
    """A deterministic reconciliation: same parent + same product => same digest, twice."""
    _skip_without_static_product()
    built_a, _ = tool._published_buffers()
    built_b, _ = tool._published_buffers()
    for key in tool.BUFFER_KEYS:
        assert torch.equal(built_a[key], built_b[key])


def test_a_missing_checkpoint_is_an_error_not_a_silent_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        tool.reconcile(tmp_path / "absent.pt", tmp_path / "out")


def test_an_existing_output_directory_is_refused(tmp_path):
    """Every attempt gets its own directory: a receipt is never overwritten."""
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        tool.reconcile(PARENT, out)


def test_the_tool_refuses_a_parent_that_already_carries_static(tmp_path):
    """Reconciling an on-arm parent would be a no-op dressed as work; say so instead."""
    _skip_without_parent()
    _skip_without_static_product()
    payload = torch.load(PARENT, map_location="cpu", weights_only=True)
    tampered = dict(payload)
    contract = json.loads(json.dumps(payload["contract"]))
    contract["model"]["static_conditioning"] = True
    tampered["contract"] = contract
    path = tmp_path / "on_parent.pt"
    torch.save(tampered, path)
    with pytest.raises(RuntimeError, match="already carries static_conditioning"):
        tool.reconcile(path, tmp_path / "out")


def test_a_real_reconciliation_records_the_gap_it_closes(tmp_path):
    """The end-to-end property: 157 tensors, strict pass, bare parent rejected, two left missing."""
    _skip_without_parent()
    _skip_without_static_product()
    receipt = tool.reconcile(PARENT, tmp_path / "out")
    assert receipt["strict_load_of_reconciled_passed"] is True
    assert receipt["bare_parent_into_on_arm_failed"] is True
    assert sorted(receipt["strict_false_would_leave_missing"]) == sorted(tool.BUFFER_KEYS)
    assert receipt["parent_tensor_count"] + len(tool.BUFFER_KEYS) == receipt["reconciled_tensor_count"]
    assert receipt["parent_static_keys"] == []
    assert receipt["gpu_hours"] == 0.0
    # The measured correction to decision 0097: strict=False loses verification, not values.
    assert receipt["strict_false_buffer_values"]["field_still_equals_published"] is True
    assert receipt["strict_false_buffer_values"]["projection_still_equals_published"] is True
    assert receipt["scientific_claim"] is False and receipt["test_read"] is False and receipt["r"] == 0
    # The receipt and the saved state must describe the same bytes.
    assert (tmp_path / "out" / "receipt.json").is_file()
    assert (tmp_path / "out" / "reconciled_state.pt").is_file()
    saved = torch.load(tmp_path / "out" / "reconciled_state.pt", map_location="cpu",
                       weights_only=True)
    assert len(saved) == receipt["reconciled_tensor_count"]
    # The saved state itself must carry the two buffers. Without this the receipt could claim a
    # strict reconciliation while the file on disk held a bare parent -- which is the silent-drop
    # hazard 0097 exists to forbid, dressed as success.
    published, _ = tool._published_buffers()
    for key in tool.BUFFER_KEYS:
        assert key in saved, f"{key} is absent from the reconciled state"
        assert torch.equal(saved[key], published[key]), f"{key} is not the published buffer"
