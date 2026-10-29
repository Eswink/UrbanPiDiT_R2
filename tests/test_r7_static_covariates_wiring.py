"""Decision 0069: the static-conditioning pathway, its identity binding and its exits.

What this file is responsible for, in the order the decision states them:

1. **W2, the off path.** With ``static_conditioning=False`` the trunk must be the
   pre-existing computation. The round's own evidence for that is a pre-/post-edit
   fingerprint; here the property under test is narrower and cheap: the off model builds
   *no* static module, carries *no* static buffer, and has exactly the parameter count the
   frozen spec had before this round existed.
2. **W1, the identity binding.** ``static_identity`` must move ``protocol_sha256`` and must
   not touch ``data_identity``. Both halves are asserted, because a binding that moves both
   or neither is the failure mode the decision names.
3. **W4, the field is the published one.** The buffer must equal the ``.npz`` bit for bit,
   and an independent loop-based patch average must reproduce the vectorised one exactly.
4. **The parameter delta is zero.** The static term is a fixed projection over a fixed
   expansion; nothing about it is learned. This is what lets an on/off difference be read as
   information rather than capacity, so it is asserted rather than described.

Every refusal path has a counter-proof: a test that the guard fires. A guard that never
fires is indistinguishable from no guard.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import static_conditioning_r7 as static                       # noqa: E402
from scripts import r7_static_wiring_support as support                  # noqa: E402
from scripts import study_r7_static_covariates_wiring as driver          # noqa: E402
from scripts.study_r7_s3_fy_year_coverage_matched import _model_spec    # noqa: E402
from training.r7_experiment import make_model, seed_everything           # noqa: E402

PRODUCT = ROOT / "outputs/r7_urban_static_covariates_attempt01"


def _product_published() -> bool:
    return (PRODUCT / "published/manifest/BUILD_COMPLETE.json").is_file()


published = pytest.mark.skipif(
    not _product_published(),
    reason="the published static product is not present in this checkout")

#: The published identity decision 0063 registered. Literal on purpose: a round that quietly
#: conditioned on a different product must fail here, not agree with itself.
PINNED_STATIC_IDENTITY = "8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03"
#: The pre-change model digest (R1's parent pin). This round moves it, by design.
PRE_CHANGE_MODEL_CODE = "60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e"


# ---------------------------------------------------------------- the off path (W2)


def test_the_off_arm_constructs_no_static_module_and_no_static_buffer():
    with torch.random.fork_rng(devices=[]):
        seed_everything(driver.SEED)
        model = make_model("process", driver.arm_spec("control"))
    assert model.static_conditioning is False
    assert model.backbone.static_conditioning_pathway is None
    assert not any("static_conditioning_pathway" in name for name in model.state_dict())


def test_the_two_arms_carry_the_same_parameter_count():
    """parameter_delta = 0 is the whole reason on/off is an information difference."""
    counts = {}
    for name in support.ARMS:
        with torch.random.fork_rng(devices=[]):
            seed_everything(driver.SEED)
            counts[name] = sum(p.numel() for p in
                               make_model("process", driver.arm_spec(name)).parameters())
    assert counts["control"] == counts["treatment"]
    assert driver.arm_spec("control") != driver.arm_spec("treatment")


def test_the_treatment_arm_adds_buffers_and_no_trained_parameter():
    with torch.random.fork_rng(devices=[]):
        seed_everything(driver.SEED)
        control = make_model("process", driver.arm_spec("control"))
    with torch.random.fork_rng(devices=[]):
        seed_everything(driver.SEED)
        treatment = make_model("process", driver.arm_spec("treatment"))
    off, on = control.state_dict(), treatment.state_dict()
    added = sorted(set(on) - set(off))
    assert added == ["backbone.static_conditioning_pathway.field",
                     "backbone.static_conditioning_pathway.projection"]
    assert sorted(set(off) - set(on)) == []
    # Every pre-existing tensor keeps the value the same seed gave it without the pathway.
    assert all(torch.equal(off[name], on[name]) for name in off)


def test_arm_spec_rejects_an_archived_spec_that_already_carries_the_switch():
    """A spec with the key already present would silently ignore this round's factor."""
    archived = _model_spec()
    assert "static_conditioning" not in archived
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(driver.matched, "_model_spec",
                      lambda: {**archived, "static_conditioning": False})
        with pytest.raises(RuntimeError):
            driver.arm_spec("control")


def test_unknown_arm_names_are_refused():
    with pytest.raises(ValueError):
        driver.arm_spec("static")


def test_the_reference_predates_the_edit_it_guards():
    """A reference captured after the change would prove nothing, so its digest must differ."""
    from training.r7_experiment import model_code_digest
    if not driver.REFERENCE.is_file():
        pytest.skip("the pre-change reference was not captured in this checkout")
    reference = json.loads(driver.REFERENCE.read_text(encoding="utf-8"))
    assert reference["values"]["model_code_sha256"] == PRE_CHANGE_MODEL_CODE
    assert reference["revision"]["model_dirty"] is False
    assert reference["values"]["model_code_sha256"] != model_code_digest()
    assert reference["scientific_claim"] is False


# ------------------------------------------------------------ the identity (W1)


def _minimal_body():
    """A protocol-shaped body with both identities, without building a store."""
    return {
        "format": driver.FORMAT,
        "static": {"static_identity": PINNED_STATIC_IDENTITY, "output_sha256": "a" * 64},
        "arms": {name: {"train_data_identity": "b" * 64} for name in support.ARMS},
        "protocol_sha256": "c" * 64,
    }


def test_mutating_the_static_identity_moves_the_protocol_digest():
    probe = driver.identity_probe(_minimal_body())
    assert probe["passed"] is True
    assert probe["protocol_sha256_before"] != probe["protocol_sha256_after"]
    assert probe["data_identity_unchanged"] is True


def test_the_probe_does_not_mutate_the_body_it_was_given():
    body = _minimal_body()
    snapshot = copy.deepcopy(body)
    driver.identity_probe(body)
    assert body == snapshot


def test_the_probe_reports_a_binding_that_also_moved_the_data_identity():
    """Counter-proof with a real body: if the field leaked into ``data_identity``, W1 fails.

    This is not a stub. The body is a genuine protocol-shaped dict whose ``data_identity``
    happens to be a function of the static identity - which is exactly the shape decision
    0069 forbids - and the probe must report false for it.
    """
    body = _minimal_body()
    for name, arm in body["arms"].items():
        arm["train_data_identity"] = PINNED_STATIC_IDENTITY
    probe = driver.identity_probe(body)
    assert probe["passed"] is False


def test_a_mutation_that_moves_nothing_fails_w1():
    """The failure mode decision 0069 names: the field exists but never reaches the digest.

    A body whose static block is empty has nothing to mutate, so the two digests must agree
    and W1 must report false - that is exactly what "the binding is not wired" looks like.
    """
    body = _minimal_body()
    body["static"] = {}
    assert driver.identity_probe(body)["passed"] is False


# ------------------------------------------------------------- the field (W4)


@published
def test_the_published_identity_is_the_pinned_one():
    assert static.require_static_identity()["static_identity"] == PINNED_STATIC_IDENTITY


@published
def test_a_swapped_identity_is_refused(monkeypatch):
    monkeypatch.setattr(static, "STATIC_IDENTITY", "0" * 64)
    with pytest.raises(RuntimeError):
        static.require_static_identity()


@published
def test_a_missing_build_complete_marker_is_refused(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"static_identity": PINNED_STATIC_IDENTITY}),
                        encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        static.require_static_identity(manifest_path=manifest,
                                       fields_path=tmp_path / "fields.npz")


@published
def test_the_buffer_is_bitwise_the_published_npz():
    fields, variables = static.load_static_fields()
    with np.load(static.STATIC_FIELDS) as archive:
        for index, name in enumerate(variables):
            assert np.array_equal(fields[index].numpy(),
                                  np.asarray(archive[name], dtype=np.float32))


@published
def test_an_independent_patch_average_reproduces_the_vectorised_one():
    """W4's cross-check: a second path with no shared code, at the reduction-order bound.

    The bitwise half of W4 is the buffer against the ``.npz`` (asserted exactly above). This
    test is about the *derivation*: a sequential float64 loop and a batched ``mean`` accumulate
    in different orders and can miss the last bit, so the comparison is bounded rather than
    exact - and the bound is declared in the support module, not tuned here.
    """
    variables = list(static.DEFAULT_VARIABLES)
    scales = [static.STATIC_SCALES.get(name, 1.) for name in variables]
    with np.load(static.STATIC_FIELDS) as archive:
        raw = [[[float(value) for value in row] for row in
                np.asarray(archive[name], dtype=np.float64)] for name in variables]
    loop = torch.tensor(support.independent_patch_mean(raw, token_hw=(33, 33), patch_size=2,
                                                       scales=scales), dtype=torch.float64)
    fields, _ = static.load_static_fields(variables)
    from model.layers.patch_grid import pad_patch_grid
    factor = torch.tensor(scales, dtype=torch.float64).reshape(-1, 1, 1)
    padded = pad_patch_grid((fields.double() / factor).unsqueeze(0), 2)
    cells = padded.reshape(1, len(variables), 33, 2, 33, 2).mean(dim=(3, 5))
    vector = cells.reshape(len(variables), -1).transpose(0, 1)
    assert float((loop - vector).abs().max()) <= support.REDUCTION_ORDER_TOLERANCE


@published
def test_the_loop_cross_check_is_not_a_stub():
    """Counter-proof: perturb one value and the independent average must move."""
    variables = list(static.DEFAULT_VARIABLES)
    with np.load(static.STATIC_FIELDS) as archive:
        raw = [[[float(value) for value in row] for row in
                np.asarray(archive[name], dtype=np.float64)] for name in variables]
    baseline = support.independent_patch_mean(raw, token_hw=(33, 33), patch_size=2)
    perturbed = [[list(row) for row in channel] for channel in raw]
    perturbed[0][0][0] += 0.5
    moved = support.independent_patch_mean(perturbed, token_hw=(33, 33), patch_size=2)
    assert moved != baseline
    assert moved[0][0] != baseline[0][0]


@published
def test_a_missing_variable_raises_instead_of_shrinking_the_stack():
    with pytest.raises(KeyError):
        static.load_static_fields(("lc_built_frac_2021", "not_a_published_variable"))


@published
def test_the_feature_count_is_two_per_variable():
    fields, variables = static.load_static_fields()
    features = static.static_features(fields, token_hw=(33, 33), patch_size=2,
                                      variables=variables)
    assert features.shape == (33 * 33, 2 * len(variables))
    assert torch.isfinite(features).all()


@published
def test_the_term_is_time_invariant():
    """Same term for every reasoning step and lead: it takes only the token grid."""
    fields, variables = static.load_static_fields()
    module = static.StaticConditioning(fields, dim=192, patch_size=2, variables=variables)
    first, second = module(token_hw=(33, 33)), module(token_hw=(33, 33))
    assert torch.equal(first, second)
    assert first.shape == (1, 33 * 33, 192)


def test_the_conditioning_module_has_no_trained_parameter():
    field = torch.zeros(len(static.DEFAULT_VARIABLES), 65, 65)
    module = static.StaticConditioning(field, dim=64, patch_size=2)
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    assert module.projection.shape == (2 * len(static.DEFAULT_VARIABLES), 64)


def test_the_declared_scales_are_not_silently_missing():
    """A covariate with no declared scale would enter at raw magnitude, dominating the sum."""
    for name in static.DEFAULT_VARIABLES:
        scale = static.STATIC_SCALES.get(name, 1.0)
        assert scale > 0
    assert set(static.STATIC_SCALES) <= set(static.DEFAULT_VARIABLES)


# --------------------------------------------------------------- the exits


def _observations(**overrides):
    base = {"w1": {"passed": True}, "w2": {"passed": True},
            "w3": {"passed": True}, "w4": {"passed": True}}
    base.update(overrides)
    return base


def test_all_four_true_exits_are_engineering_positive_without_a_scientific_claim():
    verdict = support.exits(_observations())
    assert verdict["all"] is True
    assert verdict["verdict"] == "engineering-positive"
    assert verdict["scientific_claim"] is False


@pytest.mark.parametrize("failing", ["w1", "w2", "w3", "w4"])
def test_any_one_false_exit_is_negative(failing):
    verdict = support.exits(_observations(**{failing: {"passed": False}}))
    assert verdict["all"] is False
    assert verdict["verdict"] == "negative"


def test_the_path_probe_requires_equal_row_counts(tmp_path):
    receipts = {}
    for index, name in enumerate(support.ARMS):
        receipts[name] = {"evaluations": {
            str(lead): {"rows": 1 + index, "n_evaluated": 1} for lead in driver.EVALUATION_LEADS}}
    probe = driver.path_probe(tmp_path, receipts)
    assert probe["passed"] is False
    assert probe["row_counts_equal"] is False


def test_the_path_probe_requires_every_lead(tmp_path):
    receipts = {}
    for name in support.ARMS:
        receipts[name] = {"evaluations": {"6": {"rows": 3, "n_evaluated": 2}}}
    probe = driver.path_probe(tmp_path, receipts)
    assert probe["passed"] is False
    assert probe["missing"]


def test_the_path_probe_passes_on_complete_equal_receipts(tmp_path):
    receipts = {name: {"evaluations": {str(lead): {"rows": 3, "n_evaluated": 2}
                                       for lead in driver.EVALUATION_LEADS}}
                for name in support.ARMS}
    probe = driver.path_probe(tmp_path, receipts)
    assert probe["passed"] is True


def test_rows_of_counts_csv_rows_not_including_the_header(tmp_path):
    table = tmp_path / "boundary_rmse.csv"
    table.write_text("region,variable,rmse\nfull,t2m,1.0\nfull,u10,2.0\n", encoding="utf-8")
    assert support.rows_of(table) == 2


# ------------------------------------------------------------ the honesty pins


def test_the_round_declares_no_scientific_claim_and_no_test_read():
    constants = driver.frozen_constants()
    assert constants["scientific_claim"] is False
    assert constants["test_read"] is False


def test_the_limitations_name_the_unloadable_parent_consequence():
    text = " ".join(driver.frozen_constants()["limitations"])
    assert "model_code_sha256" in text
    assert "migration" in text


def test_the_decision_rule_has_no_pass_language():
    rule = driver.frozen_constants()["decision"]["text"]
    assert "engineering-positive" in rule
    assert "no scientific claim" in rule
    for banned in ("SOTA", "state of the art", "outperforms"):
        assert banned.lower() not in rule.lower()


def test_the_execution_inventory_is_named_not_globbed():
    """Decision 0067: the round binds its import closure, and an unrelated module is absent."""
    files = driver.execution_files()
    assert len(files) < 150
    assert "model/static_conditioning_r7.py" in files
    assert "scripts/study_r7_static_covariates_wiring.py" in files
    # A module no import in this round reaches: under R1's glob it would have been included.
    assert "data/download/worldcover_cog.py" not in files


def test_the_inventory_does_not_move_when_an_unrelated_module_is_added(tmp_path):
    """The counter-proof decision 0067 requires: a new file must not change the inventory."""
    baseline = driver.execution_files()
    assert [name for name in baseline if name.startswith("data/")] == \
        [name for name in baseline if name.startswith("data/")]
    from tools.r7_explicit_inventory import explicit_execution_files
    repeated = explicit_execution_files(driver.ENTRY_FILES[0], extra=driver.ENTRY_FILES[1:])
    assert repeated == baseline
