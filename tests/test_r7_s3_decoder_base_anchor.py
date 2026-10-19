"""CI-safe checks for the S3 decoder-base-anchor driver and the ``decoder_base_mode`` switch.

The frozen factor is the decoder's base semantics: the registered persistence anchor
(``X_t = history[:, -1]``) versus a zero base. These tests run on synthetic tensors and
fixtures only: nothing reads ``outputs/`` or ``test.jsonl``. Every frozen claim has a
counterproof, including "the default persistence path is the pre-change body bit for
bit", "zero mode makes the head output its tendency", "the two modes share one tensor
set", "the lazy climatology anchor value reaches no output in both arms", and "the training path
accepts the new config on CPU".

The pre-change check rebuilds the old body from this module's own encoder/lead/spacetime/
head parts because CI may be a shallow checkout that cannot import ``git show 0becf24``.
Independent review replayed the archived old ``model`` package against the new default on
the same tiny state and batch and measured ``torch.equal=True``; that replay is evidence
outside this suite, not a substitute for running it here.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load("study_r7_s3_decoder_base_anchor")
parent = driver.parent
report = driver.report
interior = driver.interior
import model.process_step_r7 as process_step
from model.coarse_forecast import CoarseForecastHead
from training.r7_experiment import canonical_digest, make_model


def _archived_spec(dim=driver.DIM, depth=driver.DEPTH, **extra):
    return dict(architecture="window", dim=dim, depth=depth, in_channels=17, out_channels=17,
                history_steps=2, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, **extra)


class _Shared:
    ARCHIVED = ROOT / "synthetic_archive"

    def __init__(self, spec):
        self._spec = spec

    def archived_process_spec(self):
        return (self._spec, None, None)


def _tiny_anchor():
    """The real builder's synthetic anchor; no published artifact and no outputs/ read."""
    from scripts.r7_s3_anchor_isolation_parent import _tiny_anchor as build
    return build()


@pytest.fixture
def anchor_stub(monkeypatch):
    """Replace the archived spec and the published anchor with synthetic fixtures."""
    anchor_spec, _ = _tiny_anchor()
    spec = _archived_spec()
    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared(spec))
    monkeypatch.setattr(driver.anchor_arm, "anchor_payload",
                        lambda: {"spec": anchor_spec, "pins": {}})
    return anchor_spec


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _tiny_model(mode, *, anchor=None, table=None, **extra):
    spec = dict(parent._tiny_spec(), **extra)
    if mode is not None:
        spec["decoder_base_mode"] = mode
    torch.manual_seed(parent.EQUIVALENCE_SEED)
    model = make_model("process", spec)
    if anchor is not None:
        model.backbone.climatology_anchor.install(table)
    return model.eval()


# --------------------------------------------------------------------------- the model switch


def test_default_and_explicit_persistence_are_bitwise_identical():
    omitted = _tiny_model(None)
    explicit = _tiny_model("persistence")
    assert omitted.backbone.decoder_base_mode == "persistence"
    batch = parent._probe_batch()
    with torch.no_grad():
        a, b = omitted(batch), explicit(batch)
        ab, bb = omitted.backbone(batch), explicit.backbone(batch)
    assert torch.equal(a.initial_forecast, b.initial_forecast)
    assert torch.equal(a.forecast, b.forecast)
    assert torch.equal(a.draft_forecasts, b.draft_forecasts)
    assert torch.equal(ab.tendency, bb.tendency)
    assert torch.equal(ab.base_state, bb.base_state)
    for key in omitted.state_dict():
        assert torch.equal(omitted.state_dict()[key], explicit.state_dict()[key]), key


def test_the_persistence_path_is_the_pre_change_forward_bit_for_bit():
    """Recompute the pre-change body from the module's own parts and compare exactly.

    The model is built with the key omitted, which is the pinned control declaration.
    """
    model = _tiny_model(None)
    assert model.backbone.decoder_base_mode == "persistence"
    assert "decoder_base_mode" not in model.backbone.state_dict()
    batch = parent._probe_batch()
    history = batch["coarse_history"]
    B, T, C, H, W = history.shape
    with torch.no_grad():
        tokens, token_hw = model.backbone.encoder(history)
        lead = model.backbone.lead_time(batch.get("lead_time_hours"), batch=B,
                                        device=tokens.device, dtype=tokens.dtype,
                                        default_hours=model.backbone.default_lead_hours)
        context = tokens + lead[:, None, :]
        context = context + model.backbone.spacetime(batch, history=history,
                                                     token_hw=token_hw).to(tokens.dtype)
        base = history[:, -1, :model.out_channels]
        forecast, tendency = model.backbone.head(context, token_hw, (H, W), base)
        out = model.backbone(batch)
    assert torch.equal(out.forecast, forecast)
    assert torch.equal(out.tendency, tendency)
    assert torch.equal(out.context_tokens, context)
    assert torch.equal(out.base_state, base)
    # Counter-evidence: the zero arm must differ from this reconstruction on the same batch.
    zero = _tiny_model("zero")
    with torch.no_grad():
        assert not torch.equal(zero.backbone(batch).forecast, forecast)


def test_zero_mode_returns_the_tendency_as_the_forecast_and_a_zero_base():
    model = _tiny_model("zero")
    batch = parent._probe_batch()
    with torch.no_grad():
        out = model.backbone(batch)
        process = model(batch)
    assert torch.equal(out.forecast, out.tendency)
    assert torch.equal(process.initial_forecast, out.tendency)
    assert bool((out.base_state == 0).all())
    assert out.base_state.shape == batch["coarse_history"][:, -1].shape


def test_persistence_minus_zero_equals_the_base_on_exact_operands():
    """``(base + t) - t`` is not bitwise base in float32; pin the identity exactly.

    The constructed head emits a constant 0.5 tendency and the base holds integers, so
    the addition and subtraction are exact and the identity the round names holds
    bitwise. The realistic-tensor version of the same statement is the allclose check in
    the parent module's synthetic control.
    """
    head = CoarseForecastHead(dim=8, out_channels=2, patch_size=2)
    with torch.no_grad():
        head.decode[-1].weight.zero_()
        head.decode[-1].bias.fill_(0.5)
    tokens = torch.randn(1, 1, 8)
    base = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.0], [-2.0, 0.5]]]])
    forecast, tendency = head(tokens, (1, 1), (2, 2), base)
    zero_forecast, zero_tendency = head(tokens, (1, 1), (2, 2), torch.zeros_like(base))
    assert torch.equal(tendency, torch.full_like(base, 0.5))
    assert torch.equal(zero_forecast, zero_tendency)
    assert torch.equal(forecast - zero_forecast, base)


def test_the_two_modes_share_one_state_dict_key_shape_and_parameter_set():
    persistence = _tiny_model("persistence")
    zero = _tiny_model("zero")
    expected, actual = persistence.state_dict(), zero.state_dict()
    assert set(expected) == set(actual)
    for key in expected:
        assert expected[key].shape == actual[key].shape, key
        assert torch.equal(expected[key], actual[key]), key
    assert sum(parameter.numel() for parameter in persistence.parameters()) == \
        sum(parameter.numel() for parameter in zero.parameters())


def test_the_local_solver_proposal_anchor_is_zero_only_in_zero_mode(monkeypatch):
    anchor_spec, table = _tiny_anchor()
    anchored = dict(climatology_anchor_spec=anchor_spec, anchor_output_prior=False,
                    anomaly_feedback=False)
    models = {mode: _tiny_model(mode, anchor=anchor_spec, table=table, **anchored)
              for mode in ("persistence", "zero")}
    seen = {}
    original = process_step.anchored_proposal

    def spy(decoder, solver_state, token_hw, output_hw, anchor):
        seen.setdefault(current, []).append(anchor)
        return original(decoder, solver_state, token_hw, output_hw, anchor)

    monkeypatch.setattr(process_step, "anchored_proposal", spy)
    batch = parent._probe_batch()
    for current in ("persistence", "zero"):
        with torch.no_grad():
            models[current](batch, reasoning_steps=1)
    stored = batch["coarse_history"][:, -1]
    assert len(seen["persistence"]) == len(seen["zero"]) == 1
    assert torch.equal(seen["persistence"][0], stored)
    assert bool((seen["zero"][0] == 0).all())
    # The proposal difference is exactly the anchor on the same recurrent state only when
    # the decoder is linear in its anchor; the direct check above is the load-bearing one.
    assert torch.equal(seen["persistence"][0] - seen["zero"][0], stored)


def test_zeroing_the_lazy_anchor_table_changes_neither_arm():
    anchor_spec, table = _tiny_anchor()
    zero_spec = dict(anchor_spec, table_sha256=parent.table_digest(torch.zeros_like(table)))
    batch = parent._probe_batch()
    for mode in ("persistence", "zero"):
        with_table = _tiny_model(mode, anchor=anchor_spec, table=table,
                                 climatology_anchor_spec=anchor_spec,
                                 anchor_output_prior=False, anomaly_feedback=False)
        with_zero = _tiny_model(mode, anchor=zero_spec, table=torch.zeros_like(table),
                                climatology_anchor_spec=zero_spec,
                                anchor_output_prior=False, anomaly_feedback=False)
        with torch.no_grad():
            a, b = with_table(batch), with_zero(batch)
        assert torch.equal(a.initial_forecast, b.initial_forecast), mode
        assert torch.equal(a.forecast, b.forecast), mode
        assert torch.equal(a.draft_forecasts, b.draft_forecasts), mode


@pytest.mark.parametrize("bad", ["base", "", None, 0, True, "PERSISTENCE"])
def test_an_unknown_decoder_base_mode_is_refused(bad):
    with pytest.raises(ValueError, match="decoder_base_mode"):
        make_model("process", dict(parent._tiny_spec(), decoder_base_mode=bad))


def test_zero_with_the_climatology_output_prior_is_refused():
    """The switch cannot desynchronize the head anchor from base_state by construction."""
    anchor_spec, _ = _tiny_anchor()
    with pytest.raises(ValueError, match="anchor_output_prior"):
        make_model("process", dict(parent._tiny_spec(), decoder_base_mode="zero",
                                   climatology_anchor_spec=anchor_spec,
                                   anchor_output_prior=True))
    native = dict(architecture="window", in_channels=17, out_channels=17, history_steps=2,
                  dim=8, depth=1, heads=2, window_size=2, patch_size=2, dropout=0.,
                  climatology_anchor_spec=anchor_spec, decoder_base_mode="zero",
                  anchor_output_prior=True)
    with pytest.raises(ValueError, match="anchor_output_prior"):
        make_model("native", native)


def test_the_synthetic_base_switch_control_is_all_true():
    control = parent.base_switch_control({})
    assert control["synthetic"] is True
    assert all(control["checks"].values()), control["checks"]
    assert control["note"].startswith("synthetic tensors only")


# --------------------------------------------------------------------------- frozen driver


def test_frozen_constants_are_the_registered_recipe():
    assert driver.UPDATES == 2400 and driver.STEPS == driver.REASONING_STEPS == 4
    assert (driver.DIM, driver.DEPTH) == (192, 4) and driver.SEEDS == (41, 42, 43)
    assert driver.REGISTERED_PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert driver.N_PHYSICAL_STEPS == 12 and sum(driver.REGISTERED_PHYSICAL_WEIGHTS) == 3.0
    assert driver.physical_weight_block()["direct_steps"] == [0, 1, 3, 7, 11]
    assert driver.REGISTERED_DEEP_WEIGHTS == pytest.approx(
        (2 / 15, 1 / 6, 1 / 5, 7 / 30, 4 / 15), abs=1e-12)
    assert driver.draft_organization_block()["nonfinal_mass"] == pytest.approx(11. / 15., abs=1e-12)
    assert driver.decoder_base_block()["candidate_mode"] == "zero"
    assert driver.decoder_base_block()["control_mode"] == "persistence"
    assert (driver.BRIDGE_MIN, driver.BRIDGE_MAX) == (-4.8e-6, 1.8e-5)
    assert (driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) == (43200., 86400.)


def test_changed_frozen_constants_are_refused(monkeypatch):
    for name, value in (("CANDIDATE_DECODER_BASE_MODE", "persistence"),
                        ("CONTROL_DECODER_BASE_MODE", "zero")):
        monkeypatch.setattr(driver, name, value)
        with pytest.raises(RuntimeError):
            driver.decoder_base_block()
        monkeypatch.undo()
    monkeypatch.setattr(driver, "REGISTERED_PHYSICAL_WEIGHTS", (0.5,) * 12)
    with pytest.raises(RuntimeError):
        driver.physical_weight_block()
    monkeypatch.undo()
    monkeypatch.setattr(driver, "REGISTERED_DEEP_WEIGHTS", (0.2,) * 5)
    with pytest.raises(RuntimeError):
        driver.draft_organization_block()


def test_arm_spec_is_the_anchor_control_spec_plus_one_switch(anchor_stub):
    candidate = driver.arm_spec()
    control = driver.arm_spec("persistence")
    assert candidate["decoder_base_mode"] == "zero"
    assert control["decoder_base_mode"] == "persistence"
    assert driver._without_mode(candidate) == driver._without_mode(control)
    assert candidate["climatology_anchor_spec"] == anchor_stub
    assert candidate["anchor_output_prior"] is False and candidate["anomaly_feedback"] is False
    assert (candidate["dim"], candidate["depth"]) == (192, 4)
    assert driver.arm_spec("zero") == candidate


def test_arm_spec_refuses_a_switched_or_wrong_shaped_control_spec(monkeypatch):
    bad_specs = (_archived_spec(anchor_output_prior=False), _archived_spec(dim=128),
                 _archived_spec(depth=8), _archived_spec(anomaly_feedback=True))
    for bad in bad_specs:
        monkeypatch.setattr(driver.anchor_arm, "arm_spec", lambda *_a, _bad=bad, **_k: dict(_bad))
        with pytest.raises(RuntimeError):
            driver.arm_spec()


def test_protocol_constants_carry_only_the_decoder_base_change(anchor_stub):
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400 and body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert body["decoder_base"] == driver.decoder_base_block()
    assert body["candidate_draft_supervision"] == body["control_draft_supervision"] == "deep"
    assert body["draft_supervision_organization"] == driver.draft_organization_block()
    assert body["physical_weight_vector"] == driver.physical_weight_block()
    assert body["seeds"] == [41, 42, 43]
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["budget_profile_note"] == driver.BUDGET_PROFILE_NOTE
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "zero" in body["difference"] and "K=4" in body["difference"]


def test_the_identity_pins_the_candidate_and_both_references(anchor_stub, monkeypatch):
    monkeypatch.setattr(interior, "_identity", lambda: {
        "arms": {"narrow_control": {"record_id": "narrow"}}, "train_windows": {"x": 1},
        "train_data_identity": "train-id", "source_sha256": "s" * 64,
        "training_code_sha256": "0" * 64})
    monkeypatch.setattr(driver, "previous_arm_pins", lambda: {"record_id": "dose"})
    monkeypatch.setattr(driver, "paired_control_pins",
                        lambda *a, **k: {"record_id": "anchor",
                                         "training_code_sha256": "c" * 64})
    monkeypatch.setattr(parent, "install_parent_pins",
                        lambda body: body["arms"]["parent"].update(pins={"ok": 1}))
    monkeypatch.setattr("training.r7_arm_harness.sha256_file", lambda path: "0" * 64)
    body = driver._identity()
    candidate = body["arms"][driver.CANDIDATE]
    assert candidate["decoder_base_mode"] == "zero"
    assert candidate["spatial_gradient_lambda"] == 0.0 and candidate["draft_supervision"] == "deep"
    assert candidate["model"]["decoder_base_mode"] == "zero"
    assert candidate["model"]["anchor_keys"] == list(parent.ANCHOR_KEYS)
    assert candidate["selection"] == "frozen endpoint; no validation selection"
    assert body["arms"]["previous_dose_arm"] == {"record_id": "dose"}
    assert body["arms"]["paired_control_arm"]["record_id"] == "anchor"
    assert body["arms"]["narrow_control"] == {"record_id": "narrow"}
    assert candidate["model"]["spec_canonical_digest"] == canonical_digest(driver.arm_spec())
    assert body["decoder_base"] == driver.decoder_base_block()
    assert body["arms"]["parent"]["pins"] == {"ok": 1}
    assert body["code_version_comparison"]["control_training_code_sha256"] == "c" * 64
    assert body["code_version_comparison"]["candidate_training_code_sha256"] == "0" * 64
    assert body["code_version_comparison"]["equal"] is False


def test_train_arm_contract_carries_the_zero_base_and_no_other_axis(anchor_stub):
    body = {"train_data_identity": "train-id", "source_sha256": "s" * 64,
            "protocol_sha256": "p" * 64, "training_code_sha256": "t" * 64,
            "train_windows": {"window_sha256": "w" * 64, "excluded_sample_ids": []},
            "supervision": {"runner_block": {"stub": 1}}}
    expansion = {"anchor_table_sha256": "a" * 64}
    contract = driver._arm_contract(body, driver.arm_spec("zero"), "parent.pt", "f" * 64,
                                    expansion)
    assert contract["model"]["decoder_base_mode"] == "zero"
    autoregression = contract["autoregression"]
    assert autoregression["decoder_base_mode"] == "zero"
    assert autoregression["physical_weights"] == list(driver.REGISTERED_PHYSICAL_WEIGHTS)
    assert autoregression["draft_supervision"] == "deep"
    assert "ema_decay" not in autoregression and "spatial_gradient_lambda" not in autoregression


def test_candidate_receipt_must_be_the_final_zero_base_weights(tmp_path):
    receipt = {"seed": 41, "physical_weights": list(driver.REGISTERED_PHYSICAL_WEIGHTS),
               "draft_supervision": "deep", "decoder_base_mode": "persistence",
               "scored_checkpoint_kind": "final_optimizer_weights",
               "checkpoint": str(tmp_path / "update_0002400.pt"), "evaluations": {}}
    (tmp_path / "seed41_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero-base"):
        report.candidate_readings(tmp_path, {"wide_climatology_reference": {"leads": {}}}, 41)


def _seed_block(skill72, delta72, skill48, delta_short, bridge):
    return {"candidate_skill": {"6": 0.5, "12": 0.4, "24": 0.3, "48": skill48, "72": skill72},
            "delta_candidate_minus_control": {"6": delta_short, "12": delta_short,
                                              "24": delta_short, "48": -0.1, "72": delta72},
            "control_minus_previous_dose": {"6": bridge, "12": bridge, "24": bridge,
                                            "48": bridge, "72": bridge}}


def _seed_map(**overrides):
    blocks = {}
    for index, seed in enumerate(driver.SEEDS):
        values = {"skill72": 0.01, "delta72": -0.01, "skill48": 0.02,
                  "delta_short": -0.001, "bridge": 0.0}
        values.update({key: value[index] if isinstance(value, (list, tuple)) else value
                       for key, value in overrides.items()})
        blocks[str(seed)] = _seed_block(**values)
    return blocks


def test_decision_flags_all_true_when_every_predeclared_condition_holds():
    flags = driver.decision_flags(_seed_map())
    assert flags["scientific_exit_met"] is True and flags["predeclared_exit_met"] is True
    assert flags["identity_unresolved"] is False


@pytest.mark.parametrize("override", [
    {"skill72": (0.01, -0.02, 0.03)}, {"delta72": (-0.01, 0.02, -0.01)},
    {"skill48": (-0.01, 0.02, 0.01)}, {"delta_short": (0.001, 0.001, 0.001)}])
def test_each_scientific_condition_can_fail_alone(override):
    flags = driver.decision_flags(_seed_map(**override))
    assert flags["scientific_exit_met"] is False and flags["predeclared_exit_met"] is False


def test_short_lead_uses_the_seed_mean_not_each_seed():
    flags = driver.decision_flags(_seed_map(delta_short=(-0.002, 0.001, 0.0005)))
    assert flags["d_short_lead_seed_mean_not_worse"] is True


def test_out_of_band_bridge_is_identity_unresolved_not_a_science_failure():
    flags = driver.decision_flags(_seed_map(bridge=2e-5))
    assert flags["scientific_exit_met"] is True and flags["predeclared_exit_met"] is False
    assert flags["e_bridge_consistent"] is False and flags["identity_unresolved"] is True


def test_parent_constants_pin_the_predeclared_migration():
    assert parent.CURRENT_MODEL_CODE_SHA256 == "bec3859fc1386b14777300ad3e387546829c8a0fbf52e817e1cb90324b988ac1"
    assert parent.MIGRATION_RECEIPT_SHA256 == \
        "002f08439b99fc3658a422e0732183f7a48c5b43d5790c133c53c4eaf004d6c3"
    assert sorted(parent.MIGRATED_PARENT_SHA256) == [41, 42, 43]
    assert parent.MIGRATED_PARENT_SHA256 == {
        41: "e94b1dd006ee970082c4792599d658617ad924c24241d44a048b5164a5d491a2",
        42: "4a9556d32c1e92fce623259380fbfa9cd9d82e33587833d62ae0b31e3702c613",
        43: "dba3048a6bf0ae48a61caf26036baa037d89fe55bb0178b88760b5be1789bec9"}
    assert parent.MIGRATED_PARENT_STATE_DIGEST == {
        41: "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707",
        42: "857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623",
        43: "61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b"}
    assert parent.EXPANDED_STATE_SHA256 == {
        41: "00ada5e25c4e7ef16acfb24768408f06e5e6ab091d4a91bdc7505c99229de612",
        42: "baed2ff9ca898af7c6590f5ba269411d6bd660bd9601b5bcee7ae2c2a2b660a1",
        43: "4dda647f86ffee6befcab749e04153e7dbf09c6034e62d03a8f7d00bf066cce7"}
    assert parent.ANCHOR_INIT_SHA256 == \
        "de0e76beb548881b282fbd8c6f194055286ec7d0dd1d7ba88dad476fc2aa1112"
    assert parent.MIGRATED_PARENT_RUN.name == \
        "r7_s3_decoder_base_anchor_parent_migrated_20261020_attempt02"
    assert parent.ANCHOR_TABLE_RUN.name == "r7_s3_anchor_table_20261015_attempt02"


def test_execution_files_cover_the_imported_control_scripts():
    files = driver.execution_files()
    for name in ("scripts/study_r7_s3_anchor_isolation.py",
                 "scripts/r7_s3_anchor_isolation_parent.py",
                 "scripts/r7_s3_decoder_base_anchor_support.py",
                 "model/weather_forecaster_r7.py", "model/process_forecast_r7.py"):
        assert name in files, name


# --------------------------------------------------------------------------- round spec


def test_round_spec_matches_the_driver_and_splits_the_two_commits():
    text = (ROOT / "docs" / "goals" / "s3-decoder-base-anchor.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-decoder-base-anchor"
    assert payload["driver"] == "scripts/study_r7_s3_decoder_base_anchor.py"
    assert payload["output_stem"] == "r7_s3_decoder_base_anchor_20261020"
    assert payload["control_record"] == driver.PAIRED_CONTROL_RECORD
    assert payload["seeds"] == list(driver.SEEDS)
    assert payload["test_path"] == ["tests/test_r7_s3_decoder_base_anchor.py",
                                    "tests/test_r7_s3_decoder_base_anchor_protocol.py"]
    assert payload["code_archive"].endswith(".zip")
    assert "docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md" in payload["register_paths"]
    assert "docs/R7_S3_DECODER_BASE_ANCHOR_RESULT.md" not in payload["commit_paths"]
    for name in ("scripts/study_r7_s3_decoder_base_anchor.py",
                 "scripts/r7_s3_decoder_base_anchor_parent.py",
                 "scripts/r7_s3_decoder_base_anchor_support.py",
                 "scripts/r7_s3_decoder_base_anchor_text.py",
                 "tests/test_r7_s3_decoder_base_anchor.py",
                 "tests/test_r7_s3_decoder_base_anchor_protocol.py",
                 "model/weather_forecaster_r7.py", "model/process_forecast_r7.py",
                 "docs/goals/s3-decoder-base-anchor.md"):
        assert name in payload["commit_paths"]
    assert "docs/goals/main-model-climatology-campaign.md" in payload["register_paths"]
    assert payload["commit_message"] and payload["register_message"]


def test_control_and_reference_constants_are_the_registered_ones():
    assert driver.PAIRED_CONTROL_RECORD == "s3-anchor-isolation-result"
    assert driver.PAIRED_CONTROL_RUN.name == "r7_s3_anchor_isolation_20261015_attempt01"
    assert driver.PREVIOUS_DOSE_ARM_RECORD == "s3-wide-interior-dose"
    assert driver.CANDIDATE_DECODER_BASE_MODE == "zero"
    assert driver.CONTROL_DECODER_BASE_MODE == "persistence"
    assert sorted(parent.MIGRATED_PARENT_SHA256) == [41, 42, 43]
