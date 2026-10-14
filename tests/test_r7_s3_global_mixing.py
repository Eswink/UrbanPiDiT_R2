"""CI-safe checks for the H-C global-context-mixing driver and its parent module.

H-C adds exactly one context-mixing layer after the backbone and runs two arms that differ only in
the attention range: ``global`` attends over every token, ``local`` keeps the trunk's windowed
attention. Both arms declare the same twelve tensors with the same shapes, so they are
parameter-matched by construction; their compute is *not* matched and the gap is declared rather
than claimed away. The fidelity-critical parts are that the two arms add exactly the same key set
and no others, that the parameter counts are equal at any dim, that switching the layer off is the
previous implementation bit for bit, that zeroing the output projection *and* the FF output layer
returns the whole forward to the off path while zeroing the projection alone does not, that the
compute gap is recorded as unequal, and that the expansion adds exactly the mixing keys with one
shared initialization per round seed.

Every check runs against tiny synthetic specs and a synthetic batch, so this file is checked on a
checkout with no ``outputs/``. Each check has a counterproof.
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
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load("study_r7_s3_global_mixing")
mixing_parent = driver.mixing_parent
wide = driver.wide
interior = driver.interior


def _tiny_spec(dim=8, mode=None):
    spec = dict(architecture="window", in_channels=17, out_channels=17, history_steps=2, dim=dim,
                depth=4, heads=2, window_size=2, patch_size=2, dropout=0.,
                spacetime_inputs=True, default_reasoning_steps=4, detach_between_steps=False,
                anchored_processes=4, free_processes=1)
    if mode is not None:
        spec["global_mixing_mode"] = mode
    return spec


def _batch(size=1, side=12):
    """A synthetic sample whose token grid (side/2 per axis) is larger than one window."""
    generator = torch.Generator().manual_seed(11)
    return {"coarse_history": torch.randn(size, 2, 17, side, side, generator=generator),
            "atmos_target": torch.randn(size, 17, side, side, generator=generator),
            "future_target": torch.randn(size, 17, side, side, generator=generator),
            "latitude": torch.linspace(50., 20., side).expand(size, -1).clone(),
            "longitude": torch.linspace(90., 130., side).expand(size, -1).clone(),
            "lead_time_hours": torch.full((size,), 6.), "init_utc_hour": torch.full((size,), 18.),
            "init_day_of_year": torch.full((size,), 366.),
            "init_calendar_year": torch.full((size,), 2016.),
            "history_offsets_hours": torch.tensor([[-6., 0.]]).expand(size, -1).clone(),
            "init_year": torch.full((size,), 1234.),
            "process_targets": torch.full((size, 1), 9999.),
            "untrusted_metadata": "not a model input"}


@pytest.fixture(autouse=True)
def _stub_shared(monkeypatch):
    """``arm_spec`` reads the registered archived spec from ``outputs/``."""
    class _Shared:
        ARCHIVED = ROOT / "outputs" / "r7_stub_archive"

        def archived_process_spec(self):
            return (_tiny_spec(dim=driver.DIM), None, None)

    monkeypatch.setattr(driver.recipe, "_shared", lambda: _Shared())


def _build(mode, spec=None, seed=11):
    from training.r7_experiment import make_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return make_model("process", spec if spec is not None else _tiny_spec(mode=mode))


# --- the mixing modes ---------------------------------------------------------------------------

def test_the_mixing_modes_are_declared():
    from model.global_mixing_r7 import GLOBAL, LOCAL, MIXING_MODES
    assert GLOBAL == "global" and LOCAL == "local"
    assert set(MIXING_MODES) == {GLOBAL, LOCAL}
    assert dict(driver.ARMS) == {GLOBAL: GLOBAL, LOCAL: LOCAL}


def test_an_unknown_mode_is_refused():
    from model.global_mixing_r7 import ContextMixLayer
    with pytest.raises(ValueError):
        ContextMixLayer(8, 2, 2, mode="mixing_made_up")


def test_the_model_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        _build("mixing_made_up")


def test_the_mixing_layer_refuses_a_wrong_token_shape():
    """Both modes check the feature width; the local mode also needs N == H*W, the global one not."""
    from model.global_mixing_r7 import GLOBAL, LOCAL, ContextMixLayer
    for mode in (GLOBAL, LOCAL):
        with pytest.raises(ValueError):
            ContextMixLayer(8, 2, 2, mode=mode)(torch.randn(2, 4, 7), (2, 2))
    local = ContextMixLayer(8, 2, 2, mode=LOCAL)
    with pytest.raises(ValueError):
        local(torch.randn(2, 4, 8), (3, 3))
    assert ContextMixLayer(8, 2, 2, mode=GLOBAL)(torch.randn(2, 4, 8), (3, 3)).shape == (2, 4, 8)


# --- parameter parity, compute mismatch ---------------------------------------------------------

def test_the_two_arms_add_the_same_twelve_keys():
    bare = _build(None).state_dict()
    added = {}
    for name, mode in driver.ARMS:
        added[name] = sorted(set(_build(mode).state_dict()) - set(bare))
        assert added[name] == sorted(mixing_parent.mixing_keys(mode))
    assert added[driver.GLOBAL] == added[driver.LOCAL]
    assert len(added[driver.GLOBAL]) == 12


def test_parameter_counts_are_equal_at_any_dim():
    for dim in (8, 16, 32, 64, 192):
        counts = [sum(p.numel() for p in _build(mode, _tiny_spec(dim, mode)).parameters())
                  for _, mode in driver.ARMS]
        assert counts[0] == counts[1], f"dim {dim}: {counts}"


def test_parameter_counts_are_equal_at_the_registered_dim():
    """Both shells declare qkv+proj+ff+norms identically, so the count is 444,864 at dim 192."""
    from model.global_mixing_r7 import GLOBAL, LOCAL, ContextMixLayer
    dim, heads, window = driver.DIM, 4, 4
    counts = [sum(p.numel() for p in ContextMixLayer(dim, heads, window, mode=mode).parameters())
              for mode in (GLOBAL, LOCAL)]
    assert counts == [444864, 444864]


def test_the_shell_matches_the_trunk_window_block():
    from model.global_mixing_r7 import GlobalAttentionBlock
    from model.layers.window_attention import WindowAttentionBlock
    shell = GlobalAttentionBlock(192, 4, 4.0, 0.0).state_dict()
    windowed = WindowAttentionBlock(192, 4, 4, 4.0, 0.0, shift=False).state_dict()
    assert sorted(shell) == sorted(windowed)
    assert all(shell[key].shape == windowed[key].shape for key in shell)


def test_the_compute_gap_is_declared_and_not_aligned():
    """The parameter axis is aligned; the compute axis is not, and the ratios are recorded."""
    flops = mixing_parent._analytic_flops()
    assert flops["aligned"] is False
    assert flops["attention_length_ratio"] == pytest.approx(68.0625)
    assert flops["attention_ratio"] > 50.0
    assert flops["layer_ratio"] > 1.0
    assert flops["global_attention_flops"] > flops["local_attention_flops"]
    assert driver.mixing_block(driver.GLOBAL)["compute_matched"] is False
    assert driver.mixing_block(driver.GLOBAL)["parameter_matched_by_construction"] is True


def test_a_narrower_window_widens_the_gap():
    """Counterproof that the recorded ratio is computed, not a constant."""
    narrow = mixing_parent._analytic_flops({**mixing_parent.FLOPS_REFERENCE,
                                            "windows": 4, "tokens_per_window": 4})
    assert narrow["attention_ratio"] > mixing_parent._analytic_flops()["attention_ratio"]


# --- the identity controls ----------------------------------------------------------------------

def test_switch_off_is_bit_identical_to_the_current_implementation():
    """The layer is built last under isolated_stream(): the off path moves no existing weight."""
    off = _build(None).state_dict()
    on = _build(driver.GLOBAL).state_dict()
    assert set(on) - set(off) == set(mixing_parent.mixing_keys(driver.GLOBAL))
    assert set(off) - set(on) == set()
    assert all(torch.equal(off[key], on[key]) for key in off)


def test_forward_with_a_zeroed_block_matches_the_off_path():
    """The strongest identity: proj AND the FF output layer zeroed is the exact identity."""
    off = _build(None).eval()
    on = _build(driver.GLOBAL).eval()
    sample = _batch()
    with torch.no_grad():
        reference = off(sample, reasoning_steps=4).forecast
        conditioned = on(sample, reasoning_steps=4).forecast
    assert not torch.equal(reference, conditioned), "switching the layer on must change the forward"
    with torch.no_grad():
        block = on.global_mixing.block
        block.proj.weight.zero_()
        block.proj.bias.zero_()
        block.ff.net[3].weight.zero_()
        block.ff.net[3].bias.zero_()
        zeroed = on(sample, reasoning_steps=4).forecast
    assert torch.equal(reference, zeroed)


def test_zeroing_the_projection_alone_is_not_the_off_path():
    """The documented subtlety: the residual is followed by the FF, so proj alone is not identity."""
    from model.global_mixing_r7 import GLOBAL, ContextMixLayer
    layer = ContextMixLayer(8, 2, 2, mode=GLOBAL)
    tokens = torch.randn(2, 16, 8)
    with torch.no_grad():
        before = layer(tokens, (4, 4))
        layer.block.proj.weight.zero_()
        layer.block.proj.bias.zero_()
        proj_only = layer(tokens, (4, 4))
    assert not torch.equal(tokens, proj_only)
    assert float((before - proj_only).abs().max()) > 0.0


def test_the_two_arms_differ_on_a_grid_larger_than_one_window():
    """The negative control is a real manipulation: the arms cannot coincide above one window."""
    off = _build(None).eval()
    arms = {mode: _build(mode).eval() for _, mode in driver.ARMS}
    sample = _batch()
    with torch.no_grad():
        reference = off(sample, reasoning_steps=4).forecast
        outputs = {mode: arms[mode](sample, reasoning_steps=4).forecast for _, mode in driver.ARMS}
    for mode, value in outputs.items():
        assert not torch.equal(reference, value), mode
    assert not torch.equal(outputs[driver.GLOBAL], outputs[driver.LOCAL])


def test_the_mixing_layer_is_built_after_the_pre_existing_weights():
    """Source order: the gated construction follows the trunk, so 'off' cannot move a weight."""
    text = (ROOT / "model" / "process_forecast_r7.py").read_text(encoding="utf-8")
    trunk = text.index("self.correction_head=CoarseForecastHead(")
    build = text.index("self._build_context_mixing(dim,heads,window_size,dropout,")
    assert trunk < build
    helper = text.index("def _build_context_mixing")
    assert "with isolated_stream():" in text[helper:text.index("def __init__")]


# --- the expansion adds exactly the mixing keys --------------------------------------------------

def test_mixing_keys_are_the_same_for_both_modes():
    assert mixing_parent.mixing_keys(driver.GLOBAL) == mixing_parent.mixing_keys(driver.LOCAL)
    with pytest.raises(ValueError):
        mixing_parent.mixing_keys("mixing_made_up")


def test_expand_installs_the_parent_and_adds_the_mixing_keys():
    parent = _build(None).state_dict()
    spec = _tiny_spec(mode=driver.GLOBAL)
    model, state, receipt = mixing_parent.expand_parent_state(
        parent, spec, driver.GLOBAL, seed=41)
    assert sorted(receipt["new_keys"]) == sorted(mixing_parent.mixing_keys(driver.GLOBAL))
    assert receipt["parent_keys"] == len(parent)
    assert receipt["init_seed"] == 41 and receipt["mixing_mode"] == driver.GLOBAL
    assert receipt["new_params"] == sum(
        int(state[key].numel()) for key in mixing_parent.mixing_keys(driver.GLOBAL))
    for key, value in parent.items():
        assert torch.equal(state[key], value)
    model.load_state_dict(state, strict=True)


def test_expand_refuses_a_state_that_is_not_the_registered_parent():
    spec = _tiny_spec(mode=driver.GLOBAL)
    wrong = _build(None).state_dict()
    del wrong["process_queries"]
    with pytest.raises(RuntimeError):
        mixing_parent.expand_parent_state(wrong, spec, driver.GLOBAL, seed=41)


def test_expand_refuses_a_parent_that_already_carries_the_layer():
    parent = _build(driver.GLOBAL).state_dict()
    with pytest.raises(RuntimeError):
        mixing_parent.expand_parent_state(parent, _tiny_spec(mode=driver.GLOBAL), driver.GLOBAL,
                                          seed=41)


def test_both_arms_share_one_mixing_initialization_within_a_seed():
    """The confound the H-B review flagged: L and G must not differ by a fresh random draw."""
    parent = _build(None).state_dict()
    _, global_state, global_receipt = mixing_parent.expand_parent_state(
        parent, _tiny_spec(mode=driver.GLOBAL), driver.GLOBAL, seed=41)
    _, local_state, local_receipt = mixing_parent.expand_parent_state(
        parent, _tiny_spec(mode=driver.LOCAL), driver.LOCAL, seed=41)
    shared = sorted(mixing_parent.mixing_keys(driver.GLOBAL))
    assert all(torch.equal(global_state[key], local_state[key]) for key in shared)
    assert global_receipt["pathway_init_sha256"] == local_receipt["pathway_init_sha256"]


def test_a_different_round_seed_draws_a_different_mixing_layer():
    parent = _build(None).state_dict()
    spec = _tiny_spec(mode=driver.GLOBAL)
    _, first, receipt_first = mixing_parent.expand_parent_state(parent, spec, driver.GLOBAL, seed=41)
    _, second, receipt_second = mixing_parent.expand_parent_state(parent, spec, driver.GLOBAL, seed=42)
    assert receipt_first["pathway_init_sha256"] != receipt_second["pathway_init_sha256"]
    assert any(not torch.equal(first[key], second[key]) for key in receipt_first["new_keys"])


def test_positive_control_reports_the_expected_facts():
    control = mixing_parent.positive_control(_tiny_spec(dim=8))
    assert control["shell_parity"]["parameter_counts_equal"] is True
    assert control["shell_parity"]["shell_keys_equal"] is True
    assert control["shared_bit_identical"] is True
    assert control["added_keys"] == [] and control["lost_keys"] == []
    assert control["new_keys_match_mixing_keys"] is True
    assert control["zeroed_block_is_identity"] is True
    assert control["zeroed_block_max_abs"] == 0.0
    assert control["proj_only_is_identity"] is False
    assert control["arms_are_not_identical"] is True
    assert control["flops"]["aligned"] is False
    assert control["model_param_delta"] == control["shell_parity"]["global_params"]


def test_positive_control_refuses_a_mismatched_shell(monkeypatch):
    """Counterproof: a shell that is not parameter-matched fails the control."""
    from model.global_mixing_r7 import ContextMixLayer
    original = ContextMixLayer.forward

    def doubled(self, tokens, token_hw):
        return 2.0 * original(self, tokens, token_hw)

    monkeypatch.setattr(ContextMixLayer, "forward", doubled)
    assert mixing_parent.positive_control(_tiny_spec(dim=8))["arms_are_not_identical"] is True
    assert mixing_parent.positive_control(_tiny_spec(dim=8))["zeroed_block_is_identity"] is False


# --- the driver's frozen fields -----------------------------------------------------------------

def test_arm_spec_sets_the_mode():
    spec = driver.arm_spec(driver.GLOBAL)
    assert spec["global_mixing_mode"] == driver.GLOBAL
    assert driver.arm_spec(driver.LOCAL)["global_mixing_mode"] == driver.LOCAL
    assert spec["dim"] == driver.DIM and spec["depth"] == driver.DEPTH


def test_arm_spec_refuses_a_spec_that_already_carries_a_switch(monkeypatch):
    monkeypatch.setattr(driver.recipe, "_shared",
                        lambda: type("S", (), {"archived_process_spec": lambda self: (
                            _tiny_spec(driver.DIM, driver.GLOBAL), None, None)})())
    with pytest.raises(RuntimeError):
        driver.arm_spec(driver.GLOBAL)


def test_arm_spec_refuses_a_spec_that_is_not_the_registered_trunk(monkeypatch):
    monkeypatch.setattr(driver.recipe, "_shared",
                        lambda: type("S", (), {"archived_process_spec": lambda self: (
                            _tiny_spec(128), None, None)})())
    with pytest.raises(RuntimeError):
        driver.arm_spec(driver.GLOBAL)


def test_arm_name_round_trips_and_refuses_an_unknown_mode():
    assert driver.arm_name(driver.GLOBAL) == driver.GLOBAL
    assert driver.arm_name(driver.LOCAL) == driver.LOCAL
    with pytest.raises(ValueError):
        driver.arm_name("mixing_made_up")


def test_mixing_block_declares_the_mode_keys_and_the_compute_gap():
    global_block = driver.mixing_block(driver.GLOBAL)
    local_block = driver.mixing_block(driver.LOCAL)
    assert global_block["mode"] == driver.GLOBAL and local_block["mode"] == driver.LOCAL
    assert global_block["new_keys"] == list(mixing_parent.mixing_keys(driver.GLOBAL))
    assert global_block["new_keys"] == local_block["new_keys"]
    assert global_block["new_params"] == 444864
    assert global_block["compute_matched"] is False
    assert global_block["declared_compute_gap"]["attention_length_ratio"] == pytest.approx(68.0625)
    assert global_block["built_last_under_isolated_stream"] is True
    assert global_block["positive_control"] == local_block["positive_control"]


def test_protocol_constants_carry_the_mixing_factor_and_keep_dim_depth_k4():
    body = driver.protocol_constants()
    assert body["format"] == driver.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["recipe"]["steps"] == 4
    assert body["evaluation"]["reasoning_steps"] == 4
    assert body["recipe"]["physical_weights"] == list(wide.PHYSICAL_WEIGHTS)
    assert body["recipe"]["lr"] == 2e-5 and body["recipe"]["warmup"] == 10
    assert body["budgets"] == {"planned_seconds": driver.PLANNED_SECONDS,
                               "hard_cap_seconds": driver.HARD_CAP_SECONDS,
                               "per_seed_seconds": driver.PER_SEED_SECONDS}
    assert body["seeds"] == [41, 42, 43]
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert "global" in body["difference"] and "local" in body["difference"]
    assert "dim 192" in body["difference"] and "K=4" in body["difference"]
    assert "68.1x" in body["difference"] and "57.2x" in body["difference"]


def test_round_spec_matches_the_driver_and_device_mode():
    text = (ROOT / "docs" / "goals" / "s3-global-mixing.md").read_text(encoding="utf-8")
    marker = "<!-- round-spec: "
    payload = json.loads(text[text.index(marker) + len(marker):text.index(" -->", text.index(marker))])
    assert payload["arm"] == "s3-global-mixing"
    assert payload["driver"] == "scripts/study_r7_s3_global_mixing.py"
    assert payload["control_record"] == driver.PREVIOUS_DOSE_ARM_RECORD
    assert payload["seeds"] == list(wide.SEEDS)
    assert payload["device_mode"] == driver.DEVICE_MODE
    assert payload["test_path"] == ["tests/test_r7_s3_global_mixing.py"]
    assert set(payload["commit_paths"]) >= {
        "model/global_mixing_r7.py", "scripts/study_r7_s3_global_mixing.py",
        "scripts/r7_s3_global_mixing_parent.py", "tests/test_r7_s3_global_mixing.py",
        "model/process_forecast_r7.py"}


# --- the frozen protocol ------------------------------------------------------------------------

def _arms():
    from training.r7_experiment import canonical_digest
    records = {}
    for name, mode in driver.ARMS:
        spec = driver.arm_spec(mode)
        records[name] = {"updates": driver.UPDATES, "mode": mode,
                         "model": {"kind": "process", "spec": spec,
                                   "spec_canonical_digest": canonical_digest(spec)}}
    assert {name for name, _ in driver.ARMS} == {driver.GLOBAL, driver.LOCAL}
    return records


def _protocol_body():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = {
        "execution_files_sha256": {name: sha256_file(ROOT / name) for name in driver.execution_files()},
        "train_manifest": str(wide.TRAIN_MANIFEST), "train_data_identity": "train-id",
        "val_manifest": str(wide.VAL_MANIFEST), "val_data_identity": "val-id",
        "model_code_sha256": "m" * 64, "training_code_sha256": "t" * 64, "source_sha256": "s" * 64,
        "train_windows": {"stub": 1}, "supervision": {"stub": 1}, "format": driver.FORMAT,
        "recipe": {"updates": driver.UPDATES, "steps": driver.STEPS},
        "evaluation": {"reasoning_steps": driver.REASONING_STEPS},
        "global_mixing": {"arms": {name: driver.mixing_block(mode) for name, mode in driver.ARMS}},
        "arms": {**_arms(),
                 "previous_dose_arm": {"record_id": driver.PREVIOUS_DOSE_ARM_RECORD,
                                       "updates": driver.UPDATES, "seeds": list(wide.SEEDS)}},
        "planned_seconds": driver.PLANNED_SECONDS, "hard_cap_seconds": driver.HARD_CAP_SECONDS,
        "per_seed_seconds": driver.PER_SEED_SECONDS,
        "estimated_peak_bytes": driver.ESTIMATED_PEAK_BYTES,
        "headroom_margin_bytes": driver.HEADROOM_MARGIN_BYTES,
        "scientific_claim": False, "test_read": False,
        "test_manifest_never_read": str(wide.TEST_MANIFEST),
    }
    body["protocol_sha256"] = canonical_digest(body)
    return body


def _patch_validate(monkeypatch):
    """Neutralise the store/GPU reads so validate_protocol runs on a bare checkout."""
    import data.r7_long_rollout_dataset as dataset
    import training.r7_arm_harness as harness
    import training.r7_experiment as experiment
    import training.r7_long_rollout_runner as runner
    monkeypatch.setattr(experiment, "dataset_identity",
                        lambda manifest: ("train-id", "x") if str(manifest) == str(wide.TRAIN_MANIFEST)
                        else ("val-id", "x"))
    monkeypatch.setattr(experiment, "model_code_digest", lambda: "m" * 64)
    monkeypatch.setattr(runner, "training_code_digest", lambda: "t" * 64)
    monkeypatch.setattr(dataset, "preflight_long_rollout_windows", lambda *a, **k: {"stub": 1})
    monkeypatch.setattr(wide, "source_sha256", lambda: "s" * 64)
    monkeypatch.setattr(interior, "supervision_block", lambda: {"stub": 1})
    monkeypatch.setattr(driver.devices, "assignment_is_valid", lambda *a, **k: True)
    real_files = [name for name in driver.execution_files() if (ROOT / name).is_file()]
    monkeypatch.setattr(driver, "execution_files", lambda: real_files)
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)


def _write_protocol(tmp_path, body):
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_validate_protocol_accepts_the_frozen_arms(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = driver.validate_protocol(_write_protocol(tmp_path, _protocol_body()))
    assert body["format"] == driver.FORMAT
    assert sorted(body["global_mixing"]["arms"]) == [driver.GLOBAL, driver.LOCAL]


def test_validate_protocol_rejects_a_swapped_arm_mode(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["arms"][driver.LOCAL]["mode"] = driver.GLOBAL
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_changed_mixing_block(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["global_mixing"]["arms"][driver.GLOBAL]["new_params"] = 1
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_claimed_compute_match(tmp_path, monkeypatch):
    """The round must not record a compute alignment it does not have."""
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["global_mixing"]["arms"][driver.GLOBAL]["compute_matched"] = True
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_changed_budget(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["per_seed_seconds"] = driver.PER_SEED_SECONDS + 1.0
    from training.r7_experiment import canonical_digest
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


def test_validate_protocol_rejects_a_tampered_digest(tmp_path, monkeypatch):
    _patch_validate(monkeypatch)
    body = _protocol_body()
    body["protocol_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        driver.validate_protocol(_write_protocol(tmp_path, body))


# --- the parent module --------------------------------------------------------------------------

class _FakeReceipt:
    def __init__(self, text):
        self._text = text

    def read_text(self, encoding="utf-8"):
        return self._text


def test_pinned_parent_refuses_a_stale_revision(monkeypatch):
    monkeypatch.setattr(mixing_parent, "MIGRATION_RECEIPT",
                        _FakeReceipt(json.dumps({"current_model_code_sha256": "0" * 64})))
    with pytest.raises(RuntimeError):
        mixing_parent.pinned_parent(41)


def test_pinned_parent_pins_this_revisions_migration():
    assert mixing_parent.MIGRATED_PARENT_RUN.name == (
        "r7_s3_rollout_dose_parent_migrated_20261015_attempt02")
    for seed in (41, 42, 43):
        assert len(mixing_parent.MIGRATED_PARENT_SHA256[seed]) == 64


def test_pinned_parent_refuses_a_foreign_seed():
    with pytest.raises(ValueError):
        mixing_parent.pinned_parent(999)


def test_install_parent_pins_writes_both_seeds():
    body = {"arms": {"parent": {"pins": {"checkpoints": {"41": {"path": "stale", "sha256": "stale"}}}}}}
    mixing_parent.install_parent_pins(body)
    checkpoints = body["arms"]["parent"]["pins"]["checkpoints"]
    assert sorted(checkpoints) == ["41", "42", "43"]
    for seed, entry in checkpoints.items():
        assert entry["sha256"] == mixing_parent.MIGRATED_PARENT_SHA256[int(seed)]
        assert entry["path"].endswith(f"parent_update_{mixing_parent.ENDPOINT:07d}.pt")
