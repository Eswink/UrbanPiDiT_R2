"""CI-safe tests for the R2 anchor-feedback driver (no ``outputs/`` required).

The round's data artifacts are large and live outside the repository, so these tests cover
the frozen protocol shape, the one-factor claim, the matched-compute constants, the anchor
table's role as a shared, parameter-free constant, the predeclared verdict logic, the bridge
table and the parent/model-revision guards. Nothing here reads a store, a manifest or the
published anchor table: ``frozen_constants`` takes the payload as a parameter exactly so the
synthetic one can stand in.

The counterexamples matter as much as the positive cases: a driver that silently drifted back
to the closed round's 2,400 updates at batch 1, that let the two arms read different manifests,
or that let the anchor leak into one arm and not the other, would still satisfy every "the
constant equals itself" assertion.
"""
from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pytest
import torch

from scripts import r7_s3_anchor_feedback_support as support
from scripts import r7_s3_anchor_isolation_parent as anchor_parent
from scripts import study_r7_s3_anchor_feedback_65x65 as driver

ROOT = Path(driver.__file__).resolve().parents[1]
GOAL = "docs/goals/s3-anchor-anomaly-feedback-reeval.md"


def _synthetic_payload():
    """A valid 48-bucket anchor payload on a tiny grid; never weather statistics."""
    channels, height, width = 17, 4, 5
    keys = [[month, hour] for month in range(1, 13) for hour in (0, 6, 12, 18)]
    table = (torch.randn(len(keys), channels, height, width) * 0.1).contiguous()
    spec = {"format": "r7-train-climatology-anchor-v1", "kind": "train-only-month-hour-grid-mean-v1",
            "channels": [f"c{index}" for index in range(channels)], "units": ["u"] * channels,
            "source_sha256": "1" * 64, "train_data_identity": "2" * 64,
            "climatology_mean_identity_sha256": "3" * 64,
            "normalization_mean": [0.0] * channels, "normalization_std": [1.0] * channels,
            "latitude": [float(50 - 2 * index) for index in range(height)],
            "longitude": [float(90 + 2 * index) for index in range(width)],
            "bucket_keys": keys, "bucket_counts": [7] * len(keys),
            "training_years": list(range(2000, 2022)), "n_selected_steps": 7 * len(keys),
            "selection": "declared_train_years",
            "table_sha256": anchor_parent.table_digest(table), "table_shape": list(table.shape)}
    return {"spec": spec, "table": table,
            "pins": {"run": "/synthetic/run", "table_sha256": spec["table_sha256"],
                     "spec_sha256": "4" * 64, "table_file_sha256": "5" * 64,
                     "n_buckets": len(keys), "n_selected_steps": spec["n_selected_steps"],
                     "bucket_counts": [7] * len(keys), "source_sha256": "1" * 64,
                     "train_data_identity": "2" * 64, "build_receipt_sha256": "6" * 64}}


def _arm(name, *, feedback, spec=None):
    return driver.arm_protocol(
        spec if spec is not None else support.arm_spec({"dim": 192, "depth": 4},
                                                       _synthetic_payload(), feedback),
        feedback, train_identity="train-identity",
        windows={"input_windows": driver.TRAIN_WINDOWS, "window_sha256": "a" * 64},
        canonical_digest=lambda value: f"digest-{value.get('dim')}-{value.get('depth')}")


def _arms():
    # One payload for both arms, exactly as the driver's cached ``anchor_payload`` does: two
    # independent tables would be a real factor difference, and the guard must see that.
    payload = _synthetic_payload()
    control = _arm(driver.CONTROL, feedback=False, spec=support.arm_spec(
        {"dim": 192, "depth": 4}, payload, False))
    candidate = _arm(driver.CANDIDATE, feedback=True, spec=support.arm_spec(
        {"dim": 192, "depth": 4}, payload, True))
    assert support.require_single_factor(control, candidate) is None
    return control, candidate


def _arm_state(*, skill72=0.1, delta72=-1.0, skill48=0.1, deltas=(-1.0, -1.0, -1.0)):
    return {"arms": {driver.CANDIDATE: {
                "full_t2m_skill": {"6": skill48, "12": skill48, "24": skill48,
                                   "48": skill48, "72": skill72},
                "full_t2m_rmse": {"6": 2.0, "12": 2.1, "24": 2.2, "48": 2.3, "72": 2.4}}},
            "delta_candidate_minus_control": {
                "6": deltas[0], "12": deltas[1], "24": deltas[2], "72": delta72}}


def _exit_seeds(*, skill72=0.1, delta72=-1.0, skill48=0.1, deltas=(-1.0, -1.0, -1.0),
                per_seed=None):
    base = dict(skill72=skill72, delta72=delta72, skill48=skill48, deltas=deltas)
    body = {str(seed): _arm_state(**base) for seed in driver.SEEDS}
    for seed, values in (per_seed or {}).items():
        body[str(seed)] = _arm_state(**{**base, **values})
    return body


def _receipts(*, loss=None):
    losses = dict(support.BRIDGE_UPDATE1_LOSS) if loss is None else loss
    return {str(seed): {driver.CONTROL: {"update1": {"loss": losses[int(seed)],
                                                     "gradient_norm": 1.0}},
                        driver.CANDIDATE: {"update1": {"loss": 9.0, "gradient_norm": 1.0}}}
            for seed in driver.SEEDS}


# --- the matched-compute constants -------------------------------------------------------------

def test_the_compute_is_the_matched_one_and_not_the_closed_rounds():
    assert (driver.UPDATES, driver.BATCH_SIZE) == (16071, 2)
    assert driver.UPDATES * driver.BATCH_SIZE >= driver.TRAIN_WINDOWS
    assert (support.CLOSED_ROUND_UPDATES, support.CLOSED_ROUND_BATCH_SIZE) == (2400, 1)
    assert driver.UPDATES != support.CLOSED_ROUND_UPDATES
    assert driver.BATCH_SIZE != support.CLOSED_ROUND_BATCH_SIZE


def test_the_budget_is_decision_0060s_track_one_constants():
    assert (driver.PER_SEED_SECONDS, driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) \
        == (129600.0, 259200.0, 518400.0)
    # R1 measured 88,362-89,520 s per seed on the same store at the same compute, so the cap
    # must leave room above that measurement rather than sit at it.
    assert driver.PER_SEED_SECONDS > 89520.0


def test_the_peak_estimate_is_the_measured_one_plus_the_table():
    """R1 measured 5,582,618,624 B on this store at batch 2; the FY round's 4 GiB undershot it."""
    assert driver.ESTIMATED_PEAK_BYTES == 5582618624 + 48 * 17 * 65 * 65 * 4
    required = driver.ESTIMATED_PEAK_BYTES + driver.HEADROOM_MARGIN_BYTES + 6 * 2**30
    assert required == driver.ESTIMATED_PEAK_BYTES + 512 * 2**20 + 6 * 2**30


def test_the_frozen_constants_carry_the_factor_and_the_closed_round():
    body = driver.frozen_constants(_synthetic_payload())
    assert body["format"] == driver.FORMAT
    assert body["factor"]["name"] == "climatology-anchor anomaly feedback at matched compute"
    assert body["factor"]["closed_round"]["record"] == support.CLOSED_ROUND_RECORD
    assert body["factor"]["closed_round"]["updates"] == support.CLOSED_ROUND_UPDATES
    assert set(body["factor"]["changed_from_closed_round"]) == {
        "instance", "grid", "supervision region", "data span", "anchor table", "updates",
        "batch_size"}
    assert body["test_read"] is False and body["scientific_claim"] is False
    assert body["test_manifest_never_read"].endswith("test.jsonl")
    assert len(body["limitations"]) >= 8
    assert set(body["decision"]["exits"]) == {"a", "b", "c", "d"}
    assert body["decision"]["text"] == support.DECISION_RULE
    assert body["bridge_e"]["record_id"] == support.BRIDGE_RECORD
    assert body["bridge_e"]["scientific"] is False


def test_the_recipe_declares_the_epoch_floor():
    body = driver.frozen_constants(_synthetic_payload())
    assert body["recipe"]["updates"] == driver.UPDATES
    assert body["recipe"]["batch_size"] == driver.BATCH_SIZE
    assert "32142" in body["recipe"]["epochs_floor"]
    assert body["recipe"]["supervision"].startswith("full_grid")


# --- the one-factor claim ----------------------------------------------------------------------

def test_the_arms_differ_only_in_the_feedback_boolean():
    control, candidate = _arms()
    assert support.difference_keys(control, candidate) == ["anomaly_feedback", "model"]
    assert control["train_manifest"] == candidate["train_manifest"]
    assert control["train_data_identity"] == candidate["train_data_identity"]
    assert control["train_windows"] == candidate["train_windows"]
    assert control["model"]["spec"]["anomaly_feedback"] is False
    assert candidate["model"]["spec"]["anomaly_feedback"] is True


def test_both_specs_hold_the_priors_off_and_share_one_table():
    control, candidate = _arms()
    for spec in (control["model"]["spec"], candidate["model"]["spec"]):
        assert spec["anchor_output_prior"] is False
        assert spec["climatology_anchor_spec"]["table_shape"][0] == 48
        assert len(spec["climatology_anchor_spec"]["bucket_keys"]) == 48
    assert (control["model"]["spec"]["climatology_anchor_spec"]
            == candidate["model"]["spec"]["climatology_anchor_spec"])


def test_two_independent_tables_are_caught_as_a_factor_difference():
    """The driver caches one payload; two arms that derived their own would differ here."""
    first, second = _synthetic_payload(), _synthetic_payload()
    assert first["pins"]["table_sha256"] != second["pins"]["table_sha256"]
    control = _arm(driver.CONTROL, feedback=False,
                   spec=support.arm_spec({"dim": 192, "depth": 4}, first, False))
    candidate = _arm(driver.CANDIDATE, feedback=True,
                     spec=support.arm_spec({"dim": 192, "depth": 4}, second, True))
    with pytest.raises(ValueError, match="climatology_anchor_spec"):
        support.require_single_factor(control, candidate)


def test_the_specs_may_not_differ_in_anything_but_the_feedback():
    control, candidate = _arms()
    drifted = copy.deepcopy(candidate)
    drifted["model"]["spec"]["dropout"] = 0.1
    with pytest.raises(ValueError, match="differ in"):
        support.require_single_factor(control, drifted)
    moved = copy.deepcopy(candidate)
    moved["anomaly_feedback"] = False
    with pytest.raises(ValueError, match="differ in"):
        support.require_single_factor(control, moved)


def test_the_priors_may_not_be_switched_on_in_either_arm():
    control, candidate = _arms()
    for arm, value in ((control, True), (candidate, True)):
        broken = copy.deepcopy(arm)
        broken["model"]["spec"]["anchor_output_prior"] = value
        with pytest.raises(ValueError):
            support.require_single_factor(control, broken)


def test_a_manifest_difference_is_no_longer_the_factor():
    """R1's arms differed by manifest; R2's must not, or the two factors would be confounded."""
    control, candidate = _arms()
    moved = copy.deepcopy(candidate)
    moved["train_manifest"] = "elsewhere/train.jsonl"
    with pytest.raises(ValueError, match="differ in"):
        support.require_single_factor(control, moved)


def test_an_arm_spec_that_already_carries_a_switch_is_refused():
    payload = _synthetic_payload()
    for key in ("climatology_anchor_spec", "anomaly_feedback", "anchor_output_prior",
                "global_mixing_mode", "tendency_input", "static_conditioning"):
        with pytest.raises(RuntimeError, match="closed-architecture switch"):
            support.arm_spec({"dim": 192, "depth": 4, key: True}, payload, False)
    with pytest.raises(RuntimeError, match="dim-192 depth-4"):
        support.arm_spec({"dim": 256, "depth": 4}, payload, False)


def test_arm_name_rejects_non_boolean_feedback():
    assert support.arm_name(False) == driver.CONTROL
    assert support.arm_name(True) == driver.CANDIDATE
    for value in (0, 1, "control", None):
        with pytest.raises(ValueError):
            support.arm_name(value)


def test_the_anchor_blocks_declare_the_shared_table_and_zero_new_parameters():
    payload = _synthetic_payload()
    blocks = {name: support.anchor_block(name, feedback, payload)
              for name, feedback in support.ARMS}
    assert blocks[driver.CONTROL]["anchor_table"]["table_sha256"] \
        == blocks[driver.CANDIDATE]["anchor_table"]["table_sha256"]
    assert all(block["new_params"] == 0 for block in blocks.values())
    assert all(block["anchor_output_prior"] is False for block in blocks.values())
    assert blocks[driver.CONTROL]["feedback"] == "draft"
    assert blocks[driver.CANDIDATE]["feedback"] == "draft - C_valid"
    assert blocks[driver.CANDIDATE]["arm"] == driver.CANDIDATE


# --- the predeclared exits and the bridge ------------------------------------------------------

def test_all_four_exits_true_advances():
    verdict = driver.evaluate_exit(_exit_seeds())
    assert verdict["a"] and verdict["b"] and verdict["c"] and verdict["d"] and verdict["all"]
    assert verdict["verdict"] == "advance-to-independent-review"


@pytest.mark.parametrize("broken", ["a", "b", "c", "d"])
def test_any_false_exit_closes_the_branch(broken):
    per_seed = {"41": {}}
    if broken == "a":
        per_seed["41"] = {"skill72": -0.1}
    elif broken == "b":
        per_seed["41"] = {"delta72": 0.5}
    elif broken == "c":
        per_seed["41"] = {"skill48": -0.1}
    elif broken == "d":
        verdict = driver.evaluate_exit(_exit_seeds(deltas=(0.5, 0.5, 0.5)))
        assert verdict["d"] is False and verdict["all"] is False
        assert verdict["verdict"] == "close-anchor-anomaly-feedback-branch-at-matched-compute"
        return
    verdict = driver.evaluate_exit(_exit_seeds(per_seed=per_seed))
    assert verdict[broken] is False and verdict["all"] is False
    assert verdict["verdict"] == "close-anchor-anomaly-feedback-branch-at-matched-compute"


def test_the_seed_mean_is_never_averaged_away_by_a_single_seed():
    verdict = driver.evaluate_exit(_exit_seeds(deltas=(0.5, 0.5, 0.5)))
    assert verdict["d"] is False
    assert verdict["seed_mean_delta_6_12_24"]["6"] == pytest.approx(0.5)


def test_the_bridge_passes_only_on_a_bit_identical_first_update():
    assert driver.bridge_verdict(_receipts())["passed"] is True
    drifted = dict(support.BRIDGE_UPDATE1_LOSS)
    drifted[42] = drifted[42] + 1e-9
    verdict = driver.bridge_verdict(_receipts(loss=drifted))
    assert verdict["passed"] is False
    assert verdict["seeds"]["42"]["bit_identical_loss"] is False
    assert verdict["scientific"] is False


def test_the_bridge_pins_r1s_registered_treatment_arm_at_this_rounds_compute():
    assert (support.BRIDGE_RECORD, support.BRIDGE_UPDATES, support.BRIDGE_BATCH_SIZE) \
        == ("s3-fy-year-coverage-matched-result", 16071, 2)
    assert sorted(support.BRIDGE_UPDATE1_LOSS) == sorted(driver.SEEDS)
    assert all(isinstance(value, float) for value in support.BRIDGE_UPDATE1_LOSS.values())
    assert set(support.BRIDGE_UPDATE1_GRADIENT_NORM) == set(support.BRIDGE_UPDATE1_LOSS)


def test_the_readings_carry_the_compute_and_the_closed_round():
    source = Path(driver.__file__).read_text(encoding="utf-8")
    body = ast.unparse(next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == "_readings"))

    def records(field, value):
        return re.search(rf"['\"]{field}['\"]\s*:\s*{value}\b", body) is not None

    assert records("updates", "UPDATES"), "the reading must carry the updates it ran at"
    assert records("batch_size", "BATCH_SIZE"), "the reading must carry the batch size it ran at"
    assert records("record", "CLOSED_ROUND_RECORD"), \
        "the reading must name the closed round for side-by-side reading"


# --- the frozen paths and the module split -----------------------------------------------------

def test_the_round_is_pinned_to_the_multi_year_treatment_manifest():
    assert driver.INSTANCE.name == "r7_s3_fy_multiyear_20261023_attempt01"
    assert driver.TRAIN_MANIFEST.parent.name == "manifests_treatment"
    assert driver.VAL_MANIFEST.parent == driver.TRAIN_MANIFEST.parent
    assert driver.TEST_MANIFEST.parent == driver.TRAIN_MANIFEST.parent
    assert driver.TRAIN_WINDOWS == 32142 and driver.VAL_COUNT == 1458
    assert driver.ANCHOR_TABLE_RUN.name == "r7_s3_fy_anchor_table_20261025_attempt01"


def test_the_execution_inventory_is_this_rounds_import_closure():
    """Decision 0067: a named closure, not the closed rounds' 194-entry package glob."""
    from tools.r7_explicit_inventory import import_closure
    files = driver.execution_files()
    assert files == sorted(set(files))
    assert files == import_closure(driver.ENTRY_FILES[0])
    for name in ("scripts/study_r7_s3_anchor_feedback_65x65.py",
                 "scripts/r7_s3_anchor_feedback_support.py",
                 "scripts/r7_s3_anchor_feedback_parent.py",
                 "scripts/r7_s3_anchor_isolation_parent.py"):
        assert name in files
    assert all((ROOT / name).is_file() for name in files)
    assert not [name for name in files if name.startswith(("tools/", "tests/"))]
    assert not [name for name in files if "legacy" in name]
    assert not [name for name in files if name.startswith("data/download")]
    assert not [name for name in files if "r7_v2_comparison" in name], \
        "the archived protocol is a local outputs/ artifact, not repo source"


def test_the_closure_is_strictly_narrower_than_the_old_package_glob():
    """The round must not inherit the wide round's 202-entry list that cost R1 two seeds."""
    files = set(driver.execution_files())
    glob = {str(path.relative_to(ROOT)) for directory in ("training", "model", "data")
            for path in (ROOT / directory).rglob("*.py")}
    assert len(files) < len(glob)
    extra = sorted(glob - files)
    assert extra, "the closure should drop the modules this round never executes"
    assert "data/download/urban_static_sources.py" in extra


def test_adding_an_unrelated_module_does_not_move_this_rounds_inventory():
    """The counter-proof decision 0067 requires of R2-R5: the R1 failure, made harmless."""
    probe = ROOT / "data" / "unrelated_r2_probe.py"
    before = driver.execution_files()
    try:
        probe.write_text("X = 1\n", encoding="utf-8")
        after = driver.execution_files()
    finally:
        probe.unlink(missing_ok=True)
    assert after == before
    assert not probe.exists()


def test_every_runtime_imported_repo_module_is_inside_the_closure():
    """The named-list risk is omission; this is the sys.modules cross-check against it.

    It runs in a subprocess because this suite also imports other rounds' drivers, and a
    global ``sys.modules`` scan would credit them to this round (order-dependent, and it
    would blame this round for a module it never executes).
    """
    import json
    import subprocess
    import sys
    from tools.r7_explicit_inventory import import_closure
    closure = set(import_closure(driver.ENTRY_FILES[0]))
    probe = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "import scripts.study_r7_s3_anchor_feedback_65x65 as driver\n"
        f"root = Path({str(ROOT)!r})\n"
        "found = set()\n"
        "for module in list(sys.modules.values()):\n"
        "    path = getattr(module, '__file__', None)\n"
        "    if not path:\n"
        "        continue\n"
        "    try:\n"
        "        relative = str(Path(path).resolve().relative_to(root))\n"
        "    except ValueError:\n"
        "        continue\n"
        "    if not relative.endswith('.py'):\n"
        "        continue\n"
        "    if relative.startswith(('tools/', 'tests/', '.venv/', 'outputs/', 'audit/')):\n"
        "        continue\n"
        "    if 'site-packages' in relative or not (root / relative).is_file():\n"
        "        continue\n"
        "    found.add(relative)\n"
        "print(json.dumps(sorted(found)))\n"
    )
    finished = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT), timeout=900,
                              capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr[-2000:]
    imported = set(json.loads(finished.stdout.strip().splitlines()[-1]))
    assert imported, "importing the driver must pull in the round's modules"
    assert imported <= closure, sorted(imported - closure)


def test_the_support_module_carries_no_store_or_manifest_read():
    source = Path(support.__file__).read_text(encoding="utf-8")
    assert "outputs/" not in source
    assert "import json" not in source
    assert json.dumps(support.BRIDGE_UPDATE1_LOSS)  # plain JSON-serialisable data


def test_the_decision_rule_names_the_closure_terms():
    text = support.DECISION_RULE
    for phrase in ("72 h full-grid t2m skill > 0", "48 h full-grid t2m skill > 0",
                   "6/12/24 h", "permanently close", "no rename-and-reopen",
                   "never a verdict"):
        assert phrase in text, phrase
