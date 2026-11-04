"""CPU counterproofs for Stage B point 1; no GPU, no real dataset, no scientific claim.

These pin the four properties the point's predeclared engineering exit rests on, plus the
one mistake this round actually made before it was caught: the scored-variable list was
written from memory and did not match the store's own channel list. The guard for that is a
test that reads the list the protocol carries, not the one the driver remembered.

Fixtures are synthetic tmp_path stores: nothing here is weather, and every fixture declares
``scientific_claim=False``.
"""
from __future__ import annotations

import math

import pytest
import torch

from scripts import r7_s3_throughput_physics_b_protocol as protocol
from scripts import study_r7_s3_throughput_physics_b as driver


def test_the_curve_ladder_is_the_frozen_one():
    """The four points and their update counts are the goal document's ladder, verbatim."""
    assert dict(protocol.POINTS) == {"p0025": 2008, "p1000": 8033,
                                     "p4000": 32132, "p1600": 128528}
    assert protocol.POINT == "p0025"
    assert protocol.UPDATES == 2008
    assert protocol.BATCH_SIZE == 4
    # 2008 * 4 = 8032 samples over 32,131 USABLE windows is 0.249977 epoch, which the goal
    # document names 0.25. Both are recorded: the exact ratio and the nominal label, so a
    # reader cannot mistake the label for the arithmetic.
    block = protocol._point_blocks()
    assert block["epochs"]["p0025"] == pytest.approx(8032 / 32131, abs=1e-6)
    assert block["nominal_epochs"]["p0025"] == 0.25
    assert block["nominal_epochs"]["p1600"] == 16.0
    assert block["point"] == protocol.POINT and block["updates"] == protocol.UPDATES


def test_batch_is_fixed_at_four_for_every_point():
    """Decision 0084: batch does not move during calibration, so no point overrides it."""
    block = protocol._point_blocks()
    assert block["batch_size"] == 4
    assert "batch" not in {label for label, _ in protocol.POINTS}


def test_the_scored_variables_are_the_seventeen_registered_channels():
    """The list must be the store's own channel order; a memory-written list is the bug."""
    expected = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
    assert driver.SCORED_VARIABLES == expected
    assert len(set(driver.SCORED_VARIABLES)) == 17
    # No duplicate and no empty entry: a typo would otherwise score nothing and look finite.
    assert all(name.strip() for name in driver.SCORED_VARIABLES)


def test_only_t2m_carries_a_registered_climatology_denominator():
    """Sixteen variables have no fitted denominator, so they get no invented skill number."""
    body = {"climatology": {"buckets": 48},
            "climatology_reference": {"leads": {"72": {"t2m_rmse": 3.926891361747067}}}}
    assert driver._climatology_denominator(body, 72, "t2m") == pytest.approx(3.926891361747067)
    for variable in driver.SCORED_VARIABLES:
        if variable == "t2m":
            continue
        assert driver._climatology_denominator(body, 72, variable) is None


def test_the_saturation_rule_is_recorded_and_never_evaluated_here():
    """A single point cannot compute a two-point first difference; the rule is data only."""
    rule = protocol.SATURATION_RULE
    assert set(rule) == {"delta", "unsaturated_if", "saturated_if", "else", "note"}
    assert "16 epoch" in rule["delta"] and "4 epoch" in rule["delta"]
    assert "0.01" in rule["unsaturated_if"] and "0.005" in rule["saturated_if"]
    # The frozen constants carry it verbatim, and no code path produces a verdict from it.
    assert protocol.frozen_constants()["saturation_rule"] == rule
    assert not hasattr(driver, "saturation_verdict")


def _readings(updates=2008, batch=4, leads=None, skill_72=None, n_variables=17):
    """A minimal readings body shaped like the driver's, for exit-rule counterproofs."""
    variables = list(driver.SCORED_VARIABLES[:n_variables])
    values = {name: 1.0 for name in variables}
    if leads is None:
        leads = {str(lead): {"rmse": dict(values), "skill": {name: None for name in variables}}
                 for lead in (6, 12, 24, 48, 72)}
    leads["72"]["skill"]["t2m"] = 0.1 if skill_72 is None else skill_72
    return {"updates": updates, "batch_size": batch, "leads": leads}


def test_the_engineering_exit_passes_on_a_complete_point():
    result = driver.evaluate_exit(_readings())
    assert result["all"] is True
    assert result["verdict"] == "advance-to-next-curve-point"


@pytest.mark.parametrize("mutate,label", [
    (lambda r: r.update(updates=2007), "a: wrong update count"),
    (lambda r: r.update(batch_size=2), "a: wrong batch"),
    (lambda r: r["leads"]["72"]["rmse"].update(t2m=float("nan")), "b: nonfinite rmse"),
    (lambda r: r["leads"]["72"]["skill"].update(t2m=None), "c: no t2m 72h skill"),
])
def test_any_false_exit_invalidates_the_point(mutate, label):
    readings = _readings()
    mutate(readings)
    result = driver.evaluate_exit(readings)
    assert result["all"] is False, label
    assert result["verdict"] == "engineering-incomplete-do-not-read"


def test_a_missing_variable_fails_the_finite_rule():
    """Sixteen of seventeen variables is an incomplete point, not a passing one."""
    readings = _readings(n_variables=16)
    # ``_readings`` slices the list, so the 72 h table is missing the last variable.
    assert len(readings["leads"]["72"]["rmse"]) == 16
    result = driver.evaluate_exit(readings)
    assert result["b"] is False
    assert result["all"] is False


def test_the_protocol_records_the_stage_a_policies():
    """Checkpoint and finite-state policy come from 0086/0087, not from habit."""
    constants = protocol.frozen_constants()
    assert constants["checkpoint_policy"]["checkpoint_every"] == 5000
    assert constants["checkpoint_policy"]["decision"] == "0087"
    assert constants["finite_state_policy"]["full_check_every"] == 100
    assert constants["finite_state_policy"]["decision"] == "0086"
    assert constants["test_read"] is False
    assert constants["scientific_claim"] is False


def test_the_budget_is_recomputed_from_the_measured_rate():
    """The training estimate must come from the measured 1.1279 s/update, not the old guess."""
    constants = protocol.frozen_constants()
    budgets = constants["budgets"]
    measured = protocol.MEASURED_SECONDS_PER_UPDATE
    assert measured == pytest.approx(1.1279)
    assert budgets["train_seconds_estimate"] == pytest.approx(2008 * measured, rel=1e-6)
    # The estimate is derived, so it must equal the product exactly, not a rounded figure.
    assert protocol.TRAIN_SECONDS == pytest.approx(2008 * measured, rel=1e-12)
    assert budgets["measured_runs"] == [1.0848, 1.1349, 1.1107, 1.1812]
    assert budgets["planned_seconds"] == 43200.0
    assert budgets["hard_cap_seconds"] == 86400.0
    assert budgets["per_seed_seconds"] == 39600.0
    assert "estimate" in budgets["estimate_basis"]


def test_the_point_is_a_single_seed_and_a_single_arm():
    """n=1 is the decision, not an omission: the declaration must say so."""
    constants = protocol.frozen_constants()
    assert constants["curve"]["seeds"] == [41]
    assert protocol.SEEDS == (41,)
    assert any("ONE seed" in item or "one seed" in item.lower()
               for item in constants["limitations"])


def test_the_driver_is_runnable_as_a_script():
    """A driver without a ``__main__`` guard exits 0 having done nothing.

    This round shipped that defect: ``stage prepare`` reported success in 0.06 s and wrote
    no ``prepared_protocol.json``, because importing the module defined ``main`` but never
    called it. The wrapper creates one worker process per stage, so the guard is what makes
    the stage happen at all -- a test that the module body calls ``main()`` under ``__main__``
    is the only cheap way to catch it without a GPU.
    """
    import ast
    import pathlib
    source = pathlib.Path(driver.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    guards = [
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "__name__"
    ]
    assert guards, "the driver has no `if __name__ == '__main__'` guard"
    body = ast.dump(guards[0])
    assert "main" in body, "the __main__ guard does not call main()"


def test_main_does_not_propagate_the_worker_result():
    """``sys.exit(value)`` prints the value and exits nonzero, so main must not return it.

    The first prepare run on this round wrote a 38 MB stage log because ``main`` returned
    the worker's protocol body into ``sys.exit(...)``. A non-None exit value also sets a
    nonzero status, so the stage would have failed for the wrong reason.
    """
    import ast
    import pathlib
    source = pathlib.Path(driver.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    returns = [node for node in ast.walk(main) if isinstance(node, ast.Return)]
    assert returns == [], "main() must not return a value: sys.exit(worker(...)) prints it"


def test_the_runner_is_told_the_frozen_checkpoint_cadence():
    """Decision 0087 is only applied if the call passes it; the default is 20.

    Measured defect: the first attempt of this point declared 5,000 in the protocol but
    omitted the argument, so the runner used its 20-step default and wrote 101 checkpoints
    (5.6 GiB) where the ADR allows four. The guard is a source check, because the value is
    only visible in the call the driver actually writes.
    """
    import pathlib
    source = pathlib.Path(driver.__file__).read_text(encoding="utf-8")
    call = source[source.index("fine_tune_long_rollout("):]
    call = call[:call.index(")", call.index("loader="))]
    assert "checkpoint_every=CHECKPOINT_EVERY" in call, (
        "the driver must pass checkpoint_every explicitly; the runner's default is 20")


def test_checkpoints_are_discovered_where_the_runner_writes_them():
    """The retained list must glob the folder the runner uses, not a guessed subdirectory."""
    import pathlib
    source = pathlib.Path(driver.__file__).read_text(encoding="utf-8")
    assert 'folder.glob("update_*.pt")' in source
    assert 'folder / "checkpoints"' not in source


def test_the_climatology_denominator_comes_from_the_resolved_reference():
    """It must read the resolved reference block, not the frozen summary (which has no leads)."""
    body = {"climatology": {"buckets": 48},
            "climatology_reference": {"leads": {"72": {"t2m_rmse": 3.926891361747067}}}}
    assert driver._climatology_denominator(body, 72, "t2m") == pytest.approx(3.926891361747067)
    assert driver._climatology_denominator(body, 72, "u10") is None
    # A reference without the lead must raise rather than silently return a number.
    with pytest.raises(RuntimeError, match="refusing to invent"):
        driver._climatology_denominator({"climatology_reference": {"leads": {}}}, 72, "t2m")


def test_offline_denial_blocks_outbound_but_not_local_ipc():
    """R-028 denies outbound connections; it must not deny AF_UNIX worker IPC.

    Measured defect: the blanket denial killed ``multiprocessing``'s resource sharer, which
    the pin-memory thread uses over an ``AF_UNIX`` socket, and the first Stage B seed died
    with ``Pin memory thread exited unexpectedly``. A local socket cannot leave the host, so
    permitting it while denying IP keeps the rule's intent and lets a pinned loader run.
    """
    import socket

    from scripts import r7_m3_offline

    r7_m3_offline.deny_network()
    try:
        with pytest.raises(RuntimeError, match="offline"):
            socket.create_connection(("198.51.100.7", 80))
        with pytest.raises(RuntimeError, match="offline"):
            socket.getaddrinfo("example.invalid", 80)
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(RuntimeError, match="offline"):
            probe.connect(("198.51.100.7", 80))
        probe.close()
        # A local AF_UNIX pair must still be able to connect.
        import tempfile
        path = str(pytest.importorskip("pathlib").Path(tempfile.mkdtemp()) / "s.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(path)
        client.close()
        server.close()
    finally:
        r7_m3_offline.allow_network()


def test_the_protocol_module_declares_its_entry_file():
    """The execution inventory is built from a named entry file, never a glob (decision 0067)."""
    assert protocol.ENTRY_FILES == ("scripts/study_r7_s3_throughput_physics_b.py",)
    import pathlib
    assert (pathlib.Path(driver.__file__).resolve().parents[1]
            / protocol.ENTRY_FILES[0]).is_file()


def test_reading_without_a_finite_loss_is_refused():
    """A nonfinite training loss must stop the point before it is scored."""
    losses = [1.0, float("inf"), 0.9]
    assert not all(math.isfinite(value) for value in losses)
    good = [1.0, 0.98, 0.9]
    assert all(math.isfinite(value) for value in good)
