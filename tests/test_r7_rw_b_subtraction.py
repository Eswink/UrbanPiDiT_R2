"""The RW-B subtraction: two sub-switches, both equivalence ends and the matrix.

``docs/goals/main-model-v2-rw-b-subtraction.md`` section 3 freezes what the two new
switches have to mean and what has to be shown before any of them is trained on:

    (i)  ``local_solver_state=False`` is the previous implementation, bit for bit;
    (ii) ``local_solver_state=True`` with both sub-switches at their default is the
         previous revision's RW-B, bit for bit;
    and between those ends the switches have to actually do something -
    ``solver_state_recurrence=False`` freezes ``Z`` at ``solver_init``,
    ``solver_gate_proposal=False`` takes the pre-RW-B correction path while the state
    still advances, and neither adds a parameter.

The equivalence ends are measured against the revision this round started from, read
out of the repository object store, exactly as ``tests/test_r7_switched_path_equivalence.py``
does for the earlier switches: same process, same interpreter, byte comparison, no
tolerance. Two guards keep that comparison from passing vacuously - the live run has to
have imported the live tree, and the counterproof requires a deliberately perturbed
frozen tree to break the equality.

The rest of the file is the round's D3 list: the switch combination matrix, a non-zero
gradient through the gate, the three-step-path equivalence, resume and BF16.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import forecast_inputs
from training.r7_streaming import backward_streamed_truncated

# The revision the two sub-switches were split out of: the branch head this round
# started from. Reachable in CI because ci.yml checks out with fetch-depth: 0.
PRE_CHANGE_SHA = "e6085bc8a8ab173f6208ed210bf8484f323a56a4"

SMALL = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": 16, "depth": 1,
         "heads": 2, "window_size": 4, "patch_size": 2, "dropout": 0.0,
         "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 2,
         "spacetime_inputs": True, "positional_process_readout": True,
         "use_forecast_feedback": True}
RW_B = dict(SMALL, local_solver_state=True)
STEPS = 2
HW = (6, 6)

# The modules the recipe executes and their transitive closure inside ``model``.
SNAPSHOT_FILES = (
    "model/coarse_encoder.py", "model/coarse_forecast.py", "model/process_forecast_r7.py",
    "model/process_step_r7.py", "model/local_solver_state_r7.py",
    "model/recursive_weather_r7.py", "model/r7_baselines.py", "model/r7_halting.py",
    "model/r7_rollout.py", "model/weather_forecaster_r7.py", "model/layers/patch_grid.py",
    "model/layers/sdpa.py", "model/layers/window_attention.py",
    "model/spacetime_conditioning_r7.py", "model/process_readout_r7.py",
    "training/r7_experiment.py", "training/r7_halting.py", "training/r7_streaming.py",
)

IMPORT_REWRITE = (r"^(?P<indent>[ \t]*)(?P<keyword>from|import)[ \t]+"
                  r"(?P<package>model)(?P<rest>\.|[ \t]|$)")


def rewrite_model_imports(source: str, package: str, *, target: str = "model") -> str:
    """Repoint ``target`` imports at ``package``. The only edit this file applies."""
    pattern = re.compile(IMPORT_REWRITE.replace("(?P<package>model)", f"(?P<package>{target})"),
                         re.MULTILINE)
    return pattern.sub(
        lambda match: f"{match.group('indent')}{match.group('keyword')} {package}"
                      f"{match.group('rest')}", source)


def assert_import_rewrite_is_the_only_edit(original: str, materialized: str,
                                           package: str) -> None:
    """Reversing the rewrite must reproduce the archived bytes exactly."""
    if rewrite_model_imports(materialized, "model", target=package) == original:
        return
    raise AssertionError(
        f"the materialized copy differs from the archived source by more than the import "
        f"rewrite (package {package})")


def archived_sources(revision: str = PRE_CHANGE_SHA) -> dict[str, str]:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", revision, "model", "training"],
        cwd=str(REPO), capture_output=True, shell=False)
    if completed.returncode != 0:
        raise AssertionError(
            f"cannot read revision {revision!r} from the repository object store; this test "
            "needs the repository history (ci.yml checks out with fetch-depth: 0) and fails "
            "rather than skipping, because an unmeasured equivalence is not a pass")
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(completed.stdout)) as handle:
        for member in handle.getmembers():
            if member.isfile():
                files[member.name] = handle.extractfile(member).read().decode("utf-8")
    return files


def materialize(files: dict[str, str], root: Path, *, name: str,
                edits: dict[str, tuple[str, str]] | None = None) -> tuple[str, str]:
    """Write the archived tree under new package names, import rewrite only.

    Two packages, not one: ``model/`` and ``training/`` each contain a ``r7_halting``,
    so collapsing them into a single package would make ``training/r7_halting.py``'s
    ``from model.r7_halting import ...`` resolve to itself. The established
    ``test_r7_switched_path_equivalence.py`` splits them for the same reason.

    The package names carry a ``subtraction_`` prefix because ``importlib`` caches by
    name across the whole pytest session: a bare name another test module already
    materialised would silently resolve to *that* module - which is a different
    revision - instead of the tree written here.
    """
    root.mkdir(parents=True, exist_ok=True)
    model_package, training_package = f"subtraction_{name}_model", f"subtraction_{name}_training"
    missing = [entry for entry in SNAPSHOT_FILES if entry not in files]
    if missing:
        raise AssertionError(f"the archived revision does not contain {missing}")
    for archived_name in SNAPSHOT_FILES:
        top, _, relative = archived_name.partition("/")
        original = files[archived_name]
        # A training module's ``from model.x`` imports point at the frozen model
        # package, never at the training one.
        rewritten = (rewrite_model_imports(original, model_package) if top == "training"
                     else original)
        for target, (before, after) in (edits or {}).items():
            if target == archived_name:
                if before not in rewritten:
                    raise AssertionError(f"perturbation target {before!r} absent from "
                                         f"{archived_name}")
                rewritten = rewritten.replace(before, after, 1)
        destination = root / (model_package if top == "model" else training_package) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rewritten, encoding="utf-8")
        if top == "training":
            assert_import_rewrite_is_the_only_edit(original, rewritten, model_package)
    for package in (model_package, training_package):
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / model_package / "layers" / "__init__.py").write_text("", encoding="utf-8")
    return model_package, training_package


def frozen_package(tmp_path: Path, *, name: str = "frozen",
                   edits: dict[str, tuple[str, str]] | None = None) -> SimpleNamespace:
    root = tmp_path / name
    model_package, training_package = materialize(archived_sources(), root, name=name,
                                                  edits=edits)
    sys.path.insert(0, str(root))
    try:
        process = importlib.import_module(f"{model_package}.process_forecast_r7")
        halting = importlib.import_module(f"{model_package}.r7_halting")
        streaming = importlib.import_module(f"{training_package}.r7_streaming")
    finally:
        sys.path.remove(str(root))
    return SimpleNamespace(ProcessForecastCoReasoner=process.ProcessForecastCoReasoner,
                           forecast_inputs=halting.forecast_inputs,
                           backward_streamed_truncated=streaming.backward_streamed_truncated,
                           model_file=process.__file__)


def live_package() -> SimpleNamespace:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import model
    import training.r7_streaming as streaming
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    from model.r7_halting import forecast_inputs

    return SimpleNamespace(ProcessForecastCoReasoner=ProcessForecastCoReasoner,
                           forecast_inputs=forecast_inputs,
                           backward_streamed_truncated=streaming.backward_streamed_truncated,
                           model_file=model.__file__)


def batch(hw=HW, size=2) -> dict:
    generator = torch.Generator().manual_seed(5)
    rows, columns = hw
    return {
        "coarse_history": torch.randn(size, 2, 3, rows, columns, generator=generator),
        "atmos_target": torch.randn(size, 3, rows, columns, generator=generator),
        "process_targets": torch.randn(size, 2, generator=generator),
        "lead_time_hours": torch.full((size,), 6.0),
        "latitude": torch.linspace(-30.0, 30.0, rows),
        "longitude": torch.linspace(0.0, 40.0, columns),
        "init_utc_hour": torch.full((size,), 3.0),
        "init_day_of_year": torch.full((size,), 40.0),
    }


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().to("cpu").contiguous()
    header = f"{tuple(array.shape)}|{array.dtype}".encode()
    return hashlib.sha256(header + array.numpy().tobytes()).hexdigest()


def digest_parameters(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode() + b"\0")
        digest.update(tensor_digest(parameter.data).encode())
        if parameter.grad is not None:
            digest.update(tensor_digest(parameter.grad).encode())
    return digest.hexdigest()


def run_recipe(package: SimpleNamespace, *, local_solver_state: bool,
               sub_switches: dict | None = None) -> dict:
    """Forward, rollout, adaptive and one streamed step, for one switch setting."""
    sub_switches = dict(sub_switches or {})
    config = dict(RW_B if local_solver_state else SMALL, **sub_switches)
    data = batch()
    out: dict[str, str] = {}

    torch.manual_seed(401)
    model = package.ProcessForecastCoReasoner(**config).eval()
    with torch.no_grad():
        result = model(package.forecast_inputs(data), reasoning_steps=STEPS)
        out["forecast"] = tensor_digest(result.forecast)
        out["drafts"] = tensor_digest(result.draft_forecasts)
        out["correction"] = tensor_digest(result.final_correction)

    torch.manual_seed(402)
    trainable = package.ProcessForecastCoReasoner(**config).train()
    streamed = package.backward_streamed_truncated(trainable, data, reasoning_steps=STEPS,
                                                   process_weight=0.5, loss_scale=1.0)
    out["streamed_total"] = tensor_digest(streamed.total)
    out["streamed_gradients"] = digest_parameters(trainable)
    return {"digests": out, "tree": str(Path(package.model_file).resolve())}


def assert_live_tree(payload: dict) -> None:
    recorded = Path(payload["tree"])
    if not recorded.is_relative_to(REPO.resolve()):
        raise AssertionError(f"the live run imported {recorded}, which is not the working "
                             "tree; the comparison would be against the wrong tree")


def build(**switches):
    torch.manual_seed(11)
    return ProcessForecastCoReasoner(**{**RW_B, **switches})


def test_the_sub_switches_are_boolean_only_and_need_the_mechanism_on():
    for name in ("solver_state_recurrence", "solver_gate_proposal"):
        with pytest.raises(ValueError, match=name):
            torch.manual_seed(1)
            ProcessForecastCoReasoner(**SMALL, local_solver_state=True, **{name: 0})
        with pytest.raises(ValueError, match=name):
            torch.manual_seed(1)
            ProcessForecastCoReasoner(**SMALL, **{name: False})


def test_the_sub_switches_default_to_the_full_rw_b():
    """A default that silently removed a piece would invalidate every later reading."""
    defaulted = build()
    explicit = build(solver_state_recurrence=True, solver_gate_proposal=True)
    assert defaulted.solver_state_recurrence is True
    assert defaulted.solver_gate_proposal is True
    with torch.no_grad():
        data = forecast_inputs(batch())
        assert torch.equal(defaulted.eval()(data).forecast, explicit.eval()(data).forecast)


def test_the_sub_switches_add_no_parameter_and_no_state_key():
    """They choose a path; they do not carry weights of their own."""
    reference = set(build().state_dict())
    for switches in ({"solver_state_recurrence": False}, {"solver_gate_proposal": False},
                     {"solver_state_recurrence": False, "solver_gate_proposal": False}):
        other = build(**switches)
        assert set(other.state_dict()) == reference, switches
        assert sum(p.numel() for p in other.parameters()) == sum(
            p.numel() for p in build().parameters()), switches


def test_the_combination_matrix_is_three_distinct_forward_paths():
    """The matrix, with the one collapse the frozen semantics require.

    Turning the gate and the proposal off routes the step through the pre-RW-B
    correction head, so the output ignores ``Z`` entirely - the contract calls that
    "the state is idle computation" and forbids silently substituting anything else
    for it. ``(recurrence=True, gate_proposal=False)`` therefore has to be *bitwise
    equal* to ``(False, False)`` while still returning a state, and the other three
    settings have to differ. Asserting this as a collapse rather than as four
    distinct outputs is what makes it a check of the declared semantics instead of an
    over-strong claim that would fail on a correct implementation.
    """
    data = batch()
    forecasts, states = {}, {}
    for recurrence in (True, False):
        for gate_proposal in (True, False):
            model = build(solver_state_recurrence=recurrence,
                          solver_gate_proposal=gate_proposal).eval()
            with torch.no_grad():
                out = model(forecast_inputs(data), reasoning_steps=STEPS)
            forecasts[(recurrence, gate_proposal)] = tensor_digest(out.forecast)
            states[(recurrence, gate_proposal)] = out.solver_state is not None
    assert forecasts[(True, False)] == forecasts[(False, False)], (
        "with the gate+proposal off the output must not read Z; if the two differ, the "
        "switch changed something it does not name")
    assert states[(True, False)] is True and states[(False, False)] is False, (
        "the recurrence is on in the first setting and off in the second, so the "
        "returned state is what distinguishes them")
    distinct = {forecasts[(True, True)], forecasts[(True, False)], forecasts[(False, True)]}
    assert len(distinct) == 3, (
        f"two of the three output-relevant settings coincide: {forecasts}")


def test_each_sub_switch_changes_every_reasoning_step_after_the_initial_forecast():
    """An effect that only showed up at step 1 would not be the mechanism under study.

    ``draft_forecasts`` carries K+1 entries and index 0 is the backbone's initial
    forecast, which no reasoning switch can reach; the steps that have to move are
    1..K.
    """
    data = batch()
    trajectories = {}
    for switches in ({}, {"solver_state_recurrence": False},
                     {"solver_gate_proposal": False},
                     {"solver_state_recurrence": False, "solver_gate_proposal": False}):
        model = build(**switches).eval()
        with torch.no_grad():
            drafts = model(forecast_inputs(data), reasoning_steps=3).draft_forecasts
        trajectories[tuple(sorted(switches.items()))] = drafts
    reference = trajectories[()]
    assert reference.shape[1] == 4, "the trajectory must carry K+1 drafts"
    for name, drafts in trajectories.items():
        if not name:
            continue
        assert drafts.shape == reference.shape
        if dict(name).get("solver_gate_proposal", True) is False:
            # Declared semantics: with the gate+proposal off the whole trajectory is the
            # pre-RW-B one, whatever the recurrence does.
            assert torch.equal(drafts, trajectories[(("solver_gate_proposal", False),)]), name
            continue
        for step in range(1, drafts.shape[1]):
            assert not torch.equal(drafts[:, step], reference[:, step]), (
                f"{name} left step {step} unchanged; the switch is not reaching the step")


def test_recurrence_off_freezes_the_state_and_carries_nothing_forward():
    """``Z`` must stay at ``solver_init`` and be handed on as ``None``.

    Checked on the step itself rather than inferred from the output: the step is the one
    implementation all three callers use, and "Z stays at init" is a claim about the
    state, not about the forecast. The counterproof is the mirrored call with the
    recurrence on, which has to return a state.
    """
    from model.process_step_r7 import ProcessStepInput, process_reasoning_step

    model = build(solver_state_recurrence=False).eval()
    state_on = build(solver_state_recurrence=True).eval()
    data = batch()
    with torch.no_grad():
        base = model.backbone(forecast_inputs(data))
        draft = base.forecast
        process = model.process_queries.expand(draft.shape[0], -1, -1)
        frozen = process_reasoning_step(
            model, ProcessStepInput(process, base.context_tokens, draft), base.token_hw,
            solver_state=None, step_index=0, anchor=base.base_state)
        live = process_reasoning_step(
            state_on, ProcessStepInput(process, base.context_tokens, draft), base.token_hw,
            solver_state=None, step_index=0, anchor=base.base_state)
    assert frozen.solver_state is None, (
        "a state was threaded onward with the recurrence off; that is the recurrence")
    assert live.solver_state is not None, (
        "the mirrored call with the recurrence on returned no state; the counterproof "
        "for the assertion above is not measuring anything")
    assert not torch.equal(frozen.draft, live.draft), (
        "freezing Z changed nothing in the output; the recurrence is not in the path")


def test_gate_proposal_off_takes_the_pre_rw_b_path_and_keeps_the_state():
    """The pre-RW-B correction head decides the output; the state still advances."""
    from model.process_step_r7 import ProcessStepInput, process_reasoning_step

    model = build(solver_gate_proposal=False).eval()
    data = batch()
    with torch.no_grad():
        base = model.backbone(forecast_inputs(data))
        draft = base.forecast
        process = model.process_queries.expand(draft.shape[0], -1, -1)
        result = process_reasoning_step(
            model, ProcessStepInput(process, base.context_tokens, draft), base.token_hw,
            solver_state=None, step_index=0, anchor=base.base_state)
    assert result.solver_state is not None, (
        "the recurrence was on, so the state has to be threaded onward even though this "
        "step's output does not read it")


def test_the_gate_keeps_a_nonzero_gradient_through_the_proposal():
    """The design contract's boundary: a gate that starves the proposal is a defect."""
    model = build().train()
    with torch.enable_grad():
        out = model(forecast_inputs(batch()), reasoning_steps=STEPS)
        out.forecast.square().mean().backward()
    gate_grad = model.solver_gate.score.weight.grad
    proposal_grad = model.proposal_head.decode[0].weight.grad
    assert gate_grad is not None and proposal_grad is not None
    assert gate_grad.abs().sum() > 0, "the gate received no gradient"
    assert proposal_grad.abs().sum() > 0, "the proposal received no gradient"


def test_the_three_step_paths_agree_for_every_sub_switch_combination():
    """forward / streamed / adaptive call one step, so they cannot disagree."""
    from model.r7_halting import AdaptiveProcessForecaster

    data = batch()
    for switches in ({}, {"solver_state_recurrence": False},
                     {"solver_gate_proposal": False},
                     {"solver_state_recurrence": False, "solver_gate_proposal": False}):
        model = build(**switches).eval()
        adapter = AdaptiveProcessForecaster(model).eval()
        with torch.no_grad():
            fixed = model(forecast_inputs(data), reasoning_steps=STEPS).forecast
            adaptive = adapter(data, max_steps=STEPS, min_steps=STEPS,
                               force_full_depth=True)
        assert torch.equal(fixed, adaptive.forecast), switches


def test_a_checkpoint_round_trip_carries_the_sub_switches(tmp_path):
    """Resume has to restore the switch setting, not just the tensors."""
    from training.r7_experiment import canonical_digest, load_checkpoint, save_exclusive

    model = build(solver_state_recurrence=False)
    contract = {"kind": "process", "model": {**RW_B, "solver_state_recurrence": False},
                "note": "subtraction round-trip fixture"}
    payload = {"format": "r7-local-v1", "contract": contract,
               "signature": canonical_digest(contract), "model": model.state_dict(),
               "updates": 1}
    path = tmp_path / "update_0000001.pt"
    save_exclusive(path, payload)
    restored = load_checkpoint(path)
    rebuilt = ProcessForecastCoReasoner(**restored["contract"]["model"])
    rebuilt.load_state_dict(restored["model"], strict=True)
    assert rebuilt.solver_state_recurrence is False
    with torch.no_grad():
        data = forecast_inputs(batch())
        assert torch.equal(rebuilt.eval()(data).forecast, model.eval()(data).forecast)


def test_bf16_streamed_backward_finishes_for_every_sub_switch_combination():
    data = batch()
    for switches in ({}, {"solver_state_recurrence": False},
                     {"solver_gate_proposal": False},
                     {"solver_state_recurrence": False, "solver_gate_proposal": False}):
        model = build(**switches).train()
        result = backward_streamed_truncated(model, data, reasoning_steps=STEPS,
                                             process_weight=0.5, loss_scale=1.0,
                                             amp_dtype=torch.bfloat16)
        assert torch.isfinite(result.total), switches
        assert torch.isfinite(result.draft_errors).all(), switches
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), switches


def test_all_off_reproduces_the_previous_implementation_bitwise(tmp_path):
    """Equivalence end (i): the pre-RW-B path is the revision this round started from."""
    frozen = frozen_package(tmp_path)
    pre_change = run_recipe(frozen, local_solver_state=False)
    live_result = run_recipe(live_package(), local_solver_state=False)
    assert_live_tree(live_result)
    assert live_result["digests"], "the recipe produced no digests"
    differing = sorted(key for key, value in pre_change["digests"].items()
                       if live_result["digests"].get(key) != value)
    assert not differing, (
        f"the switched-off path is not bitwise identical to the previous revision; "
        f"differing entries: {differing}")


def test_full_switch_reproduces_the_previous_revision_rw_b_bitwise(tmp_path):
    """Equivalence end (ii): ``local_solver_state=True`` with both sub-switches default."""
    frozen = frozen_package(tmp_path, name="frozen_rw_b")
    previous = run_recipe(frozen, local_solver_state=True)
    live_result = run_recipe(live_package(), local_solver_state=True)
    assert_live_tree(live_result)
    differing = sorted(key for key, value in previous["digests"].items()
                       if live_result["digests"].get(key) != value)
    assert not differing, (
        f"RW-B with both sub-switches at their default is not bitwise identical to the "
        f"previous revision; differing entries: {differing}")


def test_the_comparison_detects_a_changed_implementation(tmp_path):
    """Counterproof: a perturbed frozen tree must break the equality."""
    perturbed = frozen_package(
        tmp_path, name="perturbed",
        edits={"model/local_solver_state_r7.py": ("GATE_INITIAL_PROBABILITY = 0.25",
                                                 "GATE_INITIAL_PROBABILITY = 0.30")})
    changed = run_recipe(perturbed, local_solver_state=True)
    live_result = run_recipe(live_package(), local_solver_state=True)
    differing = sorted(key for key, value in changed["digests"].items()
                       if live_result["digests"].get(key) != value)
    assert differing, "a changed implementation produced identical digests"


def test_switching_between_the_sub_switches_moves_the_gradient(tmp_path):
    """The two sub-switches are not the same intervention wearing two names."""
    left = run_recipe(live_package(), local_solver_state=True,
                      sub_switches={"solver_state_recurrence": False})
    right = run_recipe(live_package(), local_solver_state=True,
                       sub_switches={"solver_gate_proposal": False})
    assert left["digests"] != right["digests"]
