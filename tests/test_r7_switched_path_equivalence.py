"""D4: the switches must default to the previous implementation, bit for bit.

The M1/RW-A iteration adds two switches (``spacetime_inputs`` on the backbone,
``positional_process_readout`` on the process model). Their default is *off*, and
"off" has to mean the implementation that existed before the change - not "a
numerically similar path that happens to agree to within rounding". The only
artifact that can support that claim is the previous implementation itself, so
this test runs one fixed recipe (`run_digests`) twice in the *same process*:

1. against the pre-change revision, read out of the repository object store
   (``git archive``) into a temporary directory and imported there under the
   package names ``pre_change_model`` / ``pre_change_training``;
2. against the live working tree.

Same process, same interpreter, same threading configuration: the comparison is
a bitwise comparison of raw little-endian float bytes, and no tolerance is
applied anywhere. Because both sides run in one process, the frozen copy has to
be importable under different package names, which is the *only* edit applied to
it: every ``from model.`` / ``import model`` line is rewritten to the frozen
package name, and `assert_import_rewrite_is_the_only_edit` proves that the
rewrite reverses back to the bytes git handed us. Nothing else in the recorded
sources is touched, and each frozen file's SHA-256 is recorded and compared
against the archived bytes.

Two guards keep the comparison from passing vacuously:

- both sides are required to have imported the tree this test asked for, so a
  comparison of one tree with itself is reported as such;
- `test_the_comparison_detects_a_changed_implementation` runs the same recipe
  against a deliberately perturbed copy of the frozen tree and requires the
  digests to differ, so a recipe that hashed nothing cannot look like a pass.

The file is also runnable as a script (``python
tests/test_r7_switched_path_equivalence.py``), which prints both sides' digests
and the verdict - that is how the evidence document records them.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

# The revision the switches must reproduce: the last commit before this
# iteration's edits. Reachable in CI because ci.yml checks out with fetch-depth: 0.
PRE_CHANGE_SHA = "93d89aa8d37e157971261d1d6141f4027eea7575"

BASE_CONFIG = {"in_channels": 3, "out_channels": 3, "history_steps": 2,
               "architecture": "window", "dim": 16, "depth": 2, "heads": 2,
               "window_size": 2, "patch_size": 2, "dropout": 0.0}
GENERIC_CONFIG = dict(BASE_CONFIG, latent_tokens=4, default_reasoning_steps=2)
PROCESS_CONFIG = dict(BASE_CONFIG, anchored_processes=2, free_processes=2,
                      use_forecast_feedback=True, default_reasoning_steps=2)
REASONING_STEPS = 2
BATCH_SIZE = 2
HW = (4, 6)
CHANNELS = 3

# The modules the recipe actually executes, and their transitive closure inside
# ``model``/``training``. Listing them keeps the snapshot to the implementation
# under comparison: the Lightning lit-modules, for instance, are not executed here
# and are not materialized.
SNAPSHOT_FILES = (
    "model/coarse_encoder.py", "model/coarse_forecast.py", "model/process_forecast_r7.py",
    "model/recursive_weather_r7.py", "model/r7_baselines.py", "model/r7_halting.py",
    "model/r7_rollout.py", "model/weather_forecaster_r7.py", "model/layers/patch_grid.py",
    "model/layers/sdpa.py", "model/layers/window_attention.py",
    "training/r7_experiment.py", "training/r7_halting.py", "training/r7_streaming.py",
)

# ``from model.x import y`` / ``import model.x`` at any indentation, including the
# function-local imports training/r7_experiment.py uses.
IMPORT_REWRITE = r"^(?P<indent>[ \t]*)(?P<keyword>from|import)[ \t]+{package}(?P<rest>\.|[ \t]|$)"


def rewrite_model_imports(source: str, package: str, *, target: str = "model") -> str:
    """Repoint ``target`` imports at ``package``. The only edit applied at all."""
    pattern = re.compile(IMPORT_REWRITE.format(package=re.escape(target)), re.MULTILINE)
    return pattern.sub(
        lambda match: f"{match.group('indent')}{match.group('keyword')} {package}"
                      f"{match.group('rest')}", source)


def assert_import_rewrite_is_the_only_edit(original: str, materialized: str,
                                           package: str) -> None:
    """Reversing the rewrite must reproduce the archived bytes exactly."""
    reversed_text = rewrite_model_imports(materialized, "model", target=package)
    if reversed_text == original:
        return
    original_lines = original.splitlines()
    new_lines = materialized.splitlines()
    differing = [index for index, (before, after) in enumerate(zip(original_lines, new_lines))
                 if before != after]
    raise AssertionError(
        f"the materialized copy of a frozen file differs from the archived source by "
        f"more than the import rewrite (package {package}); first differing line(s): "
        f"{[original_lines[i] for i in differing[:3]]} vs "
        f"{[new_lines[i] for i in differing[:3]]}")


def tensor_digest(tensor: torch.Tensor) -> str:
    """SHA-256 over shape, dtype and the exact float bytes of a tensor."""
    array = tensor.detach().to("cpu").contiguous()
    if array.dtype == torch.bfloat16:
        raise TypeError("bfloat16 has no exact float32 view; digest it as float32 instead")
    header = f"{tuple(array.shape)}|{array.dtype}".encode()
    return hashlib.sha256(header + array.numpy().tobytes()).hexdigest()


def digest_parameters(model: torch.nn.Module) -> str:
    """One digest over every parameter and its gradient, in name order."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode() + b"\0")
        digest.update(tensor_digest(parameter.data).encode())
        if parameter.grad is not None:
            digest.update(tensor_digest(parameter.grad).encode())
    return digest.hexdigest()


def fixed_batch() -> dict[str, torch.Tensor]:
    """The whole input: seeded generators only, no dataset, store or network."""
    generator = torch.Generator().manual_seed(20240928)
    return {
        "coarse_history": torch.randn(BATCH_SIZE, 2, CHANNELS, *HW, generator=generator),
        "atmos_target": torch.randn(BATCH_SIZE, CHANNELS, *HW, generator=generator),
        "atmos_baseline": torch.randn(BATCH_SIZE, CHANNELS, *HW, generator=generator),
        "process_targets": torch.randn(BATCH_SIZE, 2, generator=generator),
        "lead_time_hours": torch.full((BATCH_SIZE,), 6.0),
    }


def run_digests(package: SimpleNamespace) -> dict:
    """The fixed recipe. ``package`` supplies the implementation under test."""
    out: dict[str, str] = {}
    batch = fixed_batch()

    for kind, config in (("native", BASE_CONFIG), ("generic", GENERIC_CONFIG),
                         ("process", PROCESS_CONFIG)):
        package.seed_everything(101)
        built = package.make_model(kind, dict(config)).eval()
        with torch.no_grad():
            if kind == "native":
                result = built(batch)
                out[f"{kind}.tendency"] = tensor_digest(result.tendency)
            else:
                result = built(batch, reasoning_steps=REASONING_STEPS)
                out[f"{kind}.draft_forecasts"] = tensor_digest(result.draft_forecasts)
                out[f"{kind}.final_correction"] = tensor_digest(result.final_correction)
                state = (result.process_state
                         if isinstance(built, package.ProcessForecastCoReasoner)
                         else result.latent_state)
                out[f"{kind}.state"] = tensor_digest(state)
            out[f"{kind}.forecast"] = tensor_digest(result.forecast)
            out[f"{kind}.context_tokens"] = tensor_digest(result.context_tokens)

    # The rollout: the one path whose lead convention this iteration touches.
    package.seed_everything(102)
    rollout_model = package.make_model("process", dict(PROCESS_CONFIG)).eval()
    rollout = package.autoregressive_rollout(
        rollout_model, {"coarse_history": batch["coarse_history"],
                        "lead_time_hours": batch["lead_time_hours"]},
        lead_hours=(6, 12, 24), step_hours=6, history_interval_hours=6,
        inference_kwargs={"reasoning_steps": REASONING_STEPS})
    out["rollout.forecasts"] = tensor_digest(rollout.forecasts)
    out["rollout.cumulative_reasoning_steps"] = tensor_digest(
        rollout.cumulative_reasoning_steps)

    # Adaptive inference at full depth (the controller is never queried).
    package.seed_everything(103)
    adapter = package.AdaptiveProcessForecaster(
        package.make_model("process", dict(PROCESS_CONFIG))).eval()
    with torch.no_grad():
        adaptive = adapter(batch, max_steps=REASONING_STEPS, min_steps=REASONING_STEPS,
                           force_full_depth=True)
    out["adaptive.forecast"] = tensor_digest(adaptive.forecast)
    out["adaptive.process_predictions"] = tensor_digest(adaptive.process_predictions)

    # One streamed truncated-BPTT step: losses and every parameter gradient, so a
    # training-path change cannot hide behind a matching loss.
    package.seed_everything(104)
    trained = package.make_model("process", dict(PROCESS_CONFIG)).train()
    streamed = package.backward_streamed_truncated(
        trained, batch, reasoning_steps=REASONING_STEPS, process_weight=0.5, loss_scale=1.0)
    out["streamed.total"] = tensor_digest(streamed.total)
    out["streamed.forecast"] = tensor_digest(streamed.forecast)
    out["streamed.process"] = tensor_digest(streamed.process)
    out["streamed.draft_errors"] = tensor_digest(streamed.draft_errors)
    out["streamed.final_forecast"] = tensor_digest(streamed.final_forecast)
    out["streamed.gradients"] = digest_parameters(trained)

    return {"digests": out,
            "tree": {"model": str(Path(package.model_file).resolve()),
                     "training": str(Path(package.training_file).resolve())},
            "environment": {"python": platform.python_version(),
                            "torch": str(torch.__version__),
                            "torch_cuda": torch.version.cuda}}


def live_package() -> SimpleNamespace:
    """The implementation under test: the live ``model``/``training`` packages."""
    if str(REPO) not in sys.path:  # only needed when run as a script
        sys.path.insert(0, str(REPO))
    import model
    import training
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    from model.r7_halting import AdaptiveProcessForecaster
    from model.r7_rollout import autoregressive_rollout
    from training.r7_experiment import make_model, seed_everything
    from training.r7_streaming import backward_streamed_truncated

    return SimpleNamespace(
        make_model=make_model, seed_everything=seed_everything,
        autoregressive_rollout=autoregressive_rollout,
        AdaptiveProcessForecaster=AdaptiveProcessForecaster,
        ProcessForecastCoReasoner=ProcessForecastCoReasoner,
        backward_streamed_truncated=backward_streamed_truncated,
        model_file=model.__file__, training_file=training.__file__)


def archived_sources(revision: str = PRE_CHANGE_SHA) -> dict[str, bytes]:
    """``model/`` and ``training/`` of one revision, straight out of git."""
    completed = subprocess.run(
        ["git", "archive", "--format=tar", revision, "model", "training"],
        cwd=str(REPO), capture_output=True, shell=False)
    if completed.returncode != 0:
        raise AssertionError(
            f"cannot read revision {revision!r} from the repository object store; this "
            "test needs the repository history (ci.yml checks out with fetch-depth: 0) "
            "and fails rather than skipping, because an unmeasured equivalence is not a "
            "pass")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(completed.stdout)) as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            source = handle.extractfile(member)
            if source is None:
                raise AssertionError(f"archived member {member.name!r} has no content")
            files[member.name] = source.read()
    return files


def materialize(files: dict[str, bytes], root: Path, *, model_package: str,
                training_package: str, edits: dict[str, tuple[str, str]] | None = None
                ) -> SimpleNamespace:
    """Write the archived tree under new package names and import it.

    ``edits`` is only used by the counterproof, which needs a deliberately
    perturbed implementation. Production use passes nothing and every frozen file
    is written byte-identical apart from the import rewrite.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = {"model": model_package, "training": training_package}
    missing = [name for name in SNAPSHOT_FILES if name not in files]
    if missing:
        raise AssertionError(f"the archived revision does not contain {missing}")
    for archived_name in SNAPSHOT_FILES:
        payload = files[archived_name]
        top, _, relative = archived_name.partition("/")
        original = payload.decode("utf-8")
        # A training module's ``from model.x`` imports point at the frozen *model*
        # package, never at the training one.
        rewritten = (rewrite_model_imports(original, model_package) if top == "training"
                     else original)
        for target, (before, after) in (edits or {}).items():
            if target == archived_name:
                if before not in rewritten:
                    raise AssertionError(f"perturbation target {before!r} absent from "
                                         f"{archived_name}")
                rewritten = rewritten.replace(before, after, 1)
        destination = root / names[top] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rewritten, encoding="utf-8")
        if top == "training":
            assert_import_rewrite_is_the_only_edit(original, rewritten, model_package)
    # Package markers. The archived ``model/__init__.py`` is a re-export list that
    # pulls in modules this comparison never executes (the urban backbone, which
    # needs Lightning), so the snapshot imports its modules explicitly instead; the
    # archived ``model/layers/__init__.py`` is materialized as it is.
    for package in names.values():
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    return import_frozen(root, model_package=model_package, training_package=training_package)


def import_frozen(root: Path, *, model_package: str, training_package: str) -> SimpleNamespace:
    """Import a materialized tree without letting it shadow the live packages."""
    sys.path.insert(0, str(root))
    try:
        model_module = importlib.import_module(f"{model_package}.weather_forecaster_r7")
        rollout_module = importlib.import_module(f"{model_package}.r7_rollout")
        halting_module = importlib.import_module(f"{model_package}.r7_halting")
        process_module = importlib.import_module(f"{model_package}.process_forecast_r7")
        experiment = importlib.import_module(f"{training_package}.r7_experiment")
        streaming = importlib.import_module(f"{training_package}.r7_streaming")
    finally:
        sys.path.remove(str(root))
    return SimpleNamespace(
        make_model=experiment.make_model, seed_everything=experiment.seed_everything,
        autoregressive_rollout=rollout_module.autoregressive_rollout,
        AdaptiveProcessForecaster=halting_module.AdaptiveProcessForecaster,
        ProcessForecastCoReasoner=process_module.ProcessForecastCoReasoner,
        backward_streamed_truncated=streaming.backward_streamed_truncated,
        model_file=model_module.__file__, training_file=experiment.__file__)


def pre_change_package(tmp_path: Path, *, name: str = "pre_change",
                       edits: dict[str, tuple[str, str]] | None = None) -> SimpleNamespace:
    root = tmp_path / name
    return materialize(archived_sources(), root,
                       model_package=f"{name}_model", training_package=f"{name}_training",
                       edits=edits)


def assert_tree(payload: dict, root: Path, label: str) -> None:
    """The run must have imported the tree it was asked to, not the other one."""
    root = root.resolve()
    for package, recorded in sorted(payload["tree"].items()):
        if not Path(recorded).is_relative_to(root):
            raise AssertionError(f"the {label} run imported {package} from {recorded}, "
                                 f"which is not under {root}; the comparison would be "
                                 "against itself rather than against the pre-change code")


def test_switches_off_reproduces_the_pre_change_implementation_bitwise(tmp_path):
    """The headline equivalence: switched off == the revision before this work."""
    frozen = pre_change_package(tmp_path)
    pre_change = run_digests(frozen)
    assert_tree(pre_change, tmp_path / "pre_change", "pre-change")
    live = run_digests(live_package())
    assert_tree(live, REPO, "live")
    assert pre_change["digests"], "the recipe produced no digests"
    assert live["digests"] == pre_change["digests"], (
        "the switched-off path is not bitwise identical to the pre-change "
        "implementation; differing entries: "
        f"{sorted(k for k, v in pre_change['digests'].items() if live['digests'].get(k) != v)}")


def test_the_comparison_detects_a_changed_implementation(tmp_path):
    """Counterproof: perturbing the frozen tree must break the equality.

    Without this, a recipe that hashed nothing - or a comparison that silently ran
    one tree twice - would look exactly like a pass.
    """
    perturbed = pre_change_package(
        tmp_path, name="perturbed",
        edits={"model/coarse_forecast.py": ("hidden=hidden or max(32,dim//2)",
                                           "hidden=hidden or max(64,dim//2)")})
    changed = run_digests(perturbed)
    assert_tree(changed, tmp_path / "perturbed", "perturbed")
    live = run_digests(live_package())
    differing = sorted(key for key, value in changed["digests"].items()
                       if live["digests"].get(key) != value)
    assert differing, "a changed implementation produced identical digests"
    # The perturbation changes the lead embedding every arm uses, so more than one
    # digest has to move.
    assert {"native.forecast", "process.forecast"} <= set(differing), differing


def test_the_import_rewrite_cannot_hide_another_edit(tmp_path):
    """The only edit applied to the frozen sources is the import rewrite."""
    files = archived_sources()
    pre_change_package(tmp_path)
    for archived_name in SNAPSHOT_FILES:
        top, _, relative = archived_name.partition("/")
        if top != "training":
            continue
        original = files[archived_name].decode("utf-8")
        materialized = (tmp_path / "pre_change" / "pre_change_training"
                        / relative).read_text(encoding="utf-8")
        assert_import_rewrite_is_the_only_edit(original, materialized, "pre_change_model")
        if "from model" in original or "import model" in original:
            assert "pre_change_model" in materialized, archived_name


def test_the_switches_default_to_off():
    """A default that silently turned the pathway on would invalidate the above."""
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    from model.weather_forecaster_r7 import NativeAtmosForecaster

    kwargs = {name: value for name, value in BASE_CONFIG.items() if name != "architecture"}
    for model, extra in ((NativeAtmosForecaster, {}),
                         (GenericRecursiveWeatherForecaster, {"latent_tokens": 2}),
                         (ProcessForecastCoReasoner, {"anchored_processes": 1,
                                                      "free_processes": 1})):
        built = model(**kwargs, **extra)
        assert built.spacetime_inputs is False, model.__name__
        assert not any(name == "spacetime" for name, _ in built.named_modules())
    reasoner = ProcessForecastCoReasoner(**kwargs, anchored_processes=1, free_processes=1)
    assert reasoner.positional_process_readout is False
    assert not any(name == "process_reader" for name, _ in reasoner.named_modules())


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        frozen = pre_change_package(Path(scratch))
        pre_change = run_digests(frozen)
        live = run_digests(live_package())
    print(json.dumps({
        "pre_change_revision": PRE_CHANGE_SHA,
        "pre_change_tree": pre_change["tree"]["model"],
        "live_tree": live["tree"]["model"],
        "digest_count": len(live["digests"]),
        "bitwise_identical": live["digests"] == pre_change["digests"],
        "pre_change_digests": pre_change["digests"],
        "live_digests": live["digests"],
    }, indent=2, sort_keys=True))
    return 0 if live["digests"] == pre_change["digests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
