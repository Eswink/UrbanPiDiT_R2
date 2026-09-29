"""D1: 0 GPU-h geometry of the archived RW-B checkpoints, one component removed at a time.

The bounded round registered ``RW-B - RW-A`` as negative at t2m 48/72 h and left one
mechanism question open: the RW-B step-1 correction is about 1.8x the RW-A one and its
error/update cosine turns positive by step 2-3 (``docs/R7_72_RW_B_PILOT.md`` section 5).
Four arms cannot say which of RW-B's three pieces produces that geometry, so this probe
removes one piece at a time from the *trained* checkpoints and re-reads the same three
quantities the round already reported - step-1 correction magnitude, the sign of the
per-step error/update cosine, and the step-3 worsening fraction.

Three interventions, none of which edits ``model/``:

- **recurrence removed** - the checkpoint's own ``solver_cell`` is replaced by a
  registered identity module, so ``Z`` stays at ``solver_init`` on every step while the
  anchored proposal and the per-position gate are still applied. This is the definition
  the subtraction round freezes for its ``solver_state_recurrence=False`` switch.
- **gate+proposal removed** - the checkpoint's shared tensors are loaded into the same
  class built with ``local_solver_state=False``, so the step takes the pre-RW-B
  correction path with the checkpoint's own shared weights.
- **role markers removed** - the same, with ``source_role_markers=False``.

All three are compositions of the frozen revision, not a second implementation of the
step. That the live switches produce the same configurations is pinned separately by
``tests/test_r7_rw_b_subtraction.py``.

**Read-only, and no new judgement.** The routine re-reads the archived
``seed_result.json`` rows beside its own, opens its output with ``'x'``, refuses an
existing path, scores the validation split only, hashes every archived checkpoint, and
introduces no threshold: the readings are the three the round already defined.
``--code-root`` selects the revision that trained the checkpoints, because
``training.r7_experiment.load_checkpoint`` refuses a checkpoint whose recorded
``model_code_sha256`` is not the running implementation's - the guard is obeyed, never
bypassed.

One confound is measured rather than left implicit: an arm trained with
``local_solver_state=True`` never routes its output through ``correction_head``, so the
gate+proposal-removed row hands that arm's correction head back at its seeded value.
The probe reports whether training moved it, and that measurement is why the round's
D4 arm retrains from scratch with the switch off instead of reusing this row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SEEDS = (41, 42)
REFERENCE_ARM = "process_spacetime_rwa"
FOCUS_ARM = "process_local_solver"
SOLVER_TENSOR_PREFIXES = ("solver_init", "solver_cell.", "solver_gate.", "proposal_head.")
ROLE_TENSOR_PREFIXES = ("role_context", "role_draft")
CELL_CALL_KEYWORDS = ("context", "draft_tokens", "read", "step_index", "token_hw")
ABLATIONS = (
    ("recurrence_removed", "solver_state_recurrence=False: Z frozen at solver_init"),
    ("gate_proposal_removed", "local_solver_state=False: the pre-RW-B correction path"),
    ("role_markers_removed", "source_role_markers=False: unmarked recurrent key"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def import_code_root(code_root: Path) -> dict:
    """Import the replay revision and prove it is the one that was asked for.

    Same discipline as ``scripts/study_r7_e0_correction_replay.py``: every module the
    probe executes has to resolve inside ``--code-root``, because a shadowing import
    would measure a different revision than the one that trained the checkpoints.
    """
    code_root = Path(code_root).resolve()
    if not (code_root / "training" / "r7_correction_diagnostic.py").is_file():
        raise FileNotFoundError(f"no R7 code under --code-root {code_root}")
    sys.path.insert(0, str(code_root))
    import torch
    from data.r7_evaluation import ZarrRolloutDataset
    from torch.utils.data import default_collate
    from training.r7_correction_diagnostic import collect_correction_terms
    from training.r7_experiment import (load_checkpoint, make_model, model_code_digest,
                                        seed_everything)

    for name in ("data.r7_evaluation", "training.r7_correction_diagnostic",
                 "training.r7_experiment", "model.process_forecast_r7",
                 "model.process_step_r7", "model.local_solver_state_r7"):
        origin = Path(sys.modules[name].__file__).resolve()
        if code_root not in origin.parents:
            raise RuntimeError(
                f"{name} resolved to {origin}, outside --code-root {code_root}; "
                "a shadowing import would replay the checkpoints with the wrong revision")
    return {"dataset": ZarrRolloutDataset, "collate": default_collate,
            "correction_terms": collect_correction_terms, "load": load_checkpoint,
            "make_model": make_model, "digest": model_code_digest,
            "seed_everything": seed_everything, "torch": torch}


def frozen_cell_module(code: dict):
    """A registered ``nn.Module`` standing in for ``solver_cell``: ``Z`` never advances.

    A real module rather than a plain callable on purpose - assigning a non-module over
    a registered submodule leaves the old one in ``_modules``, so the ablation's own
    bookkeeping would disagree with the model's. The step calls the cell with a fixed
    keyword set; asserting that set means a drifted call signature stops the probe
    instead of silently accepting the new one.
    """
    nn = code["torch"].nn
    expected = set(CELL_CALL_KEYWORDS)

    class FrozenCell(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, state, **kwargs):
            if set(kwargs) != expected:
                raise ValueError(
                    f"the frozen-cell wrapper saw call keywords {sorted(kwargs)}, "
                    f"expected {sorted(expected)}; the step signature moved and this "
                    "ablation would no longer be an identity on the same path")
            self.calls += 1
            return state

    return FrozenCell()


def geometry(code: dict, model, batch: dict, channels: tuple, max_steps: int) -> dict:
    """The round's three readings per step, over the declared validation windows.

    ``aggregate`` repeats the exact reductions the archived round used, so the two sets
    of numbers are directly comparable; ``per_variable`` repeats the reduction
    ``training.r7_correction_diagnostic`` writes into its summary rows.
    """
    torch = code["torch"]
    if any(module.training for module in model.modules()):
        raise ValueError("the geometry probe requires eval mode")
    terms = code["correction_terms"](model, batch, max_steps=max_steps)
    steps = []
    for index, term in enumerate(terms):
        energy = term["update_energy"]
        cosine = term["error_update_cosine"]
        defined = term["error_cosine_defined"]
        steps.append({
            "step": index + 1,
            "aggregate": {
                "update_norm_mean": float(energy.mean() ** 0.5),
                "update_energy_mean": float(energy.mean()),
                "error_update_cosine_mean": (
                    float(cosine[defined].mean()) if bool(defined.any()) else None),
                "cosine_defined_fraction": float(defined.float().mean()),
                "worsening_fraction": float(term["worsening"].float().mean()),
                "mse_before_mean": float(term["mse_before"].mean()),
                "mse_after_mean": float(term["mse_after"].mean()),
                "algebra_residual_max": float(term["algebra_residual"].abs().max()),
            },
            "per_variable": {
                name: {
                    "update_norm_mean": float(energy[:, column].mean() ** 0.5),
                    "error_update_cosine_mean": (
                        float(cosine[:, column][defined[:, column]].mean())
                        if bool(defined[:, column].any()) else None),
                    "worsening_fraction": float(
                        term["worsening"][:, column].float().mean()),
                }
                for column, name in enumerate(channels)
            },
        })
    if not steps:
        raise ValueError("the geometry probe produced no steps")
    return {"steps": steps, "windows": int(terms[0]["update_energy"].shape[0]),
            "channels": list(channels)}


def val_batch(code: dict, store: Path, windows: int, lead_hours: int):
    """The first ``windows`` validation windows, as the archived round read them.

    The rollout reader returns one target stack per declared lead; the geometry wants a
    single ``[B,C,H,W]`` field, so the single lead is taken out of its axis. An
    unexpected depth is an error rather than something to reshape.
    """
    torch = code["torch"]
    dataset = code["dataset"](store, split="val", lead_hours=(lead_hours,),
                              history_steps=2, step_hours=6)
    count = min(windows, len(dataset))
    batch = code["collate"]([dataset[index] for index in range(count)])
    targets = batch["rollout_targets"]
    if targets.ndim != 5 or targets.shape[1] != 1:
        raise ValueError(f"the geometry probe expects one lead, got {tuple(targets.shape)}")
    batch = dict(batch, atmos_target=targets[:, 0])
    batch = {key: value.to("cpu") if torch.is_tensor(value) else value
             for key, value in batch.items()}
    return batch, dataset.names, {"split": "val", "windows": count,
                                  "lead_hours": lead_hours, "available": len(dataset)}


def load_checkpoint_arm(code: dict, checkpoint: Path):
    """Load one archived checkpoint; the digest guard inside ``load`` is obeyed."""
    saved = code["load"](checkpoint)
    if saved["contract"].get("kind") != "process":
        raise ValueError(f"{checkpoint} is not a process checkpoint")
    config = dict(saved["contract"]["model"])
    model = code["make_model"]("process", config)
    model.load_state_dict(saved["model"], strict=True)
    return saved, config, model


def retargeted_twin(code: dict, saved: dict, config: dict, *, removed_prefixes: tuple):
    """The checkpoint's shared weights in a class built with one switch turned off.

    The tensors the switch owns are the checkpoint's keys absent from the retargeted
    class, and they have to be exactly the ones named: a silent partial load, or a
    switched-off class that still carries the tensors, would make the "same checkpoint,
    one piece removed" claim false. The load itself must be complete - nothing missing
    and nothing unexpected - so the twin is the checkpoint, not a subset of it.
    """
    torch = code["torch"]
    twin = code["make_model"]("process", config)
    target = twin.state_dict()
    removed = sorted(set(saved["model"]) - set(target))
    if not removed or any(not name.startswith(removed_prefixes) for name in removed):
        raise RuntimeError(
            f"the retargeted twin removed {removed}; the removed tensors must be exactly "
            f"the ones the switch owns ({removed_prefixes})")
    source = {name: tensor for name, tensor in saved["model"].items() if name in target}
    result = twin.load_state_dict(source, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"the retargeted twin did not load cleanly: missing={sorted(result.missing_keys)}, "
            f"unexpected={sorted(result.unexpected_keys)}")
    loaded = twin.state_dict()
    for name, tensor in source.items():
        if not torch.equal(loaded[name], tensor):
            raise RuntimeError(f"shared tensor {name} is not bitwise equal after transfer")
    return twin, removed


def measure_ablations(code: dict, saved: dict, config: dict, checkpoint: Path, batch: dict,
                      channels: tuple, seed: int, max_steps: int) -> dict:
    """All three removals on one checkpoint, plus what training moved."""
    torch = code["torch"]
    readings, notes = {}, {}

    code["seed_everything"](seed)
    initial = code["make_model"]("process", config).state_dict()
    moved = sorted(name for name, tensor in initial.items()
                   if name in saved["model"] and not torch.equal(saved["model"][name], tensor))
    notes["tensors_moved_by_training"] = moved
    notes["correction_head_moved_by_training"] = any(
        name.startswith("correction_head") for name in moved)

    # (b) the recurrence: the cell is replaced by a registered identity, so Z never
    # advances and the proposal and gate still read whatever solver_init expands to.
    _, _, model = load_checkpoint_arm(code, checkpoint)
    frozen = frozen_cell_module(code)
    original = model.solver_cell
    model.solver_cell = frozen
    model = model.eval()
    readings["recurrence_removed"] = geometry(code, model, batch, channels, max_steps)
    notes["recurrence_removed_frozen_cell_calls"] = frozen.calls
    model.solver_cell = original
    del model, frozen

    # (a) the gate and the anchored proposal: the pre-RW-B correction path with the
    # checkpoint's own shared weights.
    config_gate_off = dict(config, local_solver_state=False)
    twin_gate, missing_gate = retargeted_twin(
        code, saved, config_gate_off, removed_prefixes=SOLVER_TENSOR_PREFIXES)
    readings["gate_proposal_removed"] = geometry(code, twin_gate.eval(), batch, channels,
                                                 max_steps)
    notes["gate_proposal_removed_removed_tensors"] = missing_gate
    del twin_gate
    # (c) the role markers are their own switch, and this round does not re-test them: the
    # bounded round already registered that contrast. The ablation runs only where the
    # checkpoint carries the tensors at all, so a marker-free arm is recorded as such
    # rather than silently producing a row that removed nothing.
    if any(name.startswith(ROLE_TENSOR_PREFIXES) for name in saved["model"]):
        config_roles_off = dict(config, source_role_markers=False)
        twin_roles, missing_roles = retargeted_twin(
            code, saved, config_roles_off, removed_prefixes=ROLE_TENSOR_PREFIXES)
        readings["role_markers_removed"] = geometry(code, twin_roles.eval(), batch,
                                                    channels, max_steps)
        notes["role_markers_removed_removed_tensors"] = missing_roles
        del twin_roles
    else:
        notes["role_markers_removed"] = (
            "not applicable: this checkpoint was trained with source_role_markers=False, "
            "so it carries no role tensors; the round's registered role contrast comes "
            "from the roles arm and is not re-run here")

    return readings, notes


def compare_to_archive(code: dict, checkpoint: Path, config: dict, batch: dict,
                       channels: tuple, archived_steps: list, max_steps: int) -> dict:
    """Agreement between a CPU re-measurement and the archived GPU numbers.

    The archived rows were produced on CUDA, so the comparison is float32 reduction
    order, not bitwise identity; the probe reports the largest difference instead of
    claiming the two are the same number.
    """
    _, _, model = load_checkpoint_arm(code, checkpoint)
    fresh = geometry(code, model.eval(), batch, channels, max_steps)
    if len(fresh["steps"]) != len(archived_steps):
        raise ValueError("the archive and the re-measurement disagree on the step count")
    deltas = {}
    for fresh_step, archived_step in zip(fresh["steps"], archived_steps):
        if int(fresh_step["step"]) != int(archived_step["step"]):
            raise ValueError("archived and re-measured step indices disagree")
        for key in ("update_norm_mean", "error_update_cosine_mean", "worsening_fraction"):
            have, want = fresh_step["aggregate"][key], archived_step.get(key)
            if have is None or want is None:
                continue
            deltas[f"step{int(fresh_step['step'])}.{key}"] = abs(float(have) - float(want))
    return {"fresh": fresh, "max_abs_delta_vs_archive": max(deltas.values(), default=0.0),
            "deltas_vs_archive": deltas}


def git_revision(root: Path) -> dict:
    """The revision the replay ran at, read from the repository rather than assumed."""
    import subprocess

    def run(*arguments):
        completed = subprocess.run(["git", *arguments], cwd=str(root), shell=False,
                                   capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
        return completed.stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": run("describe", "--always", "--dirty")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-root", required=True,
                        help="revision that trained the archived checkpoints")
    parser.add_argument("--archive", default="outputs/r7_72_rw_b_pilot")
    parser.add_argument("--store", default="outputs/r7_m2_segment/store/cache.zarr")
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", type=positive_int, default=8)
    parser.add_argument("--max-steps", type=positive_int, default=3)
    parser.add_argument("--lead-hours", type=positive_int, default=6)
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    args = parser.parse_args()

    seeds = tuple(int(part) for part in args.seeds.split(",") if part)
    if not seeds:
        raise ValueError("--seeds must list at least one seed")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    archive, store = Path(args.archive), Path(args.store)
    if not store.is_dir():
        raise FileNotFoundError(store)
    if args.max_steps > 8:
        raise ValueError("bounded probe requires max_steps <= 8")

    code = import_code_root(Path(args.code_root))
    code["torch"].set_num_threads(int(args.threads))
    running_digest = code["digest"]()

    checkpoints = {str(path.relative_to(archive)): sha256_file(path)
                   for path in sorted(archive.rglob("*.pt"))}
    if not checkpoints:
        raise FileNotFoundError(f"no archived checkpoints under {archive}")

    batch, channels, validation = val_batch(code, store, args.windows, args.lead_hours)
    started = time.perf_counter()
    records = {}
    for seed in seeds:
        if time.perf_counter() - started > args.deadline_seconds:
            raise RuntimeError(
                f"the {args.deadline_seconds:.0f}s deadline was reached before seed {seed}; "
                "stopping rather than overrunning the bound")
        seed_dir = archive / f"seed{seed}"
        archived = json.loads((seed_dir / "seed_result.json").read_text(encoding="utf-8"))
        if archived.get("test_read") is not False:
            raise ValueError(f"seed {seed} archive does not declare test_read false")
        if archived.get("model_code_sha256") != running_digest:
            raise ValueError(
                f"seed {seed} was trained by model code {archived.get('model_code_sha256')} "
                f"but --code-root reports {running_digest}; the guard is obeyed, so a "
                "replay here would be measuring a different revision")
        entry = {"archived_model_code_sha256": running_digest,
                 "archived_protocol_sha256": archived.get("protocol_sha256"),
                 "selected_update": {}, "checkpoint_sha256": {}, "readings": {},
                 "archived_reference": {}, "archive_agreement": {}, "notes": {}}
        for arm in (REFERENCE_ARM, FOCUS_ARM):
            update = int(archived["training"][arm]["selected_update"])
            checkpoint = seed_dir / "training" / arm / f"update_{update:07d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"archived checkpoint missing: {checkpoint}")
            entry["selected_update"][arm] = update
            entry["checkpoint_sha256"][arm] = sha256_file(checkpoint)
            if checkpoints[str(checkpoint.relative_to(archive))] != entry[
                    "checkpoint_sha256"][arm]:
                raise RuntimeError(f"{checkpoint} is not the byte set this run hashed")
            saved, config, model = load_checkpoint_arm(code, checkpoint)
            if saved["updates"] != update:
                raise ValueError(f"{checkpoint} holds update {saved['updates']}, not {update}")
            entry["readings"][arm] = geometry(code, model.eval(), batch, channels,
                                              args.max_steps)
            archived_steps = archived["probes"]["correction"][arm]["steps"]
            entry["archived_reference"][arm] = [
                {"step": step["step"], "update_norm_mean": step["update_norm_mean"],
                 "error_update_cosine_mean": step["error_update_cosine_mean"],
                 "worsening_fraction": step["worsening_fraction"]}
                for step in archived_steps]
            entry["archive_agreement"][arm] = compare_to_archive(
                code, checkpoint, config, batch, channels, archived_steps, args.max_steps)
            if arm == FOCUS_ARM:
                ablations, notes = measure_ablations(code, saved, config, checkpoint,
                                                     batch, channels, seed,
                                                     args.max_steps)
                entry["readings"].update(ablations)
                entry["notes"].update(notes)
            del model, saved
        records[str(seed)] = entry

    payload = {
        "format": "r7-72-rw-b-subtraction-probe-v1",
        "scientific_claim": False, "test_read": False, "gpu_hours": 0.0,
        "device": "cpu", "threads": int(args.threads), "thresholds_added": 0,
        "stage": "D1 of docs/goals/main-model-v2-rw-b-subtraction.md",
        "question": ("with one RW-B piece removed at a time from the trained checkpoints, "
                     "which removal reproduces or removes the ~1.8x step-1 magnitude and "
                     "the positive error/update cosine the bounded round reported"),
        "ablation_definitions": dict(ABLATIONS),
        "units": ("update_norm_mean is (mean per-case, per-variable squared update) ** 0.5 "
                  "in training-normalized units; error_update_cosine_mean is cos(error, "
                  "update) averaged over cases, per variable; worsening_fraction is the "
                  "per-case flag mean. All three are the round's own definitions."),
        "checkpoint_sha256": checkpoints,
        "validation": validation, "channels": list(channels),
        "code_root": str(Path(args.code_root).resolve()),
        "running_model_code_sha256": running_digest,
        "revision": git_revision(Path(args.code_root)),
        "command": " ".join(sys.argv),
        "cpu_seconds": None, "records": records,
        "limitations": [
            "validation split only, 8 windows, one lead: a diagnostic, never a skill number",
            "no threshold is introduced; these are the three readings the bounded round defined",
            "each row is an inference-time removal from a checkpoint trained with the piece on",
            "the gate+proposal-removed row routes an arm through a correction head that arm "
            "never trained; the probe reports whether training moved it, and the round's D4 "
            "arm retrains from scratch with the switch off rather than reusing this row",
            "the archived rows were measured on CUDA and this replay on CPU, so agreement is "
            "float32 reduction order and the differences are reported, not asserted away",
        ],
    }
    payload["cpu_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=False)
    print(f"wrote {output} in {payload['cpu_seconds']:.1f} CPU s over {len(records)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
