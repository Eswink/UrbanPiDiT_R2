"""New-package actual loss/gradient/query/checkpoint-body engineering checks.

No M3 import or old output reads. Library imports remain lazy behind caller's
socket denial. Idle diagnostic/additive heads are declared, never counted PASS.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import math
import time
import sys


def require_source_runtime(api):
    """A shadow model/package from another sys.path cannot qualify this source."""
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("model", "training", "data", "tools") and getattr(module, "__file__", None):
            api.require(api.local_path(module.__file__).is_relative_to(api.ROOT), "shadow active-source module: " + name)


ACTIVE_PREFIXES = ("backbone.encoder.", "backbone.known_context.projection.", "process_reader.",
                   "draft_encoder.", "role_context", "role_draft", "solver_init", "solver_cell.", "solver_gate.", "proposal_head.")
IDLE_PREFIXES = ("process_readout.", "process_to_context.", "latent_to_context.", "correction_head.")
GRADIENT_GROUPS = ("first_forecast", "first_encoder", "first_known", "first_reader", "first_internal_k", "first_solver",
                   "encoder_parameters", "known_parameters", "reader_parameters", "draft_parameters", "roles", "latent",
                   "solver_parameters", "proposal_parameters")


def finite_tensors(torch, values, label):
    for name, value in values.items():
        if not torch.is_tensor(value) or ((value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all())):
            raise ValueError(label + " missing/nonfinite tensor: " + name)


def tensor_sha256(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode() + b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def exact_equal(torch, left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_equal(torch, v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def strict_capabilities(model, api):
    for flag in api.PACKAGE_FLAGS:
        api.require(hasattr(model, flag) and getattr(model, flag) is True, "public package capability required: " + flag)
    api.require(model.detach_between_steps is False and model.default_reasoning_steps == 4, "full public K configuration required")
    api.require(model.backbone.known_context_inputs is True and hasattr(model.backbone, "known_context"), "known projection module required")
    api.require(model.process_reader.draft_query_feedback is True, "reader query feedback capability required")


def require_active_gradients(torch, model, api):
    active, idle = {}, {}
    for name, parameter in model.named_parameters():
        if name.startswith(IDLE_PREFIXES):
            api.require(parameter.grad is None, "declared idle head unexpectedly participates: " + name)
            idle[name] = "inapplicable forecast objective; not passed"
            continue
        api.require(parameter.grad is not None, "active parameter has no gradient: " + name)
        finite_tensors(torch, {name: parameter.grad}, "gradient")
        active[name] = float(parameter.grad.detach().float().norm())
        api.require(active[name] > 0, "active parameter gradient is zero: " + name)
    return active, idle


def timed_update(torch, model, optimizer, batch, precision, deadline, api):
    from training.r7_autoregressive_rollout import training_two_step
    api.check_budget(deadline, reserve=0)
    if batch["coarse_history"].is_cuda:
        torch.cuda.synchronize(0)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch["coarse_history"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        out = training_two_step(model, batch, 4, .5)
    api.require(out.loss.dtype == out.l6.dtype == out.l12.dtype == torch.float32, "actual deep objective must remain FP32")
    out.loss.backward()
    finite_tensors(torch, {n: v.grad for n, v in model.named_parameters() if v.grad is not None}, "gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    api.check_budget(deadline, reserve=0)
    optimizer.step()
    finite_tensors(torch, model.state_dict(), "weight")
    for index, state in optimizer.state_dict()["state"].items():
        finite_tensors(torch, state, "optimizer " + str(index))
    if batch["coarse_history"].is_cuda:
        torch.cuda.synchronize(0)
    api.check_budget(deadline, reserve=0)
    return {"loss": float(out.loss.detach()), "l6": float(out.l6.detach()), "l12": float(out.l12.detach()),
            "gradient_norm": float(norm), "elapsed_seconds": time.perf_counter() - started, "loss_dtype": "torch.float32"}


def known_feature_oracle(torch, inputs, token_hw, patch_size):
    longitude = inputs["longitude"].detach().cpu()
    if longitude.ndim == 2:
        longitude = longitude[0]
    features = []
    for stamp in range(len(inputs["init_utc_hour"])):
        initial = datetime(int(inputs["init_calendar_year"][stamp]), 1, 1) + timedelta(
            days=float(inputs["init_day_of_year"][stamp]) - 1, hours=float(inputs["init_utc_hour"][stamp]))
        samples = []
        for _row in range(token_hw[0]):
            for column in range(token_hw[1]):
                longitudes = [math.radians(float(longitude[min(column * patch_size + i, len(longitude) - 1)])) for i in range(patch_size)]
                values = []
                for offset in (-6., 0.):
                    date = initial + timedelta(hours=offset)
                    days = (datetime(date.year + 1, 1, 1) - datetime(date.year, 1, 1)).days
                    annual = math.tau * (date.timetuple().tm_yday - 1 + date.hour / 24) / days
                    phases = [math.tau * date.hour / 24 + lon for lon in longitudes]
                    values.extend([math.sin(annual), math.cos(annual), sum(map(math.sin, phases)) / patch_size,
                                   sum(map(math.cos, phases)) / patch_size, offset / 24])
                valid = initial + timedelta(hours=6)
                phases = [math.tau * valid.hour / 24 + lon for lon in longitudes]
                samples.append(values + [sum(map(math.sin, phases)) / patch_size, sum(map(math.cos, phases)) / patch_size])
        features.append(samples)
    return torch.tensor(features, dtype=torch.float32)


def trace_rollout(torch, model, batch, precision, api, *, one_step=False):
    from training.r7_autoregressive_rollout import training_one_step, training_two_step
    strict_capabilities(model, api)
    captures = {n: [] for n in ("forecast", "encoder", "known", "reader", "internal_k", "solver", "draft", "query", "features", "calendar", "inputs")}

    def before(_module, args, kwargs):
        inputs = args[0]
        api.require(kwargs.get("detach_between_steps") is False and kwargs.get("reasoning_steps") == 4, "full internal K required")
        api.require(not any(k in inputs for k in ("atmos_target", "future_target", "process_targets", "future_process_targets", "atmos_baseline")), "target poison reached encoder inputs")
        api.require(inputs["history_offsets_hours"].tolist() == [[-6., 0.]] * 2, "explicit ordered history offsets required")
        captures["inputs"].append({k: v.detach().cpu() for k, v in inputs.items() if torch.is_tensor(v)})
        captures["calendar"].append({k: inputs[k].detach().cpu().tolist() for k in ("lead_time_hours", "init_calendar_year", "init_day_of_year", "init_utc_hour", "history_offsets_hours")})
        if captures["forecast"]:
            api.require(torch.equal(inputs["coarse_history"][:, -1], captures["forecast"][0]), "second history differs from pred1")
            api.require(torch.equal(inputs["coarse_history"][:, -2], batch["coarse_history"][:, -1]), "known history frame advanced incorrectly")

    def capture(name, value):
        api.require(value.requires_grad, name + " graph detached")
        value.retain_grad()
        captures[name].append(value)

    def query(_module, args, kwargs):
        encoded = kwargs.get("draft_tokens")
        api.require(bool(captures["draft"]) and encoded is captures["draft"][-1], "query must reuse the exact existing draft encoding")
        captures["query"].append(1)

    cell = model.reasoning_cell if hasattr(model, "reasoning_cell") else model.cell
    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
               model.register_forward_hook(lambda _m, _a, out: capture("forecast", out.forecast)),
               model.backbone.encoder.register_forward_hook(lambda _m, _a, out: capture("encoder", out[0])),
               model.backbone.known_context.register_forward_hook(lambda _m, _a, out: capture("known", out)),
               model.backbone.known_context.projection.register_forward_pre_hook(lambda _m, args: captures["features"].append(args[0].detach().float().cpu())),
               model.draft_encoder.register_forward_hook(lambda _m, _a, out: capture("draft", out[0])),
               model.process_reader.register_forward_pre_hook(query, with_kwargs=True),
               model.process_reader.register_forward_hook(lambda _m, _a, out: capture("reader", out)),
               cell.register_forward_hook(lambda _m, _a, out: capture("internal_k", out)),
               model.solver_cell.register_forward_hook(lambda _m, _a, out: capture("solver", out))]
    try:
        with torch.autocast(batch["coarse_history"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            out = training_one_step(model, batch, 4) if one_step else training_two_step(model, batch, 4, .5)
    finally:
        for handle in handles:
            handle.remove()
    count = 1 if one_step else 2
    api.require(len(captures["forecast"]) == len(captures["encoder"]) == len(captures["features"]) == count
                and all(len(captures[n]) == count * 4 for n in ("reader", "internal_k", "solver", "draft", "query")), "physical/K/query forward counts differ")
    for step, actual in enumerate(captures["calendar"]):
        dates = [datetime.fromisoformat(s) + timedelta(hours=6 * step) for s in batch["init_time"]]
        expected = {"lead_time_hours": [6.] * 2, "init_calendar_year": [d.year for d in dates], "init_day_of_year": [d.timetuple().tm_yday for d in dates],
                    "init_utc_hour": [d.hour for d in dates], "history_offsets_hours": [[-6., 0.]] * 2}
        api.require(actual == expected, "known calendar/history metadata did not advance by6")
        history = captures["inputs"][step]["coarse_history"]
        token_hw = tuple((size + model.patch_size - 1) // model.patch_size for size in history.shape[-2:])
        oracle = known_feature_oracle(torch, captures["inputs"][step], token_hw, model.patch_size)
        torch.testing.assert_close(captures["features"][step], oracle, rtol=0, atol=api.CONTROLS["calendar_feature_atol"])
    api.require(out.loss.dtype == out.l6.dtype == torch.float32 and (one_step or out.l12.dtype == torch.float32), "strict FP32 loss required")
    return out, captures


def diagnostic_checks(torch, model, batch, precision, api):
    model.zero_grad(set_to_none=True)
    out, trace = trace_rollout(torch, model, batch, precision, api)
    out.l12.backward()  # Exclusively physical +12 loss; no +6 contribution.
    groups = {"first_forecast": trace["forecast"][:1], "first_encoder": trace["encoder"][:1], "first_known": trace["known"][:1],
              "first_reader": trace["reader"][:4], "first_internal_k": trace["internal_k"][:4], "first_solver": trace["solver"][:4],
              "encoder_parameters": list(model.backbone.encoder.parameters()), "known_parameters": list(model.backbone.known_context.parameters()),
              "reader_parameters": list(model.process_reader.parameters()), "draft_parameters": list(model.draft_encoder.parameters()),
              "roles": [model.role_context, model.role_draft], "latent": [model.process_queries if hasattr(model, "process_queries") else model.latent],
              "solver_parameters": [model.solver_init, *model.solver_cell.parameters(), *model.solver_gate.parameters()],
              "proposal_parameters": list(model.proposal_head.parameters())}
    ownership = {}
    for name, values in groups.items():
        finite_tensors(torch, {str(i): v.grad for i, v in enumerate(values)}, name)
        norms = [float(v.grad.detach().float().norm()) for v in values]
        api.require(all(n > 0 for n in norms), name + " zero physical L12 gradient")
        ownership[name] = norms
    active, idle = require_active_gradients(torch, model, api)
    clean, loss = out.forecasts.detach().clone(), float(out.loss.detach())
    model.zero_grad(set_to_none=True)
    poisoned = dict(batch, atmos_target=batch["atmos_target"] + 11., future_target=batch["future_target"] + 13.,
                    process_targets=object(), future_process_targets=object(), atmos_baseline=object(), init_year=object())
    changed, _ = trace_rollout(torch, model, poisoned, precision, api)
    api.require(torch.equal(clean, changed.forecasts.detach()), "future/diagnostic targets changed predictions")
    api.require(loss != float(changed.loss.detach()), "target poison did not change actual loss")
    control, _ = trace_rollout(torch, model, batch, precision, api, one_step=True)
    api.require(torch.equal(control.forecasts[:, 0].detach(), clean[:, 0]), "one-step and two-step first forecast differ")
    model.zero_grad(set_to_none=True)
    control.loss.backward()
    finite_tensors(torch, {n: p.grad for n, p in model.named_parameters() if p.grad is not None}, "one-step gradient")
    return {"l12_gradient_ownership": ownership, "active_parameter_gradients": active, "idle_parameters": idle,
            "calendar_trace": trace["calendar"], "known_feature_sha256": [tensor_sha256(v) for v in trace["features"]],
            "two_step_forward_calls": 2, "internal_k_calls": 8, "query_reuse_calls": 8, "one_step_forward_calls": 1,
            "poison_forecasts_equal": True, "poison_loss_changed": True, "finite": True, "full_gradient": True}


def resume_paths(torch, model, batch, protocol, precision, deadline, api):
    from training.r7_experiment import canonical_digest, load_checkpoint, restore_rng, rng_state, seed_everything
    from training.r7_scheduled_runner import _publish_checkpoint
    from training.r7_v2_profile import state_hash
    initial, endings, reports = deepcopy(model.state_dict()), [], []
    root = api.output_path(protocol["output"])
    configured = protocol["configuration"]
    report = configured["mapping_reports"]["41"]["process"]
    api.require(state_hash(initial) == report["target_state_sha256"], "initial weights not the frozen mapped anchor")
    for branch in ("uninterrupted", "intentional_resume"):
        folder = api.local_path(root / precision / branch)
        if folder.exists():
            raise FileExistsError("independent branch already exists; no resurrection")
        folder.mkdir(parents=True, exist_ok=False)
        contract = {"kind": "process", "model": configured["value"]["model_specs"]["process"]["model"], "total_updates": 2, "steps": 4,
                    "controls": api.CONTROLS, "precision": precision, "output_dir": str(folder), "configuration_sha256": configured["file_sha256"],
                    "protocol_sha256": protocol["protocol_sha256"], "model_code_sha256": protocol["code"]["model_code_sha256"],
                    "data_identity": protocol["inputs"]["data_identity"], "source_sha256": protocol["inputs"]["sources"]["source_sha256"],
                    "initialization": report, "initial_state_sha256": state_hash(initial)}
        signature = canonical_digest(contract)
        api.write_json(folder / "started.json", {"contract": contract, "signature": signature, "scientific_claim": False})
        model.load_state_dict(initial, strict=True)
        seed_everything(41)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        rows = []
        for update in (1, 2):
            rows.append(timed_update(torch, model, optimizer, batch, precision, deadline, api))
            if branch == "intentional_resume" or update == 2:
                checkpoint = _publish_checkpoint(folder, update, model=model, optimizer=optimizer, signature=signature,
                                                 contract=contract, epoch=update - 1, cursor=2, intervention=None)
                if branch == "intentional_resume" and update == 1:
                    api.write_json(folder / "intentional_stop.json", {"status": "intentional-stop-not-failed", "endpoint": 2, "updates": 1,
                                   "signature": signature, "checkpoint_sha256": api.sha256_file(checkpoint)})
                    check = lambda: api.check_budget(deadline, reserve=0)
                    api.verify_code(protocol, check)
                    api.verify_inputs(protocol, check)
                    saved = load_checkpoint(api.local_path(checkpoint), expected=signature)
                    finite_tensors(torch, saved["model"], "resume weight")
                    for index, state in saved["optimizer"]["state"].items():
                        finite_tensors(torch, state, "resume optimizer " + str(index))
                    api.require(optimizer.state_dict()["param_groups"] == saved["optimizer"]["param_groups"], "frozen fresh AdamW settings differ")
                    api.require(saved["contract"] == contract and saved["updates"] == 1 and saved["epoch"] == 0 and saved["cursor"] == 2,
                                "own intermediate new configuration/source contract differs")
                    api.require(not (folder / "failed_attempt.json").exists() and checkpoint.parent == folder, "failed/foreign resume refused")
                    model.load_state_dict(saved["model"], strict=True)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                    optimizer.load_state_dict(saved["optimizer"])
                    restore_rng(saved["rng"])
        finite_tensors(torch, model.state_dict(), "endpoint weight")
        endings.append({"weights": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()), "rng": rng_state(),
                        "losses": [{k: row[k] for k in ("loss", "l6", "l12", "gradient_norm")} for row in rows]})
        reports.append({"branch": branch, "updates": 2, "losses": rows, "checkpoint": str(checkpoint), "checkpoint_sha256": api.sha256_file(checkpoint), "signature": signature})
    acceptance = {name: exact_equal(torch, endings[0][name], endings[1][name]) for name in endings[0]}
    api.write_json(Path(protocol["output"]) / precision / "resume_comparison.json", {"exact": acceptance, "scientific_claim": False, "limitations": api.LIMITATIONS})
    api.require(all(acceptance.values()), "exact same-endpoint package resume failed; no tolerance or attempt repair")
    return reports, acceptance


def validate_worker_result(result, protocol, precision, api):
    checks = result.get("checks", {})
    api.require(result.get("status") == "success" and result.get("scientific_claim") is False and result.get("limitations") == api.LIMITATIONS
                and result.get("test_read") is False and result.get("precision") == precision and result.get("protocol_sha256") == protocol["protocol_sha256"]
                and result.get("model_code_sha256") == protocol["code"]["model_code_sha256"]
                and result.get("configuration_sha256") == protocol["configuration"]["file_sha256"]
                and result.get("initial_state_sha256") == protocol["configuration"]["mapping_reports"]["41"]["process"]["target_state_sha256"]
                and result.get("actual_optimizer_updates") == 4 and result.get("baseline_allocated_bytes") == result.get("baseline_reserved_bytes") == 0,
                "complete new-package finite fresh-CUDA receipt required; no skip")
    api.require(result.get("resume_exact") == {n: True for n in ("weights", "optimizer", "rng", "losses")}, "exact own2-update resume required")
    api.require(all(checks.get(n) is True for n in ("finite", "full_gradient", "poison_forecasts_equal", "poison_loss_changed"))
                and checks.get("two_step_forward_calls") == 2 and checks.get("one_step_forward_calls") == 1
                and checks.get("internal_k_calls") == checks.get("query_reuse_calls") == 8 and len(checks.get("calendar_trace", [])) == 2
                and len(checks.get("known_feature_sha256", [])) == 2, "full gradient/calendar/query reuse/poison evidence required")
    api.require(set(checks.get("l12_gradient_ownership", {})) == set(GRADIENT_GROUPS), "all physical L12 ownership groups required")
    for norms in checks["l12_gradient_ownership"].values():
        api.require(bool(norms) and all(type(n) in (float, int) and math.isfinite(n) and n > 0 for n in norms), "finite nonzero L12 gradient required")
    active, idle = checks.get("active_parameter_gradients", {}), checks.get("idle_parameters", {})
    api.require(bool(active) and bool(idle) and not set(active) & set(idle), "explicit active/inapplicable ownership map required")
    api.require(all(type(n) in (float, int) and math.isfinite(n) and n > 0 for n in active.values())
                and all(name.startswith(IDLE_PREFIXES) and "inapplicable" in reason for name, reason in idle.items()),
                "finite active gradients and explicit idle exclusions required")
    api.require([b.get("branch") for b in result.get("branches", [])] == ["uninterrupted", "intentional_resume"]
                and all(b.get("updates") == len(b.get("losses", [])) == 2 for b in result["branches"]), "both actual own2-update endpoints required")
    for name in ("peak_reserved_bytes", "peak_allocated_bytes", "elapsed_seconds"):
        value = result.get(name)
        api.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite actual CUDA memory/timing required")
