"""Actual rollout/checkpoint-body checks for the bounded precision probe.

The caller supplies its guards and frozen contract as ``api``. Imports that can
load tensor/data libraries stay inside calls, after the caller's socket denial.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import math
import time


def timed_update(torch, model, optimizer, batch, precision, deadline, api):
    from training.r7_autoregressive_rollout import training_two_step
    api.check_budget(deadline, reserve=0)
    if batch["coarse_history"].is_cuda:
        torch.cuda.synchronize(0)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch["coarse_history"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        out = training_two_step(model, batch, 4, .5)
    out.loss.backward()
    grads = {n: v.grad for n, v in model.named_parameters() if v.grad is not None}
    api.require(bool(grads), "training has no gradients")
    api.finite_tensors(torch, grads, "gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    api.check_budget(deadline, reserve=0)
    optimizer.step()
    api.finite_tensors(torch, model.state_dict(), "weight")
    if batch["coarse_history"].is_cuda:
        torch.cuda.synchronize(0)
    api.check_budget(deadline, reserve=0)
    return {"loss": float(out.loss.detach()), "l6": float(out.l6.detach()), "l12": float(out.l12.detach()),
            "gradient_norm": float(norm), "elapsed_seconds": time.perf_counter() - started}


def trace_rollout(torch, model, batch, precision, api, *, one_step=False):
    from training.r7_autoregressive_rollout import training_one_step, training_two_step
    captures = {"forecast": [], "encoder": [], "reader": [], "internal_k": [], "calendar": [], "phases": []}

    def before(_module, args, kwargs):
        inputs = args[0]
        api.require(kwargs.get("detach_between_steps") is False and kwargs.get("reasoning_steps") == 4,
                    "full internal K required")
        api.require(not any(k in inputs for k in ("atmos_target", "future_target", "process_targets", "future_process_targets")),
                    "target poison reached model inputs")
        captures["calendar"].append({k: inputs[k].detach().cpu().tolist() for k in
                                    ("lead_time_hours", "init_calendar_year", "init_day_of_year", "init_utc_hour")})
        if captures["forecast"]:
            api.require(torch.equal(inputs["coarse_history"][:, -1], captures["forecast"][0]),
                        "second history differs from first prediction")

    def capture(name, value):
        api.require(value.requires_grad, name + " graph detached")
        value.retain_grad()
        captures[name].append(value)

    def phase_hook(_module, args):
        captures["phases"].append(args[0].detach().float().cpu()[:, :, 4:])

    handles = [model.backbone.spacetime.net.register_forward_pre_hook(phase_hook),
               model.register_forward_pre_hook(before, with_kwargs=True),
               model.register_forward_hook(lambda _m, _a, out: capture("forecast", out.forecast)),
               model.backbone.encoder.register_forward_hook(lambda _m, _a, out: capture("encoder", out[0])),
               model.process_reader.register_forward_hook(lambda _m, _a, out: capture("reader", out)),
               model.reasoning_cell.register_forward_hook(lambda _m, _a, out: capture("internal_k", out))]
    try:
        with torch.autocast(batch["coarse_history"].device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            out = training_one_step(model, batch, 4) if one_step else training_two_step(model, batch, 4, .5)
    finally:
        for handle in handles:
            handle.remove()
    count = 1 if one_step else 2
    api.require(len(captures["forecast"]) == len(captures["encoder"]) == len(captures["phases"]) == count
                and len(captures["reader"]) == len(captures["internal_k"]) == count * 4,
                "physical/internal K forward counts differ")
    for step, actual in enumerate(captures["calendar"]):
        dates = [datetime.fromisoformat(s) + timedelta(hours=6 * step) for s in batch["init_time"]]
        expected = {"lead_time_hours": [6.] * len(dates), "init_calendar_year": [d.year for d in dates],
                    "init_day_of_year": [d.timetuple().tm_yday for d in dates], "init_utc_hour": [d.hour for d in dates]}
        api.require(actual == expected, "actual transition calendar/lead differs from independent date oracle")
        angles = []
        for initial in dates:
            valid = initial + timedelta(hours=6)
            days = (datetime(valid.year + 1, 1, 1) - datetime(valid.year, 1, 1)).days
            annual = math.tau * (valid.timetuple().tm_yday - 1 + valid.hour / 24) / days
            diurnal = math.tau * valid.hour / 24
            angles.append([math.sin(annual), math.cos(annual), math.sin(diurnal), math.cos(diurnal)])
        phases = captures["phases"][step]
        oracle = torch.tensor(angles, dtype=torch.float32)[:, None].expand_as(phases)
        # Frozen FP32 feature arithmetic allowance; no forecast/resume tolerance.
        torch.testing.assert_close(phases, oracle, rtol=0, atol=api.CONTROLS["calendar_phase_atol"])
    return out, captures


def diagnostic_checks(torch, model, batch, precision, api):
    model.zero_grad(set_to_none=True)
    out, traces = trace_rollout(torch, model, batch, precision, api)
    out.l12.backward()  # No L6 contribution: prove physical and internal-K connectivity.
    ownership = {}
    for name, values in (("first_forecast", traces["forecast"][:1]), ("first_encoder", traces["encoder"][:1]),
                         ("first_reader", traces["reader"][:4]), ("first_internal_k", traces["internal_k"][:4]),
                         ("encoder_parameters", list(model.backbone.encoder.parameters())),
                         ("reader_parameters", list(model.process_reader.parameters()))):
        grads = [v.grad for v in values]
        api.require(all(g is not None for g in grads), name + " missing gradient")
        api.finite_tensors(torch, {str(i): g for i, g in enumerate(grads)}, name)
        norms = [float(g.detach().float().norm()) for g in grads]
        api.require(all(n > 0 for n in norms), name + " zero gradient")
        ownership[name] = norms
    clean = out.forecasts.detach().clone()
    clean_loss = float(out.loss.detach())
    model.zero_grad(set_to_none=True)
    poisoned, _ = trace_rollout(torch, model, dict(batch, future_target=batch["future_target"] + 13.), precision, api)
    api.require(torch.equal(clean, poisoned.forecasts.detach()), "future target changed a forecast")
    api.require(clean_loss != float(poisoned.loss.detach()), "future target poison did not change loss")
    control, one = trace_rollout(torch, model, batch, precision, api, one_step=True)
    api.require(torch.equal(control.forecasts[:, 0].detach(), clean[:, 0]), "one-step forward differs from first two-step forward")
    model.zero_grad(set_to_none=True)
    control.loss.backward()
    api.finite_tensors(torch, {n: p.grad for n, p in model.named_parameters() if p.grad is not None}, "one-step gradient")
    return {"l12_gradient_ownership": ownership, "calendar_trace": traces["calendar"],
            "valid_time_phase_trace": [value[:, 0].tolist() for value in traces["phases"]],
            "two_step_forward_calls": 2, "internal_k_calls": 8, "one_step_forward_calls": len(one["forecast"]),
            "poison_forecasts_equal": True, "poison_loss_changed": True, "finite": True, "full_gradient": True}


def resume_paths(torch, model, batch, protocol, precision, deadline, api):
    from training.r7_experiment import canonical_digest, load_checkpoint, restore_rng, rng_state, seed_everything
    from training.r7_scheduled_runner import _publish_checkpoint
    initial, endings, reports = deepcopy(model.state_dict()), [], []
    for branch in ("uninterrupted", "intentional_resume"):
        folder = api.output_path(Path(protocol["output"]) / precision / branch, fresh=True)
        folder.mkdir(parents=True, exist_ok=False)
        contract = {"kind": "process", "model": protocol["inputs"]["model_spec"], "total_updates": 2, "steps": 4,
                    "controls": api.CONTROLS, "precision": precision, "output_dir": str(folder),
                    "protocol_sha256": protocol["protocol_sha256"], "model_code_sha256": protocol["code"]["model_code_sha256"],
                    "data_identity": protocol["inputs"]["parent_import"]["data"]["data_identity"],
                    "source_sha256": protocol["inputs"]["parent_import"]["data"]["sources"]["source_sha256"],
                    "initialization": protocol["inputs"]["parent_import"]["report_sha256"]}
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
                    api.write_json(folder / "intentional_stop.json", {"status": "intentional-stop-not-failed", "endpoint": 2,
                                   "updates": 1, "signature": signature, "checkpoint_sha256": api.sha256_file(checkpoint)})
                    check = lambda: api.check_budget(deadline, reserve=0)
                    api.verify_code(protocol, check)
                    api.verify_inputs(protocol, check)
                    saved = load_checkpoint(api.local_path(checkpoint), expected=signature)
                    api.finite_tensors(torch, saved["model"], "resume weight")
                    for index, state in saved["optimizer"]["state"].items():
                        api.finite_tensors(torch, state, "resume optimizer " + str(index))
                    api.require(optimizer.state_dict()["param_groups"] == saved["optimizer"]["param_groups"],
                                "frozen fresh AdamW optimizer settings changed")
                    api.require(saved["contract"] == contract and saved["updates"] == 1 and saved["epoch"] == 0 and saved["cursor"] == 2,
                                "own intermediate new contract differs")
                    api.require(not (folder / "failed_attempt.json").exists() and checkpoint.parent == folder,
                                "failed/foreign resume refused")
                    model.load_state_dict(saved["model"], strict=True)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                    optimizer.load_state_dict(saved["optimizer"])
                    restore_rng(saved["rng"])
        api.finite_tensors(torch, model.state_dict(), "endpoint weight")
        endings.append({"weights": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()), "rng": rng_state(),
                        "losses": [{k: row[k] for k in ("loss", "l6", "l12", "gradient_norm")} for row in rows]})
        reports.append({"branch": branch, "updates": 2, "losses": rows, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": api.sha256_file(checkpoint), "signature": signature})
    acceptance = {name: api.exact_equal(torch, endings[0][name], endings[1][name]) for name in endings[0]}
    api.write_json(Path(protocol["output"]) / precision / "resume_comparison.json",
                   {"exact": acceptance, "scientific_claim": False, "limitations": api.LIMITATIONS})
    api.require(all(acceptance.values()), "exact same-endpoint resume failed; no tolerance or attempt repair")
    return reports, acceptance
