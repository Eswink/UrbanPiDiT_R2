"""Reusable machinery for the bounded multi-arm studies (#71 M1 / #72 M2-B).

A round of these studies is a handful of arms trained from one seed under one
frozen protocol. Three things have to be *measured* before any of its numbers
mean anything: that the tensors two arms share start bitwise equal, that the
reference arm's weights actually load into every other arm through the
trainer's own transfer rule, and what each arm costs in parameters and FLOPs.
Every round so far has needed the same three checks, so they live here rather
than being copied from one study script into the next. ``merge_seed_results``
and ``comparator_blocks`` are the other half: failing closed when a declared
seed is missing, or when two seeds ran under different protocol or model-code
digests, and turning the fixed comparator's output into the per-pair blocks a
round reports.

Nothing here decides a scientific question. There is no threshold, no verdict
and no reading rule in this module: those belong to the round's own script,
where they are written into ``protocol.json`` before the first optimizer step.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_config(kind, channels, config):
    """The model constructor kwargs for one arm, with its switches spread in."""
    return {"in_channels": channels, "out_channels": channels, "history_steps": 2,
            **config}


def count_flops(model, batch, *, reasoning_steps):
    """Forward-only and forward+backward FLOPs under one declared convention.

    ``FlopCounterMode`` counts linear/conv/matmul and aten-dispatched attention
    matmuls; elementwise and normalization ops are not counted, and no
    parameter hook is used (SDPA is parameterless, and a parameter hook would
    silently undercount attention). Gradients stay enabled for the counter's
    hooks. Backward is measured separately and is never assumed to be 2x the
    forward pass.
    """
    from model.r7_halting import forecast_inputs
    from torch.utils.flop_counter import FlopCounterMode

    inputs = forecast_inputs(batch)
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            model(inputs, reasoning_steps=reasoning_steps)
        forward = int(counter.get_total_flops())
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            output = model(inputs, reasoning_steps=reasoning_steps)
            output.forecast.square().mean().backward()
        forward_backward = int(counter.get_total_flops())
    model.zero_grad()
    return forward, forward_backward


def measure_arms(arms, *, channels, probe_batch, reasoning_steps, seed):
    """Parameters and FLOPs for every arm, measured before the protocol is frozen.

    Parameters come from ``training.r7_budget_audit.count_parameters`` and the
    forward count from its ``count_forward_flops`` at the same reasoning depth;
    the two forward counters have to agree, and a disagreement is an error
    rather than a convention to pick between. Forward+backward is measured
    separately with the same counter convention.
    """
    from training.r7_budget_audit import count_forward_flops, count_parameters
    from training.r7_experiment import make_model, seed_everything

    measured = {}
    for name, kind, config in arms:
        seed_everything(seed)
        model = make_model(kind, arm_config(kind, channels, config))
        counted = count_forward_flops(model, probe_batch, reasoning_steps=reasoning_steps)
        forward, forward_backward = count_flops(model, probe_batch,
                                                reasoning_steps=reasoning_steps)
        if counted != forward:
            raise RuntimeError(f"{name}: the two forward FLOP counters disagree "
                               f"({counted} vs {forward})")
        measured[name] = {"parameters": int(count_parameters(model)),
                          "forward_flops": forward,
                          "forward_backward_flops": forward_backward}
        del model
    return measured


def anchor_transfer_rule(anchor, target):
    """The transfer mapping ``run_scheduled_updates`` applies, mirrored by name."""
    applied = sorted(name for name, tensor in anchor.items()
                     if name in target and tuple(target[name].shape) == tuple(tensor.shape))
    ignored = sorted(name for name in anchor if name not in applied)
    return applied, ignored


def verify_arm_pairing(arms, *, baseline, channels, seed):
    """Measured, not assumed: every arm comes from one seeded initialization.

    Two checks, both real. First the state dicts are built from the same seed
    and compared name by name, so "shared" means the same name *and* a bitwise
    equal value for every pair of arms. Second the reference arm's tensors are
    loaded into a freshly constructed copy of each other arm through the
    trainer's own transfer rule, and every applied tensor is compared against
    the reference bitwise afterwards. The counts computed here are predictions
    of what the training report will contain, and the caller fails if the
    report disagrees.
    """
    from training.r7_experiment import make_model, seed_everything

    names = [name for name, _, _ in arms]
    states = {}
    for name, kind, config in arms:
        seed_everything(seed)
        model = make_model(kind, arm_config(kind, channels, config))
        states[name] = {key: value.clone() for key, value in model.state_dict().items()}

    pairs = {}
    for first in names:
        for second in names:
            if first >= second:
                continue
            shared = sorted(set(states[first]) & set(states[second]))
            pairs[f"{first}|{second}"] = {
                "shared_tensors": len(shared),
                "shared_parameters": int(sum(states[first][key].numel() for key in shared)),
                "shared_tensors_identical": bool(
                    all(torch.equal(states[first][key], states[second][key])
                        for key in shared)),
                "same_tensor_set": set(states[first]) == set(states[second]),
                "added_in_second": sorted(set(states[second]) - set(states[first])),
                "added_in_first": sorted(set(states[first]) - set(states[second])),
            }
    anchor = states[baseline]
    transfers = {}
    for name, kind, config in arms:
        if name == baseline:
            continue
        seed_everything(seed)
        model = make_model(kind, arm_config(kind, channels, config))
        target = model.state_dict()
        applied, ignored = anchor_transfer_rule(anchor, target)
        if not applied:
            raise RuntimeError(f"the reference arm loaded nothing into {name}")
        model.load_state_dict({key: anchor[key] for key in applied}, strict=False)
        loaded = model.state_dict()
        transfers[name] = {
            "applied_count": len(applied), "ignored_count": len(ignored),
            "applied_parameters": applied, "ignored_parameters": ignored,
            "post_load_all_applied_bitwise_equal": bool(
                all(torch.equal(loaded[key], anchor[key]) for key in applied)),
        }
        del model
    return {
        "anchor": baseline,
        "pairwise_shared_tensors": pairs,
        "all_shared_pairs_identical": bool(
            all(entry["shared_tensors_identical"] for entry in pairs.values())),
        "anchor_transfer": transfers,
        "tensor_counts": {name: len(state) for name, state in states.items()},
        "parameter_tensors": {name: int(sum(tensor.numel() for tensor in state.values()))
                              for name, state in states.items()},
        "rule": ("measured on same-seed state dicts; the reference arm is loaded into "
                 "every other arm with the trainer's own transfer rule and each applied "
                 "tensor is compared bitwise afterwards"),
    }


def merge_seed_results(output_dir, *, seeds, arms, fmt):
    """Merge per-seed runs, refusing an incomplete or inconsistent set.

    A declared seed with no result is an error: an incomplete round is reported
    as incomplete, never as a smaller round. Every (seed, arm) training record
    has to carry the one protocol digest and one model-code digest, so "one
    protocol for every run" is read off the runs themselves rather than trusted
    from the protocol file.
    """
    output_dir = Path(output_dir)
    loaded = {}
    for seed in seeds:
        path = output_dir / f"seed{seed}" / "seed_result.json"
        if not path.is_file():
            raise FileNotFoundError(f"declared seed {seed} has no result at {path}; an "
                                    "incomplete run is reported as incomplete")
        loaded[seed] = json.loads(path.read_text(encoding="utf-8"))
    digests = {result["protocol_sha256"] for result in loaded.values()}
    if len(digests) != 1:
        raise RuntimeError(f"seeds ran under different protocols: {sorted(digests)}")
    codes = {result["model_code_sha256"] for result in loaded.values()}
    if len(codes) != 1:
        raise RuntimeError(f"seeds ran on different model code: {sorted(codes)}")
    per_run = {f"{seed}/{arm}": entry["protocol_sha256"]
               for seed, result in loaded.items()
               for arm, entry in result["training"].items()}
    seen = {arm for result in loaded.values() for arm in result["training"]}
    if seen != set(arms) or len(per_run) != len(seeds) * len(arms):
        raise RuntimeError(f"expected {len(seeds) * len(arms)} arm runs, found "
                           f"{len(per_run)}: {sorted(per_run)}")
    if len(set(per_run.values())) != 1:
        raise RuntimeError(f"the runs do not share one protocol digest: {per_run}")
    for seed, result in loaded.items():
        if not result["arm_pairing"]["all_shared_pairs_identical"]:
            raise RuntimeError(f"seed {seed}: the arms did not share one initialization")
    first = loaded[seeds[0]]
    merged = {"format": fmt, "scientific_claim": False, "test_read": False,
              "seeds": list(seeds), "protocol_sha256": first["protocol_sha256"],
              "protocol": first["protocol"],
              "model_code_sha256": first["model_code_sha256"],
              "run_protocol_sha256": per_run,
              "flop_measurements": first["flop_measurements"],
              "arm_pairing": {str(seed): result["arm_pairing"]
                              for seed, result in loaded.items()},
              "segments": {str(seed): {"device": result["device"], "gpu": result["gpu"],
                                       "torch_version": result["torch_version"]}
                           for seed, result in loaded.items()},
              "training": {}, "evaluation": {}, "probes": {},
              "case_counts_by_lead": {},
              "budget": {"training_seconds_total": 0.0, "evaluation_seconds_total": 0.0}}
    for seed, result in loaded.items():
        merged["training"][str(seed)] = result["training"]
        for key, entry in result["evaluation"].items():
            merged["evaluation"][f"{seed}/{key}"] = {**entry, "seed": seed}
            merged["budget"]["evaluation_seconds_total"] += entry["elapsed_seconds"]
        merged["case_counts_by_lead"].update(result["case_counts_by_lead"])
        merged["probes"][str(seed)] = result.get("probes", {})
        merged["budget"]["training_seconds_total"] += result["budget"]["training_seconds_total"]
    if not merged["training"] or not merged["evaluation"]:
        raise RuntimeError("merged result is empty; refusing to score nothing")
    return merged


def _csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_study_tables(merged, output_dir):
    """The four cost views and the result tables, written as one set.

    Parameters and FLOPs, wall clock, peak memory and the case counts are
    reported together on purpose: forward-only FLOPs never stand in for training
    cost, and a win reported without the compute that bought it is not a
    result. Every table is written with ``'x'`` so a second finalize cannot
    silently overwrite the first.
    """
    output_dir = Path(output_dir)
    with (output_dir / "arm_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        has_trainable = all("trainable_parameters" in entry
                            for entry in merged["protocol"]["arms"])
        writer.writerow(["arm", "parameters", "forward_flops", "forward_backward_flops",
                         "switches", "gpu_seconds_total"]
                        + (["trainable_parameters"] if has_trainable else []))
        for entry in merged["protocol"]["arms"]:
            seconds = sum(per_arm[entry["name"]]["elapsed_seconds"]
                          for per_arm in merged["training"].values())
            writer.writerow([entry["name"], entry["parameters"], entry["forward_flops"],
                             entry["forward_backward_flops"],
                             " ".join(f"{key}={value}" for key, value
                                      in sorted(entry["switches"].items())),
                             f"{seconds:.1f}"]
                            + ([entry["trainable_parameters"]] if has_trainable else []))
    with (output_dir / "training_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "protocol_sha256", "selected_update", "updates_run",
                         "early_stopped", "seconds_per_update", "elapsed_seconds",
                         "shared_applied", "shared_ignored"])
        for seed in sorted(merged["training"], key=int):
            for arm, entry in sorted(merged["training"][seed].items()):
                shared = entry["shared_initial_state"]
                writer.writerow([seed, arm, entry["protocol_sha256"],
                                 entry["selected_update"], entry["updates_run"],
                                 entry["early_stopped"],
                                 f"{entry['seconds_per_update']:.6f}",
                                 f"{entry['elapsed_seconds']:.2f}",
                                 shared["applied_count"], shared["ignored_count"]])
    with (output_dir / "memory_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "phase", "lead_hours", "peak_allocated_mib",
                         "peak_reserved_mib", "elapsed_seconds"])
        for seed in sorted(merged["training"], key=int):
            for arm, entry in sorted(merged["training"][seed].items()):
                writer.writerow([seed, arm, "training", "",
                                 f"{(entry['peak_allocated_bytes'] or 0) / 2**20:.1f}",
                                 f"{(entry['peak_reserved_bytes'] or 0) / 2**20:.1f}",
                                 f"{entry['elapsed_seconds']:.2f}"])
        for _, entry in sorted(merged["evaluation"].items(),
                               key=lambda item: (item[1]["seed"], item[1]["arm"],
                                                 item[1]["lead_hours"])):
            writer.writerow([entry["seed"], entry["arm"], "evaluation", entry["lead_hours"],
                             f"{(entry.get('peak_allocated_bytes') or 0) / 2**20:.1f}", "",
                             f"{entry['elapsed_seconds']:.2f}"])
    with (output_dir / "rmse_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "lead_hours", "variable", "unit", "rmse",
                         "rmse_climatology", "mse_skill", "n_initializations"])
        for _, entry in sorted(merged["evaluation"].items(),
                               key=lambda item: (item[1]["seed"], item[1]["arm"],
                                                 item[1]["lead_hours"])):
            rmse = {row["variable"]: row for row in _csv_rows(entry["rmse_csv"])}
            skill = {row["variable"]: row for row in _csv_rows(entry["skill_csv"])}
            for variable, row in rmse.items():
                skill_row = skill.get(variable, {})
                writer.writerow([entry["seed"], entry["arm"], entry["lead_hours"], variable,
                                 row["unit"], row["rmse"],
                                 skill_row.get("rmse_climatology", ""),
                                 skill_row.get("mse_skill", "") or "undefined",
                                 row.get("n_initializations", entry["n_evaluated"])])
    with (output_dir / "case_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "lead_hours", "split", "n_available_windows",
                         "n_evaluated", "test_read"])
        for _, entry in sorted(merged["evaluation"].items(),
                               key=lambda item: (item[1]["seed"], item[1]["arm"],
                                                 item[1]["lead_hours"])):
            writer.writerow([entry["seed"], entry["arm"], entry["lead_hours"],
                             entry["split"], entry["n_available_windows"],
                             entry["n_evaluated"], "False"])


def pair_cells(blocks, *, leads):
    """Per-pair cell tables and three-state counts read off the comparator blocks.

    The counts are computed from the comparator's own per-cell verdicts, so a
    cell only ever lands in ``improved`` or ``worsened`` when every declared
    seed agreed in sign; everything else stays ``unresolved`` and is never
    averaged into a win. The full per-lead breakdown is kept beside the totals
    because a round whose leads disagree has that disagreement as its result.
    """
    pairs = {}
    for (focus, baseline), block in blocks.items():
        cells = {}
        for entry in block["seed_paired"]:
            cells[f"{int(entry['lead_hours'])}h|{entry['variable']}"] = {
                "outcome": entry["direction"], "unit": entry["unit"],
                "sign_consistent": entry["sign_consistent"],
                "seed_deltas": {str(item["seed"]): item["delta"]
                                for item in entry["seed_deltas"]},
                "direction_rule": ("the comparator labels a cell improved/worsened only "
                                   "when every declared seed's delta agrees in sign; "
                                   "otherwise it is unresolved")}
        pairs[f"{focus} - {baseline}"] = {
            "focus_arm": focus, "baseline_arm": baseline,
            "footnote": ("negative delta = the focus arm has the lower RMSE; the "
                         f"baseline is {baseline}"),
            "totals": {name: sum(1 for value in cells.values() if value["outcome"] == name)
                       for name in ("improved", "worsened", "unresolved")},
            "per_lead": {str(lead): {name: sum(1 for key, value in cells.items()
                                                 if key.startswith(f"{lead}h|")
                                                 and value["outcome"] == name)
                                      for name in ("improved", "worsened", "unresolved")}
                         for lead in leads},
            "cells": cells,
            "block": {key: block[key] for key in
                      ("arm", "baseline", "depth", "case_identity", "variables_improved",
                       "variables_worsened", "variables_compared", "variables_unresolved",
                       "variables_sign_consistent", "established_improved",
                       "established_worsened", "beats_baseline_everywhere")}}
    return pairs


def comparator_blocks(merged, *, pairs, depth, identity):
    """The fixed #60 comparator, reduced to the declared ``(focus, baseline)`` pairs.

    The comparator groups by variable *and* physical unit and refuses to pool
    across either, so a unit or case-set mismatch raises there instead of being
    averaged away here. A declared pair the comparator returns no block for is
    an error rather than a silently missing row.
    """
    from training.r7_coreasoning_compare import compare, summarize

    records = [{"arm": entry["arm"], "seed": entry["seed"], "depth": depth,
                "evaluation_dir": entry["evaluation_dir"], "identity": identity}
               for entry in merged["evaluation"].values()]
    table = summarize(records)
    wanted = set(pairs)
    blocks = {}
    for baseline in sorted({baseline for _, baseline in pairs}):
        for block in compare(table, baseline=baseline, depth=depth):
            if (block["arm"], block["baseline"]) in wanted:
                blocks[(block["arm"], block["baseline"])] = block
    missing = [pair for pair in pairs if pair not in blocks]
    if missing:
        raise RuntimeError(f"the comparator returned no block for {missing}")
    return table, blocks
