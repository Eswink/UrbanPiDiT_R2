"""One predeclared 200-to-800 resume control, using validation only."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
from .r7_cpu_study import model_settings, _report_matrix
from .r7_experiment import canonical_digest

VARIANTS = ("generic", "process", "process_no_aux")
SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 72)
FINAL_UPDATES = 800


def validate_parent(saved, variant, seed, identity):
    if variant not in VARIANTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("unsupported original variant/seed")
    kind, model, weight = model_settings(variant)
    expected = dict(kind=kind, model=model, data_identity=identity, batch_size=2,
        accumulation=1, steps=3, seed=seed, lr=2e-4, process_weight=weight,
        clip=1., bf16=False, device_type="cpu", dataset_length=120,
        optimization="streamed-truncated")
    contract = saved["contract"]
    if type(saved.get("updates")) is not int or saved["updates"] != 200:
        raise ValueError("original checkpoint must be exactly update200")
    if saved.get("signature") != canonical_digest(contract):
        raise ValueError("original contract digest mismatch")
    for key, value in expected.items():
        if contract.get(key) != value or type(contract.get(key)) is not type(value):
            raise ValueError(f"original training contract mismatch: {key}")
    return contract


def summarize_control(records):
    expected = {(v,s,e,k) for v,s in product(VARIANTS,SEEDS)
                for e,k in ((200,3),(800,3),(800,1))}
    seen, paired, groups = set(), None, {}
    for record in records:
        key = tuple(record[n] for n in ("variant","seed","updates","depth"))
        if key not in expected or key in seen:
            raise ValueError("unexpected/duplicate control record")
        seen.add(key)
        r = record["report"]
        if r["split"] != "val" or tuple(r["lead_hours"]) != LEADS:
            raise ValueError("control reports must be validation-only at fixed leads")
        sig, matrix = _report_matrix(r)
        counts = Counter(datetime.fromisoformat(x["init_time"]).month for x in r["initializations"])
        if counts != {1:6,4:6,7:6,9:6}:
            raise ValueError("all24 balanced validation cases required")
        if paired is not None and paired != sig:
            raise ValueError("unpaired validation identities/cases/schema")
        paired = sig
        for i, lead in enumerate(LEADS):
            for j, name in enumerate(r["channels"]):
                group = (key[0],key[2],key[3],lead,name,r["units"][j])
                groups.setdefault(group, []).append(float(matrix[i,j]))
    if seen != expected:
        raise ValueError("missing predeclared control records")
    return [dict(variant=v,updates=e,depth=k,lead_hours=h,variable=n,unit=u,
                 seed_rmse_mean=float(np.mean(x)),seed_rmse_sd=float(np.std(x,ddof=1)),seeds=len(x))
            for (v,e,k,h,n,u),x in sorted(groups.items())]


def _write(path, value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False)


def run_extended_control(artifact_root, output_dir):
    import torch
    from data.restore_pilot_cache import restore_pilot_cache
    from .r7_experiment import load_checkpoint, dataset_identity
    from .r7_local_runner import run_local_updates
    from .r7_calibration_runner import file_sha256
    from .r7_evaluate import evaluate_local
    root, out = Path(artifact_root)/"forecasts", Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    parents = {(v,s):root/"training"/f"{v}_{s}"/"update_0000200.pt" for v,s in product(VARIANTS,SEEDS)}
    hashes = {f"{v}_{s}":file_sha256(p) for (v,s),p in parents.items()}
    _, restoration = restore_pilot_cache(root/"source"/"era5_pressure_pilot.nc",
        root/"source"/"receipt.json",root/"manifests",parents["process",41],out/"restored")
    train = out/"restored"/"manifests"/"train.jsonl"
    val = out/"restored"/"manifests"/"val_balanced.jsonl"
    identity, dataset = dataset_identity(train)
    contracts = {(v,s):validate_parent(load_checkpoint(p),v,s,identity) for (v,s),p in parents.items()}
    protocol = dict(format="r7-extended-validation-control-v1",variants=list(VARIANTS),
        seeds=list(SEEDS),parent_updates=200,final_updates=FINAL_UPDATES,
        extra_updates_per_run=600,inference_depths=[1,3],lead_hours=list(LEADS),validation_cases=24,
        original_checkpoint_hashes=hashes,training_identity=identity,source=restoration["pinned_source_sha256"],
        test_evaluated=False,threshold_selection=False,new_architecture=False,
        scope="one fixed endpoint; no extension-until-win or test-driven selection")
    protocol["protocol_sha256"] = canonical_digest(protocol)
    _write(out/"protocol.json",protocol)
    records, resources = [], []
    original = json.loads((root/"study_result.json").read_text())
    for (variant,seed),parent in parents.items():
        if time.monotonic()-started > 1080:
            raise RuntimeError("extended-control wall budget exhausted")
        contract = contracts[variant,seed]
        baseline = evaluate_local(val,output_dir=out/"evaluation"/f"{variant}_{seed}_200_K3",
            checkpoint=parent,lead_hours=LEADS,max_samples=24,device_name="cpu")
        old = next(x["report"] for x in original["records"] if x["variant"]==variant and
                   x["seed"]==seed and x["updates"]==200 and x["split"]=="val")
        if _report_matrix(baseline)[0] != _report_matrix(old)[0]:
            raise ValueError("original validation pairing not reproduced")
        if not np.allclose([x["mse"] for x in baseline["initializations"]],
                           [x["mse"] for x in old["initializations"]],rtol=1e-5,atol=1e-12):
            raise ValueError("original validation predictions not reproduced")
        records.append(dict(variant=variant,seed=seed,updates=200,depth=3,report=baseline))
        checkpoint, report = run_local_updates(dataset,kind=contract["kind"],
            model_config=contract["model"],data_identity=identity,output_dir=out/"training"/f"{variant}_{seed}",
            total_updates=FINAL_UPDATES,batch_size=2,accumulation=1,steps=3,seed=seed,lr=2e-4,
            process_weight=contract["process_weight"],clip=1.,device_name="cpu",bf16=False,resume=parent)
        if report["updates_this_run"] != 600 or [x["update"] for x in report["losses"]] != list(range(201,801)):
            raise ValueError("resume did not execute exactly201..800")
        resources.append(dict(variant=variant,seed=seed,parent_sha256=hashes[f"{variant}_{seed}"],
            final_checkpoint_sha256=file_sha256(checkpoint),training=report))
        for depth in (3,1):
            result = evaluate_local(val,output_dir=out/"evaluation"/f"{variant}_{seed}_800_K{depth}",
                checkpoint=checkpoint,lead_hours=LEADS,max_samples=24,device_name="cpu",reasoning_steps=depth)
            records.append(dict(variant=variant,seed=seed,updates=800,depth=depth,report=result))
        print(json.dumps(dict(variant=variant,seed=seed,extra_updates=600,validation_complete=True)),flush=True)
    if hashes != {f"{v}_{s}":file_sha256(p) for (v,s),p in parents.items()}:
        raise ValueError("original checkpoint files changed")
    result = dict(format="r7-extended-validation-result-v1",protocol=protocol,records=records,
        summary=summarize_control(records),resources=resources,restoration=restoration,
        elapsed_seconds=time.monotonic()-started,scientific_claim=False,gpu_used=False,
        test_evaluated=False,original_checkpoints_unchanged=True,
        limitations=["Small four sampled-month tile, not full seasons or final paper test",
            "800updates are not a convergence guarantee; all outcomes retained",
            "Per-seed variation is not a confidence interval over weather events",
            "Same optimizer updates do not match FLOPs; no adaptive speedup is measured"])
    _write(out/"extended_result.json",result)
    return result
