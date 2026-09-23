"""Retrospective common-case audit, not an error-selected scientific benchmark."""
from __future__ import annotations
from itertools import product
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from data.explicit_initializations import _stamp
from .r7_cpu_study import _report_matrix
from .r7_extended_control import VARIANTS, SEEDS, LEADS


def index_validation_report(report):
    _report_matrix(report)  # finite, shape, count, duplicate textual IDs
    if report["split"] != "val" or report.get("step_hours", 6) != 6:
        raise ValueError("six-hour validation reports only")
    leads = report["lead_hours"]
    if not leads or any(type(h) is not int or h <= 0 or h > 72 or h % 6 for h in leads) or sorted(set(leads)) != leads:
        raise ValueError("unique increasing six-hour leads through72h required")
    result = {}
    for row in report["initializations"]:
        t = _stamp(row["init_time"])
        key = t.isoformat()
        if key in result:
            raise ValueError("duplicate equivalent initialization")
        actual = [_stamp(v) for v in row["valid_times"]]
        if actual != [t+pd.Timedelta(hours=h) for h in leads]:
            raise ValueError("valid times do not equal init plus lead")
        result[key] = row
    return result


def compare_common_cases(left, right):
    """Intersect timestamp identities ONLY; recompute RMSE from per-case MSE."""
    a,b = index_validation_report(left),index_validation_report(right)
    for key in ("channels","units","lead_hours"):
        if left[key] != right[key]:
            raise ValueError(f"unpaired metric schema: {key}")
    if any(u == "normalized" for u in left["units"]):
        raise ValueError("changed normalization requires physical-unit comparisons")
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no common initialization")
    mse_a = np.asarray([a[k]["mse"] for k in common], dtype=np.float64)
    mse_b = np.asarray([b[k]["mse"] for k in common], dtype=np.float64)
    rmse_a,rmse_b = np.sqrt(mse_a.mean(0)),np.sqrt(mse_b.mean(0))
    return dict(common_initializations=common,n_common=len(common),
        n_left=len(a),n_right=len(b),left_only=sorted(set(a)-set(b)),right_only=sorted(set(b)-set(a)),
        left_manifest_sha256=left["evaluation_manifest_sha256"],
        right_manifest_sha256=right["evaluation_manifest_sha256"],
        left_training_identity=left.get("training_identity"),right_training_identity=right.get("training_identity"),
        valid_times=[[(_stamp(k)+pd.Timedelta(hours=h)).isoformat() for h in left["lead_hours"]] for k in common],
        channels=left["channels"],units=left["units"],lead_hours=left["lead_hours"],
        left_case_mse=mse_a.tolist(),right_case_mse=mse_b.tolist(),
        left_rmse=rmse_a.tolist(),right_rmse=rmse_b.tolist(),
        rmse_difference_right_minus_left=(rmse_b-rmse_a).tolist())


def compare_profile_records(left, right):
    expected = set(product(VARIANTS,SEEDS,(1,3)))
    def keyed(records):
        found = {}
        signature = None
        for row in records:
            key = (row["variant"],row["seed"],row["depth"])
            if key not in expected or key in found or type(row.get("updates")) is not int or row["updates"] != 800:
                raise ValueError("unexpected, duplicate or wrong-budget record")
            index = index_validation_report(row["report"])
            current = (tuple(sorted(index)),row["report"]["evaluation_manifest_sha256"],
                       row["report"].get("training_identity"))
            if signature is not None and signature != current:
                raise ValueError("unpaired cases/identity within a profile")
            signature = current
            found[key] = row
        if set(found) != expected:
            raise ValueError("missing paired seeds/variants/depths")
        return found
    a,b = keyed(left),keyed(right)
    pairs,groups = [],{}
    pairing = None
    for key in sorted(expected):
        joined = compare_common_cases(a[key]["report"],b[key]["report"])
        if tuple(joined["lead_hours"]) != LEADS:
            raise ValueError("profile control uses fixed6/12/24/72h leads")
        current = (joined["common_initializations"],joined["left_only"],joined["right_only"])
        if pairing is not None and pairing != current:
            raise ValueError("cross-profile intersection differs across runs")
        pairing = current
        pairs.append(dict(variant=key[0],seed=key[1],depth=key[2],comparison=joined))
        for i,h in enumerate(LEADS):
            for j,name in enumerate(joined["channels"]):
                group = (key[0],key[2],h,name,joined["units"][j])
                groups.setdefault(group,[]).append([joined["left_rmse"][i][j],joined["right_rmse"][i][j]])
    summary = []
    for (variant,depth,h,name,unit),values in sorted(groups.items()):
        x = np.asarray(values)
        summary.append(dict(variant=variant,depth=depth,lead_hours=h,variable=name,unit=unit,seeds=len(x),
            left_seed_rmse_mean=float(x[:,0].mean()),right_seed_rmse_mean=float(x[:,1].mean()),
            paired_seed_rmse_difference_mean=float((x[:,1]-x[:,0]).mean()),
            paired_seed_rmse_difference_sd=float((x[:,1]-x[:,0]).std(ddof=1))))
    return dict(pairs=pairs,summary=summary,common_initializations=pairing[0],
                left_only=pairing[1],right_only=pairing[2])


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _model_sources(path):
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith("model/") and n.endswith(".py") and "legacy" not in n)
        if not names or len(set(names)) != len(names):
            raise ValueError("missing/duplicate model source files")
        return {name:hashlib.sha256(archive.read(name)).hexdigest() for name in names}


def compare_original_studies(left_root,right_root,output):
    """Read exact #54/#56 JSONs/code; never load a model or change a forecast."""
    left_root,right_root,output=Path(left_root),Path(right_root),Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files=[left_root/"extended_result.json",right_root/"continuous_control_result.json"]
    raw=[p.read_bytes() for p in files]
    if any(len(v)>32*2**20 for v in raw):
        raise ValueError("study JSON exceeds bounded read budget")
    left,right=[json.loads(v) for v in raw]
    if (left["format"]!="r7-extended-validation-result-v1" or right["format"]!="r7-continuous-training-result-v1"
        or left["test_evaluated"] is not False or right["test_evaluated"] is not False):
        raise ValueError("original validation-only study formats required")
    for study in (left,right):
        protocol=study["protocol"]
        if protocol["protocol_sha256"]!=_digest({k:v for k,v in protocol.items() if k!="protocol_sha256"}):
            raise ValueError("original protocol digest mismatch")
    sources=_model_sources(left_root/"code.zip")
    if sources != _model_sources(right_root/"code.zip"):
        raise ValueError("original model implementations differ")
    left_source=left["protocol"]["source"]
    right_source=right["source"]["source_netcdf_sha256"]
    if (left_source!="3b5d7df8973a46d522508a3b483a52394a07c0f9d2db6e07894adb1bcb41e5bb"
        or right_source!="0609fa38c1d88b82b985a15f93dd502c7eb7031f5d2b7452bbe971bf936dd9c1"):
        raise ValueError("original source pins mismatch")
    resources=[]
    for study,n,field in ((left,120,"final_checkpoint_sha256"),(right,998,"checkpoint_sha256")):
        found={}
        for row in study["resources"]:
            key=(row["variant"],row["seed"])
            if key not in set(product(VARIANTS,SEEDS)) or key in found:
                raise ValueError("invalid original resource ownership")
            contract=row["training"]["contract"]
            if (contract["dataset_length"]!=n or contract["seed"]!=key[1] or contract["batch_size"]!=2
                or contract["lr"]!=2e-4 or contract["steps"]!=3 or contract["device_type"]!="cpu"):
                raise ValueError("incompatible original training contract")
            found[key]=(contract,row[field])
        if set(found)!=set(product(VARIANTS,SEEDS)):
            raise ValueError("missing original run resources")
        resources.append(found)
    for key in resources[0]:
        ca,cb=resources[0][key][0],resources[1][key][0]
        ignored={"data_identity","dataset_length"}
        if {k:v for k,v in ca.items() if k not in ignored}!={k:v for k,v in cb.items() if k not in ignored}:
            raise ValueError("model/seed/optimization contract mismatch")
    selected=[]
    for study,rs in zip((left,right),resources):
        rows=[r for r in study["records"] if r["updates"]==800]
        for row in rows:
            contract,ckpt=rs[row["variant"],row["seed"]]
            if (row["report"]["training_identity"]!=contract["data_identity"] or
                row["report"]["checkpoint_sha256"]!=ckpt):
                raise ValueError("evaluation does not match original checkpoint/data")
        selected.append(rows)
    result=compare_profile_records(*selected)
    if len(result["common_initializations"])!=21 or len(result["left_only"])!=3 or len(result["right_only"])!=3:
        raise ValueError("actual original case intersection differs from issue58 predeclaration")
    result.update(format="r7-cross-profile-common-case-v1",scientific_claim=False,
        training_executed=False,new_forecasts_executed=False,test_evaluated=False,selection_uses_errors=False,
        input_json_sha256=[hashlib.sha256(v).hexdigest() for v in raw],
        input_code_sha256=[hashlib.sha256((p/"code.zip").read_bytes()).hexdigest() for p in (left_root,right_root)],
        source_pins=[left_source,right_source],identical_original_model_files_verified=sources,
        units_scope="physical per-variable units, never cross-normalization squared-error totals",
        limitations=["Retrospective timestamp intersection, not a pristine predeclared paper test",
            "Changes training coverage, normalization and sample order under800updates; not isolated causality",
            "Three correlated excluded cases per side are listed, never dropped based on error",
            "Raw truth-array equivalence between source archives is not checked by this JSON-only comparator",
            "Seed dispersion is not independent-event uncertainty or significance"])
    with output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
    return result
