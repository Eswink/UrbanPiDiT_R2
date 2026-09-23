from copy import deepcopy
from itertools import product
import pytest
from training.r7_extended_control import validate_parent, summarize_control, VARIANTS, SEEDS, LEADS
from training.r7_cpu_study import model_settings
from training.r7_experiment import canonical_digest


def parent(variant="process",seed=41):
    kind,model,weight = model_settings(variant)
    c=dict(kind=kind,model=model,data_identity="pinned",batch_size=2,accumulation=1,
        steps=3,seed=seed,lr=2e-4,process_weight=weight,clip=1.,bf16=False,
        device_type="cpu",dataset_length=120,optimization="streamed-truncated",torch_version="test")
    return dict(contract=c,signature=canonical_digest(c),updates=200)


@pytest.mark.parametrize("v,s", list(product(VARIANTS,SEEDS)))
def test_parent_contract(v,s):
    assert validate_parent(parent(v,s),v,s,"pinned")["seed"] == s


@pytest.mark.parametrize("key,value", [("batch_size",True),("process_weight",0.),("seed",42),
    ("dataset_length",30),("device_type","cuda"),("steps",4),("lr",.002)])
def test_wrong_parent_rejected(key,value):
    p=parent();p["contract"][key]=value;p["signature"]=canonical_digest(p["contract"])
    with pytest.raises(ValueError):validate_parent(p,"process",41,"pinned")


def test_endpoint_and_digest_guards():
    for updates in (True,20,201,200.):
        p=parent();p["updates"]=updates
        with pytest.raises(ValueError):validate_parent(p,"process",41,"pinned")
    p=parent();p["signature"]="bad"
    with pytest.raises(ValueError,match="digest"):validate_parent(p,"process",41,"pinned")


def report(value=1.):
    from datetime import datetime,timedelta
    cases=[]
    for month in (1,4,7,9):
        for hour in (6,12,18,24,30,36):
            init=datetime(2019,month,1)+timedelta(hours=hour)
            cases.append(dict(init_time=init.isoformat(),valid_times=[(init+timedelta(hours=h)).isoformat() for h in LEADS],
                mse=[[value**2,4*value**2] for _ in LEADS]))
    return dict(channels=["t2m","u10"],units=["K","m/s"],lead_hours=list(LEADS),
        split="val",evaluation_manifest_sha256="same",initializations=cases,n_evaluated=24)


def records():
    return [dict(variant=v,seed=s,updates=e,depth=k,report=report()) for v,s in product(VARIANTS,SEEDS)
            for e,k in ((200,3),(800,3),(800,1))]


def test_complete_paired_summary():
    rows=summarize_control(records())
    assert len(rows)==3*3*4*2
    assert all(r["seeds"]==3 and r["seed_rmse_sd"]==0 for r in rows)
    assert {r["seed_rmse_mean"] for r in rows}=={1.,2.}


@pytest.mark.parametrize("problem",["missing","duplicate","test","manifest","unbalanced","nonfinite"])
def test_bad_reports(problem):
    r=records()
    if problem=="missing":r.pop()
    elif problem=="duplicate":r.append(deepcopy(r[0]))
    elif problem=="test":r[0]["report"]["split"]="test"
    elif problem=="manifest":r[0]["report"]["evaluation_manifest_sha256"]="other"
    elif problem=="unbalanced":r[0]["report"]["initializations"][0]["init_time"]="2019-02-01T00:00:00"
    elif problem=="nonfinite":r[0]["report"]["initializations"][0]["mse"][0][0]=float("nan")
    with pytest.raises(ValueError):summarize_control(r)
