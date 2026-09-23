from copy import deepcopy
from itertools import product
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import pytest
from data.explicit_initializations import select_explicit_initializations
from training.r7_profile_comparison import compare_common_cases,compare_profile_records
from training.r7_extended_control import VARIANTS,SEEDS,LEADS


def report(ids=(0,1,2),errors=(1.,2.,3.)):
    cases=[]
    for i,e in zip(ids,errors):
        t=datetime(2019,1,1)+timedelta(hours=6*i)
        cases.append(dict(init_time=t.isoformat(),
            valid_times=[(t+timedelta(hours=h)).isoformat() for h in LEADS],
            mse=[[e**2] for _ in LEADS]))
    return dict(channels=["t2m"],units=["K"],lead_hours=list(LEADS),step_hours=6,split="val",
        initializations=cases,n_evaluated=len(cases),evaluation_manifest_sha256="manifest",
        training_identity="training")


def test_intersection_is_timestamp_only_and_recomputes_squared_errors():
    a=report((0,1,2),(100.,2.,4.))
    b=report((3,2,1),(1000.,1.,3.))
    b["evaluation_manifest_sha256"]="different";b["training_identity"]="other"
    r=compare_common_cases(a,b)
    assert r["n_common"]==2 and r["n_left"]==r["n_right"]==3
    assert r["left_rmse"][0][0]==pytest.approx(np.sqrt(10))
    assert r["right_rmse"][0][0]==pytest.approx(np.sqrt(5))
    assert len(r["left_only"])==len(r["right_only"])==1
    assert r["left_manifest_sha256"]!=r["right_manifest_sha256"]


@pytest.mark.parametrize("bad",["duplicate","valid_time","empty","unit","nan","test","negative","timezone"])
def test_bad_comparisons(bad):
    a,b=report(),report()
    if bad=="duplicate":b["initializations"][1]=deepcopy(b["initializations"][0])
    elif bad=="valid_time":b["initializations"][0]["valid_times"][0]="2019-01-01T00:00:00"
    elif bad=="empty":b=report((4,5,6))
    elif bad=="unit":b["units"]=["normalized"]
    elif bad=="nan":b["initializations"][0]["mse"][0][0]=float("nan")
    elif bad=="test":b["split"]="test"
    elif bad=="negative":b["initializations"][0]["mse"][0][0]=-1
    elif bad=="timezone":b["initializations"][0]["init_time"]+="Z"
    with pytest.raises(ValueError):compare_common_cases(a,b)


def records():
    return [dict(variant=v,seed=s,depth=k,updates=800,report=report())
            for v,s,k in product(VARIANTS,SEEDS,(1,3))]


def test_full_record_groups_and_missing_seed_guards():
    a,b=records(),records()
    r=compare_profile_records(a,b)
    assert len(r["pairs"])==18 and len(r["summary"])==24
    assert all(x["paired_seed_rmse_difference_mean"]==0 for x in r["summary"])
    for mode in ("missing","duplicate","manifest","endpoint"):
        wrong=deepcopy(b)
        if mode=="missing":wrong.pop()
        elif mode=="duplicate":wrong.append(deepcopy(wrong[0]))
        elif mode=="manifest":wrong[0]["report"]["evaluation_manifest_sha256"]="other"
        else:wrong[0]["updates"]=200
        with pytest.raises(ValueError):compare_profile_records(a,wrong)


def explicit_fixture():
    t=pd.date_range("2019-01-01",periods=40,freq="6h")
    records=[dict(split="val",init_time=t[i].isoformat(),
        history_times=[t[i-1].isoformat(),t[i].isoformat()],target_time=t[i+1].isoformat(),
        atmos_target="poison-not-a-numeric-field") for i in range(1,27)]
    return t,records


def test_explicit_selection_preserves_declared_ids_not_first_n():
    times,records=explicit_fixture()
    desired=[records[9]["init_time"],records[3]["init_time"]]
    got=select_explicit_initializations(records,times,desired)
    assert [r["init_time"] for r in got]==sorted(desired)
    assert all(r["atmos_target"]=="poison-not-a-numeric-field" for r in got)


@pytest.mark.parametrize("bad",["missing","duplicate","gap","bad_history","train","bad_target","timezone"])
def test_explicit_failure_is_not_a_silent_subset(bad):
    times,records=explicit_fixture()
    desired=[records[0]["init_time"]]
    if bad=="missing":desired=["2019-02-01T00:00:00"]
    elif bad=="duplicate":desired*=2
    elif bad=="gap":times=times.delete(3)
    elif bad=="bad_history":records[0]["history_times"]=["2018-12-31T18:00:00",records[0]["init_time"]]
    elif bad=="train":
        for r in records:r["split"]="train"
    elif bad=="bad_target":records[0]["target_time"]=records[0]["init_time"]
    elif bad=="timezone":desired[0]+="Z"
    with pytest.raises(ValueError):select_explicit_initializations(records,times,desired)
