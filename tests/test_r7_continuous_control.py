"""Synthetic integrity fixtures only; actual source is verified separately in the workflow."""
from copy import deepcopy
from itertools import product
import hashlib
import json
import h5py
import numpy as np
import pandas as pd
import pytest
import data.download.continuous_pilot_replay as replay
from data.download.earthmover_pilot import FIELDS,SOURCE,SNAPSHOT
from data.download.seasonal_sampling import requested_times
from data.preprocess.r7_preflight import canonical_unit
from training.r7_continuous_control import continuous_protocol,summarize_continuous
from training.r7_extended_control import VARIANTS,SEEDS
from test_r7_extended_control import report as mock_report


@pytest.fixture
def synthetic_pair(tmp_path,monkeypatch):
    source,receipt=tmp_path/'SYNTHETIC.nc',tmp_path/'SYNTHETIC.json'
    times=requested_times('continuous-250d')
    coords=dict(latitude=np.linspace(42,39.25,12),longitude=np.linspace(114,116.75,12))
    rows=[]
    with h5py.File(source,'w') as f:
        f.attrs.update(source=SOURCE,snapshot_id=SNAPSHOT)
        for n,a in coords.items():f.create_dataset(n,data=a)
        t=f.create_dataset('time',data=np.asarray((times-pd.Timestamp('2018-01-01'))/pd.Timedelta(hours=1),dtype='i8'))
        t.attrs.update(units='hours since 2018-01-01 00:00:00',calendar='proleptic_gregorian')
        for variable,level,name in FIELDS:
            a=np.ones((3000,12,12),dtype='f4')
            units=canonical_unit(name)
            d=f.create_dataset(name,data=a)
            d.attrs.update(units=units,source_variable=variable,source_pressure_hpa=-1 if level is None else level)
            rows.append(dict(name=name,source_variable=variable,pressure_hpa=level,
                shape=[3000,12,12],units=units,payload_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    # Only the synthetic UNIT TEST monkeypatches the pin, never production/replay.
    monkeypatch.setattr(replay,'PINNED_CONTINUOUS_SHA256',sha)
    r=dict(source=SOURCE,snapshot_id=SNAPSHOT,source_netcdf_sha256=sha,source_netcdf_bytes=source.stat().st_size,
        source_is_real_reanalysis=True,scientific_claim=False,license='CC-BY-4.0',
        years=[2018,2019,2020],times_per_year=1000,sampling_profile='continuous-250d',months=list(range(1,10)),
        selected_times_utc=[t.isoformat() for t in times],variables=rows,
        coordinates={k:v.tolist() for k,v in coords.items()},same_time_chunks_as_four_season_verified=True,
        decoded_budget_bytes=192*2**20,decoded_charged_bytes=180142968,cropped_state_float32_bytes=19008000,
        network_body_bytes=None,time_coverage_by_year={str(y):dict(first=times[times.year==y][0].isoformat(),
            last=times[times.year==y][-1].isoformat(),count=1000) for y in (2018,2019,2020)})
    receipt.write_text(json.dumps(r))
    return source,receipt,r


def test_copy_exact_no_overwrite_old_pins_still_strict(synthetic_pair,tmp_path):
    from data.download.seasonal_pilot_replay import verify_seasonal_pilot
    source,receipt,_=synthetic_pair
    out=tmp_path/'new'/'source.nc'
    _,r=replay.copy_verified_continuous_pilot(source,receipt,out,out.with_suffix('.json'))
    assert out.read_bytes()==source.read_bytes()
    assert r['replay']['source_network_requests']==0
    with pytest.raises(FileExistsError):
        replay.copy_verified_continuous_pilot(source,receipt,out,out.with_suffix('.json'))
    with pytest.raises(ValueError):verify_seasonal_pilot(source,receipt)


@pytest.mark.parametrize('mutation',['hash','time','payload','coverage','budget','units','level','bytes'])
def test_corrupt_receipt_no_output(synthetic_pair,tmp_path,mutation):
    source,receipt,r=synthetic_pair
    if mutation=='hash':r['source_netcdf_sha256']='0'*64
    elif mutation=='time':r['selected_times_utc'][0]='2018-01-02T00:00:00'
    elif mutation=='payload':r['variables'][0]['payload_sha256']='0'*64
    elif mutation=='coverage':r['time_coverage_by_year']['2020']['count']=999
    elif mutation=='budget':r['decoded_budget_bytes']*=2
    elif mutation=='units':r['variables'][0]['units']='degC'
    elif mutation=='level':r['variables'][4]['pressure_hpa']=700
    elif mutation=='bytes':r['source_netcdf_bytes']+=1
    receipt.write_text(json.dumps(r))
    out=tmp_path/'not_created'/'source.nc'
    with pytest.raises(ValueError):
        replay.copy_verified_continuous_pilot(source,receipt,out,out.with_suffix('.json'))
    assert not out.parent.exists()


def test_corrupted_file_self_written_receipt_not_sufficient(synthetic_pair):
    source,receipt,r=synthetic_pair
    with source.open('ab') as f:f.write(b'corruption')
    r['source_netcdf_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    r['source_netcdf_bytes']=source.stat().st_size
    receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='pinned'):replay.verify_continuous_pilot(source,receipt)


def test_protocol_frozen_and_source_bound():
    from training.r7_experiment import canonical_digest
    r=continuous_protocol(replay.PINNED_CONTINUOUS_SHA256)
    digest=r.pop('protocol_sha256')
    assert canonical_digest(r)==digest and r['updates']==800 and r['train_windows']==998
    assert not r['test_evaluated'] and not r['controller_trained']
    with pytest.raises(ValueError):continuous_protocol('0'*64)


def records():
    return [dict(variant=v,seed=s,depth=k,updates=800,report=mock_report())
            for v,s,k in product(VARIANTS,SEEDS,(1,3))]


def test_complete_summary_and_pairing():
    rows=summarize_continuous(records(),mock_report())
    assert len(rows)==3*2*4*2
    assert all(r['seeds']==3 and r['seed_rmse_sd']==0 for r in rows)


@pytest.mark.parametrize('bad',['missing','duplicate','test','unpaired','nonfinite','endpoint'])
def test_bad_study_records(bad):
    rows=records()
    if bad=='missing':rows.pop()
    elif bad=='duplicate':rows.append(deepcopy(rows[0]))
    elif bad=='test':rows[0]['report']['split']='test'
    elif bad=='unpaired':rows[0]['report']['evaluation_manifest_sha256']='changed'
    elif bad=='nonfinite':rows[0]['report']['initializations'][0]['mse'][0][0]=float('nan')
    elif bad=='endpoint':rows[0]['updates']=200
    with pytest.raises(ValueError):summarize_continuous(rows,mock_report())
