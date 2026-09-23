"""Synthetic receipt guards only; real replay is a separately recorded workflow."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pandas as pd
import pytest
import data.download.pressure_pilot_replay as mod
from data.download.earthmover_pilot import FIELDS, SNAPSHOT, SOURCE


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Deliberately synthetic, NOT an alternative real-data source. Patch only the
    # test pin to exercise metadata guards; production pin remains exact #44 bytes.
    source, receipt = tmp_path/'synthetic.nc', tmp_path/'receipt.json'
    coords = {'latitude': np.linspace(42.,39.25,12), 'longitude': np.linspace(114.,116.75,12)}
    times = pd.DatetimeIndex(np.concatenate([pd.date_range(f'{y}-01-01',periods=32,freq='6h').values for y in (2018,2019,2020)]))
    rows = []
    with h5py.File(source,'w') as f:
        f.attrs.update(source=SOURCE,snapshot_id=SNAPSHOT)
        for name,values in coords.items():
            f.create_dataset(name,data=values)
        d = f.create_dataset('time',data=np.asarray((times-pd.Timestamp('2018-01-01'))/pd.Timedelta(hours=1),dtype='i8'))
        d.attrs.update(units='hours since 2018-01-01 00:00:00',calendar='proleptic_gregorian')
        for variable,level,name in FIELDS:
            values = np.ones((96,12,12),dtype='f4')
            d = f.create_dataset(name,data=values)
            d.attrs.update(units='fixture-unit',source_variable=variable,source_pressure_hpa=-1 if level is None else level)
            rows.append(dict(name=name,source_variable=variable,pressure_hpa=level,shape=[96,12,12],units='fixture-unit',payload_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(mod,'PINNED_SOURCE_SHA256',digest)
    r = dict(source=SOURCE,snapshot_id=SNAPSHOT,source_netcdf_sha256=digest,
        source_netcdf_bytes=source.stat().st_size,source_is_real_reanalysis=True,
        scientific_claim=False,license='CC-BY-4.0',years=[2018,2019,2020],times_per_year=32,
        coordinates={k:v.tolist() for k,v in coords.items()},variables=rows)
    receipt.write_text(json.dumps(r))
    return source,receipt,r


def test_verify_and_exclusive_copy(fixture,tmp_path):
    source,receipt,_ = fixture
    dest = tmp_path/'out'/'copy.nc'
    out_receipt = dest.with_suffix('.json')
    copied,r = mod.copy_verified_pressure_pilot(source,receipt,dest,out_receipt)
    assert copied.read_bytes()==source.read_bytes()
    assert r['replay']['source_network_requests']==0 and r['replay']['verified_variables']==11
    with pytest.raises(FileExistsError):
        mod.copy_verified_pressure_pilot(source,receipt,dest,out_receipt)


@pytest.mark.parametrize('field', ['hash','variables','coordinate','units','level','time','source'])
def test_inconsistent_receipt_fails_before_output(fixture,tmp_path,field):
    source,receipt,r = fixture
    if field=='hash': r['variables'][0]['payload_sha256']='0'*64
    elif field=='variables': r['variables'].reverse()
    elif field=='coordinate': r['coordinates']['latitude'][0]=0.
    elif field=='units': r['variables'][0]['units']='wrong'
    elif field=='level': r['variables'][4]['pressure_hpa']=700
    elif field=='time': r['times_per_year']=31
    else: r['source']='unrelated'
    receipt.write_text(json.dumps(r))
    dest=tmp_path/'no_output'/'copy.nc'
    with pytest.raises(ValueError):
        mod.copy_verified_pressure_pilot(source,receipt,dest,dest.with_suffix('.json'))
    assert not dest.parent.exists()


def test_self_reported_hash_cannot_bypass_pin(fixture):
    source,receipt,r = fixture
    raw=bytearray(source.read_bytes())
    raw[-1] ^= 1
    source.write_bytes(raw)
    r['source_netcdf_sha256']=hashlib.sha256(raw).hexdigest()
    receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='pinned'):
        mod.verify_pressure_pilot(source,receipt)


def test_cli_requires_source_receipt_pair_before_output(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts'/'real_r7_pressure_pilot.py'
    out=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(script),'--out',str(out),'--source','missing.nc'],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'together' in result.stderr
    assert not out.exists()
