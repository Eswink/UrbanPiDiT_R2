"""Synthetic pin/receipt contract fixtures only, not fabricated real observations."""
import hashlib
import json
import h5py
import numpy as np
import pytest
from data.download.earthmover_pilot import SOURCE,SNAPSHOT,FIELDS
from data.download.seasonal_sampling import requested_times
import data.download.seasonal_pilot_replay as seasonal
import data.download.pressure_pilot_replay as january


def fixture(tmp_path,monkeypatch):
    import pandas as pd
    source,receipt=tmp_path/'synthetic.nc',tmp_path/'receipt.json'
    times=requested_times('four-season')
    coords={'latitude':np.linspace(42,39.25,12),'longitude':np.linspace(114,116.75,12)}
    rows=[]
    with h5py.File(source,'w') as f:
        f.attrs.update(source=SOURCE,snapshot_id=SNAPSHOT)
        for name,values in coords.items():f.create_dataset(name,data=values)
        t=f.create_dataset('time',data=np.asarray((times-pd.Timestamp('2018-01-01'))/pd.Timedelta(hours=1),dtype='i8'))
        t.attrs.update(units='hours since 2018-01-01 00:00:00',calendar='proleptic_gregorian')
        for variable,level,name in FIELDS:
            values=np.ones((384,12,12),dtype='f4')
            d=f.create_dataset(name,data=values)
            d.attrs.update(units='synthetic-unit',source_variable=variable,source_pressure_hpa=-1 if level is None else level)
            rows.append(dict(name=name,source_variable=variable,pressure_hpa=level,shape=[384,12,12],units='synthetic-unit',payload_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(seasonal,'PINNED_SEASONAL_SHA256',sha) # Test fixture only, not production behavior.
    r=dict(source=SOURCE,snapshot_id=SNAPSHOT,source_netcdf_sha256=sha,source_netcdf_bytes=source.stat().st_size,
        source_is_real_reanalysis=True,scientific_claim=False,license='CC-BY-4.0',years=[2018,2019,2020],
        times_per_year=128,sampling_profile='four-season',months=[1,4,7,9],selected_times_utc=[t.isoformat() for t in times],
        coordinates={k:v.tolist() for k,v in coords.items()},variables=rows)
    receipt.write_text(json.dumps(r))
    return source,receipt,r


def test_seasonal_copy_is_exact_and_original_pin_is_unchanged(tmp_path,monkeypatch):
    source,receipt,_=fixture(tmp_path,monkeypatch)
    dest=tmp_path/'out'/'same.nc'
    _,report=seasonal.copy_verified_seasonal_pilot(source,receipt,dest,dest.with_suffix('.json'))
    assert dest.read_bytes()==source.read_bytes() and report['replay']['source_network_requests']==0
    assert january.PINNED_SOURCE_SHA256=='13fb72807d3f6988b0fcf2b6b2f130f890207242916018556fb85553686b9a12'
    with pytest.raises(ValueError):january.verify_pressure_pilot(source,receipt)
    with pytest.raises(FileExistsError):seasonal.copy_verified_seasonal_pilot(source,receipt,dest,dest.with_suffix('.json'))


@pytest.mark.parametrize('key',['months','sampling_profile','times_per_year','selected_times_utc','payload','bytes'])
def test_seasonal_receipt_corruption_cannot_create_output(tmp_path,monkeypatch,key):
    source,receipt,r=fixture(tmp_path,monkeypatch)
    if key=='payload':r['variables'][0]['payload_sha256']='0'*64
    elif key=='bytes':r['source_netcdf_bytes']+=1
    else:r[key]=None
    receipt.write_text(json.dumps(r))
    dest=tmp_path/'out'/'same.nc'
    with pytest.raises(ValueError):seasonal.copy_verified_seasonal_pilot(source,receipt,dest,dest.with_suffix('.json'))
    assert not dest.parent.exists()
