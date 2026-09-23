"""Actual tiny ERA5 grid fixture, not generated weather. No network access."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
import pandas as pd
import xarray as xr
from data.preprocess.r7_preflight import prepare_local
from training.r7_experiment import dataset_identity
from training.r7_local_runner import run_local_updates
from training.r7_evaluate import evaluate_local


def test_verified_real_fixture_replays_pipeline_offline(tmp_path,monkeypatch):
    def no_network(*args,**kwargs):
        raise AssertionError('network is forbidden in pinned-fixture regression')
    monkeypatch.setattr(socket,'create_connection',no_network)
    path=Path(__file__).parent/'fixtures'/'r7_arco_t2m.json'
    obj=json.loads(path.read_text())
    data=base64.b64decode(obj['data_base64'],validate=True)
    assert obj['data_sha256']=='5f859932025f2fa889928991225f0310bfa75853e606cf7819f1b684151ffef1'
    assert hashlib.sha256(data).hexdigest()==obj['data_sha256']
    assert obj['scientific_training_ready'] is False
    assert obj['provenance']['artifact_sha256']=='e7f63e7972a95e3a094662b41adbd064ab67755b5f11552ed58fcddf722eacc1'
    values=np.frombuffer(data,dtype=obj['dtype']).reshape(obj['shape'])
    assert values.shape==(9,5,7)
    ds=xr.Dataset({'2m_temperature':(('time','latitude','longitude'),values)},
        coords={'time':pd.DatetimeIndex(obj['timestamps']),'latitude':obj['latitude'],'longitude':obj['longitude']})
    ds['2m_temperature'].attrs={'units':'K','source':obj['provenance']['source']}
    source=tmp_path/'source.nc'
    ds.to_netcdf(source,engine='h5netcdf')
    cfg={'channels':[{'variable':'2m_temperature','name':'t2m'}],
        'split_years':{'train':[2018],'val':[2019],'test':[2020]}}
    report=prepare_local(source,cfg,write=True,store_path=tmp_path/'store.zarr',
        manifest_dir=tmp_path/'manifest',max_raw_gib=.01)
    assert report['windows']=={'train':1,'val':1,'test':1}
    identity,train=dataset_identity(tmp_path/'manifest'/'train.jsonl')
    checkpoint,run=run_local_updates(train,kind='native',model_config={'in_channels':1,'dim':16,'depth':1,'heads':4,'window_size':2},
        data_identity=identity,output_dir=tmp_path/'train',total_updates=1,process_weight=0.)
    assert run['updates_this_run']==1
    results=evaluate_local(tmp_path/'manifest'/'test.jsonl',checkpoint=checkpoint,
        output_dir=tmp_path/'eval',lead_hours=(6,),max_samples=1)
    assert results['n_evaluated']==1 and results['scientific_claim'] is False
    assert np.isfinite(results['initializations'][0]['mse']).all()
