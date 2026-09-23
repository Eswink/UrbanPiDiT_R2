import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from data.r7_evaluation import ZarrRolloutDataset,fit_training_climatology,normalized_climatology
from training.r7_acc import RolloutACCAccumulator
from training.r7_rollout_metrics import RolloutRMSEAccumulator
from training.r7_evaluate import evaluate_local
from training.r7_local_runner import run_local_updates
from training.r7_experiment import dataset_identity
from test_r7_storage_safety import build,fixture


def test_exact_window_times_and_missing_frames(tmp_path):
    ds=fixture()
    missing=pd.Timestamp('2019-01-01T12:00:00')
    ds=ds.sel(time=ds.time!=np.datetime64(missing))
    build(tmp_path,ds=ds)
    window=ZarrRolloutDataset(tmp_path/'output',split='val',lead_hours=(6,12,18))
    for history,targets in window.windows:
        init=window.times[history[-1]]
        expected=pd.date_range(init-pd.Timedelta(hours=6),init+pd.Timedelta(hours=18),freq='6h')
        assert missing not in expected
        assert all(t.year==2019 for t in expected)
        assert window.times[targets[-1]]==init+pd.Timedelta(hours=18)
    first=window[0]
    assert first['rollout_targets'].shape==(3,1,3,4)
    assert 'process_targets' not in first
    with pytest.raises(ValueError,match='held-out'):
        ZarrRolloutDataset(tmp_path/'output',split='train',lead_hours=(6,))


def test_climatology_uses_training_only(tmp_path):
    build(tmp_path/'a',ds=fixture(future_offset=100))
    build(tmp_path/'b',ds=fixture(future_offset=9999))
    a,b=(fit_training_climatology(tmp_path/p/'output') for p in ['a','b'])
    assert a['training_years']==[2018] and a['counts']==b['counts']
    for key in a['means']:
        np.testing.assert_array_equal(a['means'][key],b['means'][key])
    with pytest.raises(ValueError,match='missing'):
        normalized_climatology(a,['2020-07-01'],np.array([0]),np.array([1]))


def test_acc_perfect_opposite_zero_energy_and_batch_invariance():
    g=torch.Generator().manual_seed(9)
    target=torch.randn(3,2,2,3,4,generator=g)
    climate=torch.zeros_like(target)
    lat=torch.tensor([40,39.75,39.5])
    for sign in [1,-1]:
        full=RolloutACCAccumulator((6,12),('t','u'))
        chunk=RolloutACCAccumulator((6,12),('t','u'))
        full.update(sign*target,target,climate,lat)
        chunk.update(sign*target[:1],target[:1],climate[:1],lat)
        chunk.update(sign*target[1:],target[1:],climate[1:],lat)
        torch.testing.assert_close(full.compute(),torch.full((2,2),float(sign),dtype=torch.float64))
        torch.testing.assert_close(full.compute(),chunk.compute())
    zero=RolloutACCAccumulator((6,12),('t','u'))
    zero.update(climate,target,climate,lat)
    assert torch.isnan(zero.compute()).all()
    rmse=RolloutRMSEAccumulator((6,12),('t','u'))
    rmse.update(target,target,lat)
    assert torch.count_nonzero(rmse.compute())==0


def test_evaluation_checkpoint_and_cli(tmp_path,monkeypatch):
    paths=build(tmp_path/'data')
    identity,ds=dataset_identity(paths['train'])
    config=dict(in_channels=1,dim=16,depth=1,heads=4,window_size=2)
    checkpoint,_=run_local_updates(ds,kind='native',model_config=config,data_identity=identity,
        output_dir=tmp_path/'training',total_updates=1,process_weight=0.)
    import training.r7_evaluate as module
    original=module.make_model
    observed=[]
    def checked_model(*args):
        model=original(*args)
        def hook(_module,inputs):
            observed.append(set(inputs[0]))
        model.register_forward_pre_hook(hook)
        return model
    monkeypatch.setattr(module,'make_model',checked_model)
    result=evaluate_local(paths['test'],output_dir=tmp_path/'eval',checkpoint=checkpoint,lead_hours=(6,12),max_samples=2)
    assert result['n_evaluated']==2 and result['checkpoint_sha256']
    assert all(keys=={'coarse_history','lead_time_hours'} for keys in observed)
    assert (tmp_path/'eval'/'rmse.csv').is_file() and (tmp_path/'eval'/'acc.csv').is_file()
    with pytest.raises(FileExistsError):
        evaluate_local(paths['test'],output_dir=tmp_path/'eval',lead_hours=(6,12),max_samples=1)
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,'evaluate_r7_local.py','--manifest',str(paths['test']),
        '--persistence','--out',str(tmp_path/'cli'),'--leads','6','12','--max-samples','2']
    run=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads((tmp_path/'cli'/'provenance.json').read_text())['n_evaluated']==2
