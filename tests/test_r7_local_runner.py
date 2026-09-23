import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from data.synthetic_atmos import SyntheticAtmosDataset
from training.r7_experiment import load_checkpoint,select_device,dataset_identity
from training.r7_local_runner import run_local_updates


def options(tmp_path,kind):
    model=dict(in_channels=4,history_steps=2,dim=32,depth=1,heads=4,window_size=4,dropout=.15)
    if kind=='generic':
        model['latent_tokens']=4
    if kind=='process':
        model.update(anchored_processes=2,free_processes=2)
    return dict(kind=kind,model_config=model,data_identity='synthetic-test-only',output_dir=tmp_path,
        batch_size=2,accumulation=2,steps=2,process_weight=0.,seed=42)


def nested_equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: nested_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): nested_equal(x,y)
    else:
        assert a==b


@pytest.mark.parametrize('kind',['native','generic','process'])
def test_checkpoint_resume_equals_uninterrupted(tmp_path,kind):
    ds=SyntheticAtmosDataset(length=5,hw=(8,12),channels=4)
    full,report=run_local_updates(ds,total_updates=4,**options(tmp_path/'full',kind))
    first,_=run_local_updates(ds,total_updates=2,**options(tmp_path/'split',kind))
    resumed,rreport=run_local_updates(ds,total_updates=4,resume=first,**options(tmp_path/'split',kind))
    a,b=load_checkpoint(full),load_checkpoint(resumed)
    nested_equal(a['model'],b['model'])
    nested_equal(a['optimizer'],b['optimizer'])
    nested_equal(a['rng'],b['rng'])
    assert a['cursor']==b['cursor'] and a['epoch']==b['epoch']
    assert [x['samples'] for x in report['losses']]==[4,1,4,1]
    assert report['peak_allocated_bytes'] is None
    assert rreport['updates_this_run']==2
    before=first.read_bytes()
    with pytest.raises(FileExistsError):
        run_local_updates(ds,total_updates=2,**options(tmp_path/'split',kind))
    assert first.read_bytes()==before
    changed=options(tmp_path/'wrong',kind)
    changed['data_identity']='tampered'
    with pytest.raises(ValueError,match='identity'):
        run_local_updates(ds,total_updates=5,resume=first,**changed)


def test_cuda_unavailable_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='unavailable'):
        select_device('cuda')


def test_process_labels_required_unless_explicit_ablation(tmp_path):
    cfg=options(tmp_path/'labels','process')
    cfg['process_weight']=.1
    with pytest.raises(ValueError,match='diagnostic'):
        run_local_updates(SyntheticAtmosDataset(length=5,hw=(8,12),channels=4),total_updates=1,**cfg)


def test_local_zarr_runner_and_identity(tmp_path):
    from test_r7_storage_safety import build
    paths=build(tmp_path/'data')
    identity,ds=dataset_identity(paths['train'])
    model=dict(in_channels=1,dim=16,depth=1,heads=4,window_size=2)
    checkpoint,_=run_local_updates(ds,kind='native',model_config=model,data_identity=identity,
        output_dir=tmp_path/'run',total_updates=1,process_weight=0.)
    assert load_checkpoint(checkpoint)['contract']['data_identity']==identity
    import zarr
    root=zarr.open_group(str(tmp_path/'data'/'output'),mode='r+')
    root['normalization_mean'][:]=root['normalization_mean'][:]+1
    new_identity,_=dataset_identity(paths['train'])
    assert new_identity!=identity


def test_local_cli_smoke(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'train_r7_local.py','--config','configs/r7_local_smoke.yaml',
        '--synthetic','--out',str(tmp_path/'cli'),'--updates','1'],cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'cli'/'update_0000001.pt').exists()
