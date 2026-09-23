import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from test_r7_storage_safety import build
from training.r7_experiment import dataset_identity,load_checkpoint,make_model
from training.r7_local_runner import run_local_updates
from training.r7_calibration_runner import run_calibration,load_calibrated_adapter,state_digest
from training.r7_evaluate import evaluate_local


def parent_model(tmp_path):
    paths=build(tmp_path/'data')
    identity,ds=dataset_identity(paths['train'])
    cfg=dict(in_channels=1,dim=16,depth=1,heads=4,window_size=2,anchored_processes=2,free_processes=2)
    path,_=run_local_updates(ds,kind='process',model_config=cfg,data_identity=identity,
        output_dir=tmp_path/'train',total_updates=1,process_weight=0.,steps=2)
    return path,paths,identity,cfg


def test_calibration_and_adaptive_rollout_pipeline(tmp_path):
    parent,paths,identity,cfg=parent_model(tmp_path)
    original=parent.read_bytes()
    controller,report=run_calibration(parent,paths['train'],output_dir=tmp_path/'controller',updates=2,max_steps=3)
    assert report['forecaster_unchanged'] and report['training_only']
    assert parent.read_bytes()==original
    forecaster=make_model('process',cfg)
    forecaster.load_state_dict(load_checkpoint(parent)['model'])
    before=state_digest(forecaster)
    adapter,policy=load_calibrated_adapter(forecaster,controller,parent_checkpoint=parent,training_identity=identity)
    assert adapter.controller.optimizer_updates.item()==2
    assert state_digest(adapter.forecaster)==before
    adaptive=evaluate_local(paths['test'],output_dir=tmp_path/'adaptive',checkpoint=parent,
        controller_checkpoint=controller,lead_hours=(6,12),max_samples=2)
    full=evaluate_local(paths['test'],output_dir=tmp_path/'full',checkpoint=parent,
        controller_checkpoint=controller,lead_hours=(6,12),max_samples=2,force_full_depth=True)
    assert [r['init_time'] for r in adaptive['initializations']]==[r['init_time'] for r in full['initializations']]
    assert all(r['cumulative_reasoning_steps']==[3,6] for r in full['initializations'])
    assert all(1<=r['cumulative_reasoning_steps'][0]<=3 for r in adaptive['initializations'])
    with pytest.raises(ValueError,match='identity'):
        load_calibrated_adapter(forecaster,controller,parent_checkpoint=parent,training_identity='wrong')
    bad=tmp_path/'other-parent.pt'
    bad.write_bytes(b'not the parent')
    with pytest.raises(ValueError,match='parent'):
        load_calibrated_adapter(forecaster,controller,parent_checkpoint=bad,training_identity=identity)
    for split in ['val','test']:
        with pytest.raises(ValueError,match='training-only'):
            run_calibration(parent,paths[split],output_dir=tmp_path/split,updates=1,max_steps=2)


def test_calibration_cli_smoke(tmp_path):
    parent,paths,_,_=parent_model(tmp_path)
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'calibrate_r7_local.py','--checkpoint',str(parent),
        '--train-manifest',str(paths['train']),'--out',str(tmp_path/'cli'),'--updates','1','--max-steps','2'],
        cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['forecaster_unchanged'] is True
    result=subprocess.run([sys.executable,'evaluate_r7_local.py','--checkpoint',str(parent),
        '--controller',str(tmp_path/'cli'/'controller.pt'),'--manifest',str(paths['test']),
        '--out',str(tmp_path/'eval-cli'),'--leads','6','12','--max-samples','1'],
        cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
