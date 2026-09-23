import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from torch.utils.data import default_collate
from test_r7_calibration_runner import parent_model
from training.r7_calibration_runner import run_calibration,load_calibrated_adapter
from training.r7_experiment import make_model,load_checkpoint
from training.r7_policy_selection import run_policy_search
from training.r7_inference_profile import profile_local
from data.r7_zarr_dataset import ZarrAtmosWindowDataset


def test_matched_fixed_adaptive_profile_and_frozen_policy(tmp_path):
    parent,paths,identity,cfg=parent_model(tmp_path)
    controller,_=run_calibration(parent,paths['train'],output_dir=tmp_path/'controller',updates=1,max_steps=2)
    originals=(parent.read_bytes(),controller.read_bytes())
    common=dict(checkpoint=parent,warmup=1,repetitions=2,batch_size=1)
    fixed=profile_local(paths['test'],output=tmp_path/'fixed.json',reasoning_steps=2,**common)
    full=profile_local(paths['test'],output=tmp_path/'full.json',controller_checkpoint=controller,force_full_depth=True,**common)
    adaptive=profile_local(paths['test'],output=tmp_path/'adaptive.json',controller_checkpoint=controller,**common)
    assert fixed['input_sha256']==full['input_sha256']==adaptive['input_sha256']
    assert fixed['actual_reasoning_steps_per_sample']==full['actual_reasoning_steps_per_sample']==[[2],[2]]
    assert all(1<=r[0]<=2 for r in adaptive['actual_reasoning_steps_per_sample'])
    model=make_model('process',cfg)
    model.load_state_dict(load_checkpoint(parent)['model'])
    model.eval()
    adapter,_=load_calibrated_adapter(model,controller,parent_checkpoint=parent,training_identity=identity)
    b=default_collate([ZarrAtmosWindowDataset(paths['test'])[0]])
    inputs={k:b[k] for k in ('coarse_history','lead_time_hours')}
    with torch.no_grad():
        torch.testing.assert_close(adapter(inputs,max_steps=2,force_full_depth=True).forecast,model(inputs,reasoning_steps=2).forecast)
    run_policy_search(paths['val'],checkpoint=parent,controller_checkpoint=controller,output_dir=tmp_path/'search',
        gain_thresholds=(0.,),probability_thresholds=(.5,),lead_hours=(6,12),max_samples=1)
    selected=profile_local(paths['test'],output=tmp_path/'selected.json',controller_checkpoint=controller,
        policy_selection=tmp_path/'search'/'selection.json',**common)
    assert selected['policy_selection_sha256'] and selected['input_sha256']==fixed['input_sha256']
    assert originals==(parent.read_bytes(),controller.read_bytes())
    with pytest.raises(FileExistsError):
        profile_local(paths['test'],output=tmp_path/'fixed.json',**common)
    with pytest.raises(ValueError,match='overridden'):
        profile_local(paths['test'],output=tmp_path/'bad.json',controller_checkpoint=controller,
            policy_selection=tmp_path/'search'/'selection.json',force_full_depth=True,**common)


def test_profile_cli_and_no_cuda_fallback(tmp_path,monkeypatch):
    parent,paths,_,_=parent_model(tmp_path)
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'cli.json'
    result=subprocess.run([sys.executable,'profile_r7_inference.py','--checkpoint',str(parent),
        '--manifest',str(paths['test']),'--out',str(output),'--warmup','1','--repetitions','2','--device','cpu'],
        cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(output.read_text())['device']=='cpu'
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA requested'):
        profile_local(paths['test'],checkpoint=parent,output=tmp_path/'not-gpu.json',device_name='cuda',warmup=1,repetitions=1)
    assert not (tmp_path/'not-gpu.json').exists()
