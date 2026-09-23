import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from torch.utils.data import default_collate
from test_r7_calibration_runner import parent_model
from training.r7_experiment import make_model,load_checkpoint
from training.r7_halting import per_sample_latitude_mse
from training.r7_gain_oracle import collect_process_errors,run_oracle_diagnostic
from data.r7_zarr_dataset import ZarrAtmosWindowDataset


def test_streaming_diagnostics_match_drafts_and_hide_targets(tmp_path):
    parent,paths,_,cfg=parent_model(tmp_path)
    model=make_model('process',cfg).eval()
    model.load_state_dict(load_checkpoint(parent)['model'])
    batch=default_collate([ZarrAtmosWindowDataset(paths['val'])[0]])
    seen=[]
    hook=model.backbone.register_forward_pre_hook(lambda module,args:seen.append(set(args[0])))
    errors=collect_process_errors(model,batch,max_steps=4)
    hook.remove()
    assert seen==[{'coarse_history','lead_time_hours'}]
    with torch.no_grad():
        out=model(batch,reasoning_steps=4)
        expected=torch.stack([per_sample_latitude_mse(out.draft_forecasts[:,k],batch['atmos_target'],batch['latitude']) for k in range(1,5)],1)
    torch.testing.assert_close(errors,expected)
    assert not errors.requires_grad
    with pytest.raises(ValueError,match='capped'):
        collect_process_errors(model,batch,max_steps=17)


def test_validation_oracle_cli_and_no_overwrite(tmp_path):
    parent,paths,_,_=parent_model(tmp_path)
    before=parent.read_bytes()
    report=run_oracle_diagnostic(paths['val'],checkpoint=parent,output=tmp_path/'oracle.json',max_steps=3,max_samples=2,step_cost=.01)
    assert report['n_cases']==2 and report['deployable'] is False and report['target_labels_used'] is True
    assert len(report['errors_by_initialization'])==2 and report['forecaster_unchanged'] is True
    assert before==parent.read_bytes()
    with pytest.raises(FileExistsError):
        run_oracle_diagnostic(paths['val'],checkpoint=parent,output=tmp_path/'oracle.json',max_steps=3,max_samples=2)
    with pytest.raises(ValueError,match='validation-only'):
        run_oracle_diagnostic(paths['test'],checkpoint=parent,output=tmp_path/'forbidden.json',max_steps=3,max_samples=2)
    assert not (tmp_path/'forbidden.json').exists()
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'diagnose_r7_gain.py','--manifest',str(paths['val']),
        '--checkpoint',str(parent),'--out',str(tmp_path/'cli.json'),'--max-steps','3','--max-samples','1'],
        cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['deployable'] is False
