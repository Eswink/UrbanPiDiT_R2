import json
from pathlib import Path
import subprocess
import sys
import pytest
from test_r7_calibration_runner import parent_model
from training.r7_calibration_runner import run_calibration
from training.r7_policy_selection import run_policy_search
from training.r7_evaluate import evaluate_local


def test_validation_selection_then_frozen_test_evaluation(tmp_path):
    parent,paths,_,_=parent_model(tmp_path)
    controller,_=run_calibration(parent,paths['train'],output_dir=tmp_path/'controller',updates=1,max_steps=2)
    originals=(parent.read_bytes(),controller.read_bytes())
    selection=run_policy_search(paths['val'],checkpoint=parent,controller_checkpoint=controller,
        output_dir=tmp_path/'search',gain_thresholds=(0.,),probability_thresholds=(.3,.7),
        lead_hours=(6,12),max_samples=1)
    assert len(selection['candidates'])==3
    assert selection['validation_manifest_sha256']
    selected=tmp_path/'search'/'selection.json'
    evaluated=evaluate_local(paths['test'],checkpoint=parent,controller_checkpoint=controller,
        output_dir=tmp_path/'test',lead_hours=(6,12),max_samples=1,policy_selection=selected)
    assert evaluated['halting_policy']==selection['selected_policy']
    assert evaluated['policy_selection_sha256']
    assert evaluated['evaluation_manifest_sha256']!=selection['validation_manifest_sha256']
    assert originals==(parent.read_bytes(),controller.read_bytes())
    with pytest.raises(ValueError,match='validation-only'):
        run_policy_search(paths['test'],checkpoint=parent,controller_checkpoint=controller,
            output_dir=tmp_path/'forbidden',lead_hours=(6,),max_samples=1)
    assert not (tmp_path/'forbidden').exists()
    with pytest.raises(ValueError,match='validation-only'):
        evaluate_local(paths['test'],checkpoint=parent,controller_checkpoint=controller,
            output_dir=tmp_path/'bad-override',lead_hours=(6,),max_samples=1,validation_thresholds=(0.,.2))
    with pytest.raises(ValueError,match='overridden'):
        evaluate_local(paths['test'],checkpoint=parent,controller_checkpoint=controller,
            output_dir=tmp_path/'bad-depth',policy_selection=selected,reasoning_steps=4)
    with pytest.raises(ValueError,match='metadata'):
        evaluate_local(paths['test'],checkpoint=parent,controller_checkpoint=controller,
            output_dir=tmp_path/'different-leads',policy_selection=selected,lead_hours=(6,),max_samples=1)
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'evaluate_r7_local.py','--manifest',str(paths['test']),
        '--checkpoint',str(parent),'--controller',str(controller),'--policy-selection',str(selected),
        '--leads','6','12','--max-samples','1','--out',str(tmp_path/'cli-test')],
        cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['n_evaluated']==1


def test_policy_search_cli_is_bounded_and_validation_only(tmp_path):
    parent,paths,_,_=parent_model(tmp_path)
    controller,_=run_calibration(parent,paths['train'],output_dir=tmp_path/'controller',updates=1,max_steps=2)
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'tune_r7_halting.py','--manifest',str(paths['val']),
        '--checkpoint',str(parent),'--controller',str(controller),'--gain-thresholds','0',
        '--probability-thresholds','0.5','--leads','6','12','--max-samples','1',
        '--out',str(tmp_path/'cli-search')],cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'cli-search'/'selection.json').is_file()
