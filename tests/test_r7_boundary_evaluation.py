import csv
from pathlib import Path
import subprocess
import sys
import pytest
from test_r7_storage_safety import build
from test_r7_calibration_runner import parent_model
from training.r7_evaluate import evaluate_local


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize('use_model',[False,True])
def test_same_forecasts_full_scores_and_boundary_report(tmp_path,use_model):
    if use_model:
        parent,paths,_,_=parent_model(tmp_path)
    else:
        parent=None
        paths=build(tmp_path/'data')
    common=dict(checkpoint=parent,lead_hours=(6,12),max_samples=2)
    plain=evaluate_local(paths['test'],output_dir=tmp_path/'plain',**common)
    edge=evaluate_local(paths['test'],output_dir=tmp_path/'edge',boundary_margins=(1,),**common)
    assert edge['initializations']==plain['initializations']
    assert (tmp_path/'plain'/'rmse.csv').read_bytes()==(tmp_path/'edge'/'rmse.csv').read_bytes()
    full=rows(tmp_path/'plain'/'rmse.csv')
    stratified=rows(tmp_path/'edge'/'boundary_rmse.csv')
    selected=[r for r in stratified if r['region']=='full']
    assert len(selected)==len(full)
    for a,b in zip(full,selected):
        assert float(a['rmse'])==pytest.approx(float(b['rmse']),rel=1e-12,abs=1e-12)
    definitions=edge['boundary_scoring']['regions']
    assert definitions[1]['full_area_fraction']+definitions[2]['full_area_fraction']==pytest.approx(1.)
    assert plain['boundary_scoring'] is None
    with pytest.raises(ValueError,match='margin'):
        evaluate_local(paths['test'],output_dir=tmp_path/'invalid',boundary_margins=(999,),**common)
    assert not (tmp_path/'invalid').exists()


def test_boundary_cli(tmp_path):
    paths=build(tmp_path/'data')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'evaluate_r7_local.py','--persistence','--manifest',str(paths['test']),
        '--out',str(tmp_path/'cli'),'--leads','6','12','--max-samples','1','--boundary-margins','1'],
        cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'cli'/'boundary_rmse.csv').exists()
