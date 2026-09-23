from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from training.r7_paired_comparison import paired_rmse_bootstrap,compare_files


def report(scale=1.):
    rows=[]
    for i,t in enumerate(pd.date_range('2020-01-01',periods=64,freq='6h')):
        rows.append({'init_time':t.isoformat(),'valid_times':[(t+pd.Timedelta(hours=h)).isoformat() for h in [6,12]],
            'mse':(np.array([[1.,2.],[3.,4.]])*(1+i/64)*scale).tolist()})
    return {'evaluation_manifest_sha256':'same-synthetic-manifest','channels':['t','u'],'units':['K','m/s'],
        'lead_hours':[6,12],'step_hours':6,'split':'test','n_evaluated':64,'initializations':rows}


def test_identity_direction_reorder_and_reproducibility():
    a=report()
    zero=paired_rmse_bootstrap(a,a,replicates=64)
    assert all(r['rmse_A_minus_B']==r['interval_lower']==r['interval_upper']==0 for r in zero['rows'])
    b=report(4.)
    b['initializations'].reverse()
    first=paired_rmse_bootstrap(a,b,replicates=64,seed=7)
    again=paired_rmse_bootstrap(a,b,replicates=64,seed=7)
    assert first==again
    assert all(r['rmse_A_minus_B']<0 and r['interval_upper']<0 for r in first['rows'])


@pytest.mark.parametrize('change',['missing','duplicate','unit','lead','nan','negative','manifest'])
def test_invalid_pairs_are_rejected(change):
    a,b=report(),report()
    if change=='missing': b['initializations'].pop();b['n_evaluated']-=1
    if change=='duplicate': b['initializations'][1]=deepcopy(b['initializations'][0])
    if change=='unit': b['units'][0]='C'
    if change=='lead': b['initializations'][0]['valid_times'][0]='2020-01-02T00:00:00'
    if change=='nan': b['initializations'][0]['mse'][0][0]=float('nan')
    if change=='negative': b['initializations'][0]['mse'][0][0]=-1
    if change=='manifest': b['evaluation_manifest_sha256']='different'
    with pytest.raises(ValueError): paired_rmse_bootstrap(a,b,replicates=20)


def test_file_outputs_are_exclusive(tmp_path):
    a,b=tmp_path/'a.json',tmp_path/'b.json'
    a.write_text(json.dumps(report()));b.write_text(json.dumps(report(2)))
    out=tmp_path/'comparison.json'
    result=compare_files(a,b,out=out,replicates=32)
    assert result['n_initializations']==64 and len(result['input_sha256'])==2
    before=out.read_bytes()
    with pytest.raises(FileExistsError): compare_files(a,b,out=out,replicates=32)
    assert before==out.read_bytes()
