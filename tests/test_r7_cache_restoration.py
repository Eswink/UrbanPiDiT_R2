import json
from pathlib import Path
import pytest
from data.restore_pilot_cache import restoration_config,restore_pilot_cache
from data.download.earthmover_pilot import FIELDS


def metadata():
    return dict(schema_version=1,time_unit='ns',channels=[n for _,_,n in FIELDS],
        shape=[384,11,12,12],chunks=[8,11,12,12],store_path='../cache.zarr',
        history_steps=2,history_interval_hours=6,lead_time_hours=6,sample_stride_hours=6,
        split_years=dict(train=[2018],val=[2019],test=[2020]),normalization_years=[2018],
        samples_by_split=dict(train=120,val=120,test=120),native_grid_spacing_deg=.25,
        physical_units_retained=True,spatial_resampling=False,process_diagnostics_enabled=True,
        source='/historical/provenance/NOT_A_PATH_TO_READ.nc')


def test_historical_provenance_is_not_a_path_lookup():
    cfg=restoration_config(metadata())
    assert cfg['time_chunk']==8 and cfg['compute_process_targets']
    assert 'source' not in cfg


@pytest.mark.parametrize('key,value',[('store_path','../../other.zarr'),('shape',[96,11,12,12]),
    ('chunks',[16,11,12,12]),('history_interval_hours',12),('normalization_years',[2018,2019]),
    ('source','bad\x00path'),('process_diagnostics_enabled',False),('schema_version',True)])
def test_recovery_never_accepts_changed_layout_or_normalization(key,value):
    m=metadata();m[key]=value
    with pytest.raises(ValueError):restoration_config(m)


def test_corrupt_source_fails_before_output_creation(tmp_path):
    raw=tmp_path/'bad.nc';raw.write_bytes(b'not the pinned data')
    receipt=tmp_path/'receipt.json';receipt.write_text('{}')
    output=tmp_path/'rebuilt'
    with pytest.raises(ValueError):restore_pilot_cache(raw,receipt,tmp_path/'manifests',tmp_path/'checkpoint',output)
    assert not output.exists()


def test_constraint_diagnosis_does_not_retune_rejected_candidates():
    import copy
    from scripts.diagnose_r7_restored_pilot import candidate_failures
    from training.r7_policy_selection import digest
    selection=dict(channels=['t2m','mslp'],units=['K','Pa'],lead_hours=[6],relative_rmse_tolerance=.01,
        candidates=[dict(policy=dict(force_full_depth=True),rmse=[[1.,100.]],feasible=True,mean_cumulative_reasoning_steps=3.),
                    dict(policy=dict(force_full_depth=False),rmse=[[1.02,99.]],feasible=False,mean_cumulative_reasoning_steps=1.)])
    selection['signature']=digest(selection)
    original=copy.deepcopy(selection)
    result=candidate_failures(selection)
    assert result[1]['failed_pairs']==1 and result[1]['failures'][0]['variable']=='t2m'
    assert selection==original
    selection['candidates'][1]['feasible']=True
    selection['signature']=digest({k:v for k,v in selection.items() if k!='signature'})
    with pytest.raises(ValueError,match='feasibility'):candidate_failures(selection)
