from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from data.download.seasonal_sampling import requested_times, select_balanced_records
from data.download.earthmover_pilot import extract_dataset
from data.preprocess.r7_era5_zarr import _window_records
from test_r7_earthmover_plan import Array, root_fixture


def test_exact250days_leap_and_unchanged_old_profiles():
    t = requested_times('continuous-250d')
    assert len(t) == 3000 and t.is_unique and t.is_monotonic_increasing
    assert Counter(t.year) == {2018:1000,2019:1000,2020:1000}
    assert t[t.year==2018][-1] == pd.Timestamp('2018-09-07T18:00')
    assert t[t.year==2019][-1] == pd.Timestamp('2019-09-07T18:00')
    assert t[t.year==2020][-1] == pd.Timestamp('2020-09-06T18:00')
    for y in (2018,2019,2020):
        assert np.all(np.diff(t[t.year==y].asi8) == pd.Timedelta(hours=6).value)
    old = requested_times('four-season')
    def chunks(x):
        return set(np.asarray((x-pd.Timestamp('1940-01-01'))/pd.Timedelta(hours=1),dtype='i8')//8736)
    assert chunks(t) == chunks(old) == {78,79,80}
    assert len(old)==384 and len(requested_times())==96


def test_exact_windows_and_no_bridged_years():
    t=requested_times('continuous-250d')
    records=_window_records(t,split_sets={'train':{2018},'val':{2019},'test':{2020}},
        store_path=Path('cache.zarr'),manifest_dir=Path('manifests'),history_steps=2,
        history_interval_hours=6,lead_time_hours=6,sample_stride_hours=6)
    assert {k:len(v) for k,v in records.items()} == dict(train=998,val=998,test=998)
    for split, rows in records.items():
        for r in rows:
            stamps=list(r['history_times'])+[r['target_time']]
            assert len({pd.Timestamp(s).year for s in stamps})==1
    selected=select_balanced_records(records['val'],t)
    assert len(selected)==24 and Counter(pd.Timestamp(r['init_time']).month for r in selected)=={1:6,4:6,7:6,9:6}


def continuous_root():
    # Unit-test mapping only: synthetic finite fields do NOT certify real data.
    root=root_fixture()
    # Include original Sept8 references as source coordinates for actual set comparison.
    times=requested_times('continuous-250d').union(requested_times('four-season')).sort_values()
    hours=np.asarray((times-pd.Timestamp('1940-01-01'))/pd.Timedelta(hours=1),dtype='i8')
    for group in root.values():
        group['valid_time']=Array(hours,(len(hours),),attrs={'units':'hours since 1940-01-01 00:00:00','calendar':'proleptic_gregorian'})
        for name,a in list(group.items()):
            if len(a.shape)<3:continue
            shape=(len(times),)+a.shape[1:]
            # 32time-chunks can differ because reference days are sparse; use one whole-year-like chunk.
            chunk=(10000,)+a.chunks[1:]
            group[name]=Array(np.arange(np.prod(shape),dtype='f4').reshape(shape),chunk,a.metadata.dimension_names,a.attrs)
    return root,times


def test_continuous_exact_fields_under_unchanged_byte_cap():
    root,times=continuous_root()
    ds,r=extract_dataset(root,sampling='continuous-250d')
    assert r['times_per_year']==1000 and ds.sizes['time']==3000
    assert r['same_time_chunks_as_four_season_verified'] is True
    assert r['cropped_state_float32_bytes']==19008000
    assert r['decoded_budget_bytes']==192*2**20
    assert r['decoded_charged_bytes']<=r['decoded_budget_bytes']
    idx=times.get_indexer(requested_times('continuous-250d'))
    np.testing.assert_array_equal(ds.t850.values,root['pressure/temporal']['t'].x[idx,0])
    assert r['time_coverage_by_year']['2020']['last']=='2020-09-06T18:00:00'


def test_changed_source_chunk_geometry_rejected_before_weather():
    root,_=continuous_root()
    a=root['single/temporal']['t2m']
    a.chunks=(1,)+a.chunks[1:]
    with pytest.raises(ValueError,match='time-chunk'):
        extract_dataset(root,sampling='continuous-250d')
    assert not a.calls
