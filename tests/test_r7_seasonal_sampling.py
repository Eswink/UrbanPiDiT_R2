from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from test_r7_earthmover_plan import Array, root_fixture
from data.download.seasonal_sampling import requested_times, select_balanced_records
from data.download.earthmover_pilot import extract_dataset


def seasonal_root():
    root = root_fixture()
    times = requested_times('four-season')
    hours = np.asarray((times-pd.Timestamp('1940-01-01'))/pd.Timedelta(hours=1),dtype='i8')
    for group in root.values():
        group['valid_time'] = Array(hours,(384,),attrs={'units':'hours since 1940-01-01 00:00:00','calendar':'proleptic_gregorian'})
        for name, array in list(group.items()):
            if len(array.shape) < 3: continue
            shape = (384,) + array.shape[1:]
            group[name] = Array(np.arange(np.prod(shape),dtype='f4').reshape(shape),array.chunks,array.metadata.dimension_names,array.attrs)
    return root


def test_named_sampling_exact_and_same_upstream_time_chunks():
    jan, four = requested_times(), requested_times('four-season')
    assert len(jan) == 96 and len(four) == 384
    assert set(four.month) == {1,4,7,9} and four.is_unique and four.is_monotonic_increasing
    def chunk_ids(times):
        return set(np.asarray((times-pd.Timestamp('1940-01-01'))/pd.Timedelta(hours=1),dtype='i8')//8736)
    assert chunk_ids(jan) == chunk_ids(four) == {78,79,80}
    assert all(t in four for t in jan)


@pytest.mark.parametrize('profile',['all',None,True,'october'])
def test_unknown_profiles_rejected(profile):
    with pytest.raises(ValueError): requested_times(profile)


def test_seasonal_values_exact_and_original_january_unchanged():
    root = seasonal_root()
    ds, receipt = extract_dataset(root,sampling='four-season')
    assert ds.sizes['time'] == 384 and receipt['times_per_year'] == 128
    np.testing.assert_array_equal(ds.t850.values,root['pressure/temporal']['t'].x[:,0])
    jan, original = extract_dataset(root)
    np.testing.assert_array_equal(jan.time.values,requested_times().values)
    indices = requested_times('four-season').get_indexer(requested_times())
    np.testing.assert_array_equal(jan.t850.values,ds.t850.values[indices])
    assert original['times_per_year'] == 32


def records_and_times():
    times = requested_times('four-season')
    records = []
    for month in (1,4,7,9):
        local = times[(times.year == 2020)&(times.month == month)]
        for i in range(1,len(local)-1):
            records.append(dict(init_time=local[i].isoformat(),split='test',
                history_times=[local[i-1].isoformat(),local[i].isoformat()]))
    return records, times


def test_balanced_selection_not_first_january_only_or_target_driven():
    records,times = records_and_times()
    original = deepcopy(records)
    selected = select_balanced_records(list(reversed(records)),times)
    assert len(selected) == 24 and records == original
    for month in (1,4,7,9):
        assert sum(pd.Timestamp(r['init_time']).month == month for r in selected) == 6
    assert selected == sorted(selected,key=lambda r:r['init_time'])


def test_no_gap_bridging_and_no_missing_season_fallback():
    records,times = records_and_times()
    times = times[~((times.year==2020)&(times.month==7)&(times.day==2))]
    selected = select_balanced_records(records,times)
    july = [pd.Timestamp(r['init_time']) for r in selected if pd.Timestamp(r['init_time']).month == 7]
    assert all(t.day >= 3 for t in july)  # Later complete trajectories are allowed.
    times = times[~((times.year==2020)&(times.month==7)&(times.day < 7))]
    with pytest.raises(ValueError,match='month 7'):
        select_balanced_records(records,times)
    records,times = records_and_times()
    for r in records: r['split']='train'
    with pytest.raises(ValueError,match='held-out'): select_balanced_records(records,times)


def test_available_time_unit_is_explicit_not_pandas_default():
    records,times = records_and_times()
    assert select_balanced_records(records,times.as_unit('us')) == select_balanced_records(records,times.as_unit('ns'))
