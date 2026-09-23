from types import SimpleNamespace
import numpy as np
import pytest
from data.download.earthmover_pilot import DecodedBudget, bounded_selection, selection_plan, extract_dataset, FIELDS


class Array:
    def __init__(self, x, chunks, dims=(), attrs=None):
        self.x, self.shape, self.dtype, self.chunks = x, x.shape, x.dtype, chunks
        self.shards = None
        self.calls = []
        self.metadata = SimpleNamespace(dimension_names=dims)
        self.attrs = attrs or {}
    def __getitem__(self, item):
        self.calls.append(item)
        return self.x[item]


def test_reads_unique_chunks_exact_unsorted_selection():
    x = np.arange(17*3*7*8, dtype='f4').reshape(17,3,7,8)
    a = Array(x, (8,1,4,4))
    idx = (np.array([16,1,3]), np.array([2,0]), np.array([6,1]), np.array([7,2]))
    b = DecodedBudget()
    got = bounded_selection(a, idx, b)
    np.testing.assert_array_equal(got, x[np.ix_(*idx)])
    assert len(a.calls) == 16 and b.reads == 16
    assert b.used == 16*8*1*4*4*4


def test_budget_rejects_before_data_access():
    a = Array(np.ones((4,4),dtype='f4'), (4,4))
    with pytest.raises(RuntimeError,match='BEFORE'):
        bounded_selection(a, (np.arange(4),np.arange(4)), DecodedBudget(limit=63))
    assert not a.calls


@pytest.mark.parametrize('idx', [(np.array([]),np.arange(4)),(np.array([1,1]),np.arange(4)),(np.array([4]),np.arange(4)),(np.array([1.5]),np.arange(4))])
def test_invalid_selection(idx):
    with pytest.raises(ValueError):
        selection_plan(Array(np.ones((4,4)),(2,2)), idx)


def test_global_pressure_slabs_refused():
    a = SimpleNamespace(shape=(24,37,721,1440), chunks=(1,37,721,1440), shards=None, dtype=np.dtype('f4'))
    with pytest.raises(ValueError,match='8 MiB'):
        selection_plan(a, tuple(np.array([0]) for _ in range(4)))


def test_sharded_missing_deadline_guards():
    a = Array(np.ones((4,4),dtype='f4'),(2,2))
    a.shards = (4,4)
    with pytest.raises(ValueError,match='sharded'):
        selection_plan(a,(np.arange(4),np.arange(4)))
    a.shards = None
    a.x[0,0] = np.nan
    with pytest.raises(ValueError,match='missing'):
        bounded_selection(a,(np.arange(4),np.arange(4)),DecodedBudget())
    with pytest.raises(RuntimeError,match='deadline'):
        bounded_selection(a,(np.arange(4),np.arange(4)),DecodedBudget(started=0))


def root_fixture():
    import pandas as pd
    from data.preprocess.r7_preflight import canonical_unit
    times = pd.DatetimeIndex(np.concatenate([pd.date_range(f'{y}-01-01',periods=32,freq='6h').values for y in (2018,2019,2020)]))
    hours = np.asarray((times-pd.Timestamp('1940-01-01'))/pd.Timedelta(hours=1),dtype='i8')
    groups = {kind: {'latitude':Array(np.linspace(42,39.25,12),(12,)),
        'longitude':Array(np.linspace(114,116.75,12),(12,)),
        'valid_time':Array(hours.copy(),(96,),attrs={'units':'hours since 1940-01-01 00:00:00','calendar':'proleptic_gregorian'})} for kind in ('single','pressure')}
    groups['pressure']['pressure_level'] = Array(np.array([850.,500.]),(2,),attrs={'units':'hPa'})
    for variable, level, alias in FIELDS:
        group = groups['single' if level is None else 'pressure']
        if variable in group:
            continue
        dims = ('valid_time','latitude','longitude') if level is None else ('valid_time','pressure_level','latitude','longitude')
        shape = (96,12,12) if level is None else (96,2,12,12)
        group[variable] = Array(np.arange(np.prod(shape),dtype='f4').reshape(shape),
            (32,12,12) if level is None else (32,1,12,12),dims,{'units':canonical_unit(alias)})
    return {k+'/temporal':v for k,v in groups.items()}


def test_exact_pressure_time_and_payload_extraction():
    root = root_fixture()
    ds, report = extract_dataset(root)
    assert dict(ds.sizes) == {'time':96,'latitude':12,'longitude':12}
    assert len(ds.data_vars) == 11 and report['source_is_real_reanalysis']
    # This fixture is synthetic; it verifies mapping, not the declared remote source.
    np.testing.assert_array_equal(ds.t850.values,root['pressure/temporal']['t'].x[:,0])
    np.testing.assert_array_equal(ds.t500.values,root['pressure/temporal']['t'].x[:,1])
    assert report['decoded_charged_bytes'] < report['decoded_budget_bytes']
    assert report['network_body_bytes'] is None


@pytest.mark.parametrize('failure', ['pressure','units','coordinates'])
def test_bad_metadata_rejected_before_weather_reads(failure):
    root = root_fixture()
    if failure == 'pressure':
        root['pressure/temporal']['pressure_level'].x[:] = [850,850]
    elif failure == 'units':
        root['single/temporal']['t2m'].attrs['units'] = 'degC'
    else:
        root['pressure/temporal']['latitude'].x[0] = 41.99
    with pytest.raises(ValueError):
        extract_dataset(root)
    assert not root['single/temporal']['t2m'].calls
    assert not root['pressure/temporal']['t'].calls
