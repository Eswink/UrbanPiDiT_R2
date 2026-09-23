import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from data.preprocess.contracts import RunningMoments, chronological_splits, utc_time_index
from data.preprocess.r7_era5 import ERA5ChannelSpec, build_r7_era5_npz_from_dataset, stack_era5_channels
from data.preprocess.r7_era5_zarr import build_r7_era5_zarr_from_dataset
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from data.r7_dataset import ManifestAtmosNPZDataset
from data.r7_store import validate_store


def fixture(unit='ns', future_offset=100):
    times=np.concatenate([pd.date_range(f'{y}-01-01',periods=8,freq='6h').to_numpy(dtype=f'datetime64[{unit}]') for y in [2018,2019,2020]])
    values=np.arange(24*3*4,dtype=np.float32).reshape(24,3,4)/10+280
    values[8:]+=future_offset
    ds=xr.Dataset({'t':(('time','latitude','longitude'),values)},coords={'time':times,'latitude':[40,39.75,39.5],'longitude':[115,115.25,115.5,115.75]})
    ds['t'].attrs['units']='K'
    return ds


def build(path,kind='zarr',ds=None,**kwargs):
    options=dict(ds=fixture() if ds is None else ds,manifest_dir=path/'manifest',
        specs=(ERA5ChannelSpec('t',name='t2m'),),split_years={'train':[2018],'val':[2019],'test':[2020]})
    options.update(kwargs)
    if kind=='npz':
        return build_r7_era5_npz_from_dataset(out_dir=path/'output',**options)
    return build_r7_era5_zarr_from_dataset(store_path=path/'output',time_chunk=3,**options)


@pytest.mark.parametrize('kind',['npz','zarr'])
def test_existing_outputs_are_never_overwritten(tmp_path,kind):
    build(tmp_path,kind)
    sentinel=tmp_path/'output'/'USER_SENTINEL'
    sentinel.write_bytes(b'user-owned')
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        build(tmp_path,kind,ds=fixture(future_offset=9999))
    after={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after


def test_zarr_failure_is_not_a_complete_dataset(tmp_path,monkeypatch):
    import data.preprocess.r7_era5_zarr as module
    original=module.stack_era5_channels
    calls=0
    def fail_later(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            raise RuntimeError('injected after partial write')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'stack_era5_channels',fail_later)
    with pytest.raises(RuntimeError,match='injected'):
        build(tmp_path)
    root=zarr.open_group(str(tmp_path/'output'),mode='r')
    assert root.attrs['build_complete'] is False
    assert not (tmp_path/'manifest'/'BUILD_COMPLETE.json').exists()
    with pytest.raises(ValueError,match='incomplete'):
        validate_store(root)
    assert not list(tmp_path.glob('*.r7-build.lock'))


def test_us_ns_time_and_train_only_moments(tmp_path):
    build(tmp_path/'a',ds=fixture('ns'))
    build(tmp_path/'b',ds=fixture('us',future_offset=99999))
    a=zarr.open_group(str(tmp_path/'a'/'output'),mode='r')
    b=zarr.open_group(str(tmp_path/'b'/'output'),mode='r')
    np.testing.assert_array_equal(a['time_ns'][:],b['time_ns'][:])
    assert a.attrs['time_unit']=='ns'
    np.testing.assert_array_equal(a['time_ns'][:],utc_time_index(fixture().time.values).asi8)
    for key in ['normalization_mean','normalization_std']:
        np.testing.assert_array_equal(a[key][:],b[key][:])
    raw=fixture()['t'].values[:8].astype(np.float64)
    np.testing.assert_allclose(a['normalization_std'][:],[raw.std()],rtol=1e-6)
    sample=ZarrAtmosWindowDataset(tmp_path/'a'/'manifest'/'train.jsonl')[0]
    assert tuple(sample['coarse_history'].shape)==(2,1,3,4)


@pytest.mark.parametrize('change',['complete','unit','std','record'])
def test_reader_rejects_tampering(tmp_path,change):
    paths=build(tmp_path)
    root=zarr.open_group(str(tmp_path/'output'),mode='r+')
    if change=='complete':
        root.attrs['build_complete']=False
    elif change=='unit':
        root.attrs['time_unit']='us'
    elif change=='std':
        root['normalization_std'][:]=0
    else:
        records=[json.loads(s) for s in paths['train'].read_text().splitlines()]
        records[0]['target_index']=0
        paths['train'].write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(ValueError):
        ZarrAtmosWindowDataset(paths['train'])[0]


def test_npz_reader_requires_completion_marker(tmp_path):
    path=tmp_path/'train.jsonl'
    path.write_text('{}\n')
    with pytest.raises(ValueError,match='incomplete'):
        ManifestAtmosNPZDataset(path)


@pytest.mark.parametrize('splits',[
    {'train':[2020],'val':[2019],'test':[2021]},
    {'train':[2018],'val':[2018],'test':[2020]},
    {'train':[2018.0],'val':[2019],'test':[2020]},
    {'train':[True],'val':[2019],'test':[2020]},
])
def test_splits_are_strictly_chronological(splits):
    with pytest.raises(ValueError):
        chronological_splits(splits)


def test_duplicate_channels_are_rejected():
    with pytest.raises(ValueError,match='unique'):
        stack_era5_channels(fixture(),(ERA5ChannelSpec('t'),ERA5ChannelSpec('t')))


def test_centered_moments_do_not_cancel_large_offset():
    x=1e12+np.arange(128,dtype=np.float64).reshape(64,2)*.25
    stats=RunningMoments(2)
    for chunk in np.array_split(x,8):
        stats.update(chunk)
    _,std=stats.finish()
    np.testing.assert_allclose(std,x.std(0),rtol=1e-6)
    assert stats.count==64
