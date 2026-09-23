import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml
from test_r7_storage_safety import fixture
from data.preprocess.r7_preflight import inspect_era5,prepare_local
from data.r7_zarr_dataset import ZarrAtmosWindowDataset


def config():
    return {'channels':[{'variable':'t','name':'t2m'}],
        'split_years':{'train':[2018],'val':[2019],'test':[2020]}}


@pytest.mark.parametrize('units',['','degC','C','hPa'])
def test_wrong_units_are_not_silently_converted(units):
    ds=fixture()
    ds['t'].attrs['units']=units
    with pytest.raises(ValueError,match='units'):
        inspect_era5(ds,config())


def test_pressure_unit_is_required():
    ds=fixture()
    ds['p']=ds['t'].expand_dims(level=[850])
    cfg=config()
    cfg['channels']=[{'variable':'p','name':'t850','level_hpa':850}]
    ds['level'].attrs['units']='Pa'
    with pytest.raises(ValueError,match='hPa'):
        inspect_era5(ds,cfg)
    ds['level'].attrs['units']='hPa'
    report,_=inspect_era5(ds,cfg)
    assert report['channels'][0]['canonical_units']=='K'


def test_local_netcdf_dry_run_write_and_cap(tmp_path):
    source=tmp_path/'source.nc'
    fixture().to_netcdf(source,engine='h5netcdf')
    before=source.read_bytes()
    report=prepare_local(source,config())
    assert report['mode']=='read-only-preflight'
    assert report['fingerprint']['scope']=='full-local-file'
    assert report['scientific_training_certified'] is False
    assert set(tmp_path.iterdir())=={source}
    with pytest.raises(ValueError,match='cap'):
        prepare_local(source,config(),write=True,store_path=tmp_path/'output',manifest_dir=tmp_path/'m',max_raw_gib=1e-12)
    assert not (tmp_path/'output').exists()
    result=prepare_local(source,config(),write=True,store_path=tmp_path/'output',manifest_dir=tmp_path/'m',max_raw_gib=.01)
    assert len(ZarrAtmosWindowDataset(result['manifests']['train']))>0
    assert source.read_bytes()==before
    assert (tmp_path/'m'/'source_preflight.json').is_file()


def test_cli_defaults_to_no_writes(tmp_path):
    source=tmp_path/'test.nc'
    fixture().to_netcdf(source,engine='h5netcdf')
    cfg=tmp_path/'data.yaml'
    cfg.write_text(yaml.safe_dump(config()))
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,'prepare_r7_local.py','--source',str(source),'--config',str(cfg)]
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['mode']=='read-only-preflight'
    assert set(tmp_path.iterdir())=={source,cfg}


def test_remote_sources_never_opened():
    with pytest.raises(ValueError,match='local'):
        prepare_local('gs://example/not-authorized.zarr',config())
