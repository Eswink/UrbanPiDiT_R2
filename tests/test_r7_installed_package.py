"""Verify actual wheel contents/imports with no checkout on the import path."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def test_built_wheel_imports_from_clean_directory(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'source'
    source.mkdir()
    for name in ['data','model','training']:
        shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('__pycache__','legacy*','raw','processed','interim','manifests'))
    modules=['train_r7_local','evaluate_r7_local','prepare_r7_local','calibrate_r7_local']
    for name in ['pyproject.toml']+[m+'.py' for m in modules]:
        shutil.copy2(root/name,source/name)
    wheels=tmp_path/'wheels'
    command=[sys.executable,'-m','pip','wheel',str(source),'--no-deps','--no-build-isolation','--no-index','-w',str(wheels)]
    built=subprocess.run(command,capture_output=True,text=True,timeout=90)
    assert built.returncode==0,built.stdout+built.stderr
    wheel=next(wheels.glob('*.whl'))
    with zipfile.ZipFile(wheel) as z:
        names=set(z.namelist())
        assert 'data/preprocess/r7_era5_zarr.py' in names
        assert 'data/preprocess/r7_preflight.py' in names
        assert 'data/download/arco_era5.py' in names
        assert 'training/r7_streaming.py' in names
        assert all(m+'.py' in names for m in modules)
        assert not any('/legacy' in n or '/raw/' in n or n.startswith('tests/') for n in names)
    target=tmp_path/'installed'
    installed=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(target),str(wheel)],
        capture_output=True,text=True,timeout=90)
    assert installed.returncode==0,installed.stdout+installed.stderr
    env=dict(os.environ,PYTHONPATH=str(target))
    clean=tmp_path/'clean'
    clean.mkdir()
    script='''
from pathlib import Path
import data.preprocess.r7_era5_zarr as source
import model.r7_halting
import training.r7_local_runner
import training.r7_evaluate
import train_r7_local, evaluate_r7_local, prepare_r7_local, calibrate_r7_local
import importlib.metadata as metadata
assert "installed" in str(Path(source.__file__).resolve())
entries=metadata.distribution("urbanpidit-r2-v6").entry_points
assert {e.name for e in entries}>={"urbanpidit-r7-train","urbanpidit-r7-evaluate","urbanpidit-r7-prepare","urbanpidit-r7-calibrate"}
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=clean,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    for module in modules:
        result=subprocess.run([sys.executable,'-m',module,'--help'],cwd=clean,env=env,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
