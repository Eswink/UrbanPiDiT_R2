"""Read-only by default. Local NetCDF4/Zarr only; no download calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from data.preprocess.r7_preflight import prepare_local


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--config',required=True)
    ap.add_argument('--write',action='store_true')
    ap.add_argument('--store')
    ap.add_argument('--manifests')
    ap.add_argument('--max-raw-gib',type=float)
    ap.add_argument('--report')
    a=ap.parse_args()
    if a.report and Path(a.report).exists():
        raise FileExistsError(a.report)
    source=Path(a.source).resolve()
    for value in [a.store,a.manifests,a.report]:
        if value is not None:
            dest=Path(value).resolve()
            if dest==source or source in dest.parents:
                raise ValueError('outputs must not modify the input source tree')
    config=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    report=prepare_local(a.source,config,write=a.write,store_path=a.store,
        manifest_dir=a.manifests,max_raw_gib=a.max_raw_gib)
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if a.report:
        with Path(a.report).open('x',encoding='utf-8') as f:
            f.write(text)
    print(text)


if __name__=='__main__':
    main()
