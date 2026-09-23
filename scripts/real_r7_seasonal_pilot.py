"""Small true four-season ERA5 acquisition and CPU integration, not a benchmark."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from data.download.earthmover_pilot import download_pilot, FIELDS
    from data.download.seasonal_sampling import requested_times, write_balanced_manifest
    from data.preprocess.r7_preflight import prepare_local
    from training.r7_experiment import dataset_identity
    from training.r7_local_runner import run_local_updates
    from training.r7_evaluate import evaluate_local
    from training.r7_cpu_study import _report_matrix, model_settings
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    protocol = dict(sampling='four-season',months=[1,4,7,9],years=[2018,2019,2020],
        time_count=384,channels=[n for _,_,n in FIELDS],seed=42,updates=20,steps=3,
        max_samples=24,selection='first six complete 72h trajectories per sampled month',
        lead_hours=[6,12,24,72],scientific_claim=False)
    with (out/'protocol.json').open('x') as f: json.dump(protocol,f,indent=2)
    source,receipt = download_pilot(out/'source'/'era5_four_season.nc',out/'source'/'receipt.json',sampling='four-season')
    config = {'channels':[{'variable':n,'name':n} for _,_,n in FIELDS],
        'split_years':{'train':[2018],'val':[2019],'test':[2020]},
        'history_steps':2,'history_interval_hours':6,'lead_time_hours':6,
        'sample_stride_hours':6,'time_chunk':8,'compute_process_targets':True}
    prepared = prepare_local(source,config,write=True,store_path=out/'cache.zarr',manifest_dir=out/'manifests',max_raw_gib=.01)
    if prepared['windows'] != dict(train=120,val=120,test=120):
        raise RuntimeError('unexpected seasonal window counts')
    identity,dataset = dataset_identity(out/'manifests'/'train.jsonl')
    labels = torch.stack([dataset[i]['process_targets'] for i in range(len(dataset))])
    if labels.shape != (120,8) or not torch.isfinite(labels).all():
        raise RuntimeError('seasonal process diagnostic supervision invalid')
    balanced = {s:write_balanced_manifest(out/'manifests'/f'{s}.jsonl',requested_times('four-season')) for s in ('val','test')}
    kind,model,pw = model_settings('process')
    checkpoint,training = run_local_updates(dataset,kind=kind,model_config=model,data_identity=identity,
        output_dir=out/'training',total_updates=20,batch_size=2,steps=3,seed=42,process_weight=pw,device_name='cpu')
    reports,season_rows = {},[]
    for name,cp,kw in [('persistence',None,{}),('process_K3',checkpoint,{}),('process_K1',checkpoint,{'reasoning_steps':1})]:
        report = evaluate_local(balanced['test'],output_dir=out/'evaluation'/name,checkpoint=cp,
            max_samples=24,lead_hours=(6,12,24,72),device_name='cpu',**kw)
        counts = Counter(pd.Timestamp(r['init_time']).month for r in report['initializations'])
        if dict(counts) != {1:6,4:6,7:6,9:6}:
            raise RuntimeError('evaluation is not season-balanced')
        reports[name] = report
        for month in (1,4,7,9):
            subset = dict(report,initializations=[r for r in report['initializations'] if pd.Timestamp(r['init_time']).month==month],n_evaluated=6)
            _,rmse = _report_matrix(subset)
            for i,lead in enumerate(report['lead_hours']):
                for j,(channel,unit) in enumerate(zip(report['channels'],report['units'])):
                    season_rows.append(dict(model=name,month=month,lead_hours=lead,variable=channel,unit=unit,rmse=float(rmse[i,j]),n_cases=6))
    result = dict(real_seasonal_cpu_integration_passed=True,scientific_claim=False,gpu_used=False,
        protocol=protocol,source=receipt,preparation=prepared,process_label_shape=list(labels.shape),
        training=training,reports=reports,season_rmse=season_rows,elapsed_seconds=time.monotonic()-started,
        limitations=['Eight-day excerpts per sampled month, not complete years or seasons',
            'Single small tile and 20 updates: integration only, no forecast-skill superiority',
            'Same decoded source chunk budget does not measure HTTP traffic or peak RAM'])
    with (out/'seasonal_result.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    import csv
    with (out/'season_rmse.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(season_rows[0]));writer.writeheader();writer.writerows(season_rows)
    print(json.dumps({'complete':True,'source_bytes':receipt['source_netcdf_bytes'],
        'decoded_charged_bytes':receipt['decoded_charged_bytes'],'windows':prepared['windows'],
        'evaluation_cases_per_month':dict(counts),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__ == '__main__': main()
