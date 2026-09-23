"""One explicit tiny real-source attempt; finite CPU steps, no scientific claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.download.arco_tiny_bounded import download_fixed_t2m
from data.preprocess.r7_preflight import prepare_local
from training.r7_experiment import dataset_identity
from training.r7_local_runner import run_local_updates
from training.r7_evaluate import evaluate_local


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    config={'channels':[{'variable':'2m_temperature','name':'t2m'}],
        'split_years':{'train':[2018],'val':[2019],'test':[2020]}}
    try:
        source=download_fixed_t2m(out/'source')
        preflight=prepare_local(source,config,write=True,store_path=out/'cache.zarr',
            manifest_dir=out/'manifests',max_raw_gib=.01)
        identity,ds=dataset_identity(out/'manifests'/'train.jsonl')
        checkpoint,training=run_local_updates(ds,kind='native',
            model_config={'in_channels':1,'dim':16,'depth':1,'heads':4,'window_size':2},
            data_identity=identity,output_dir=out/'training',total_updates=2,process_weight=0.,device_name='cpu')
        evaluation=evaluate_local(out/'manifests'/'test.jsonl',output_dir=out/'evaluation',
            checkpoint=checkpoint,lead_hours=(6,),max_samples=1,device_name='cpu')
        result={'status':'real-source-engineering-smoke-passed','scientific_forecast_claim':False,
            'source_receipts':'source/receipts.json','source_shape':preflight['shape'],
            'training_updates':training['updates_this_run'],'heldout_initializations':evaluation['n_evaluated'],
            'note':'single-variable nine-frame proof only; no multi-year forecast skill or GPU claim'}
    except Exception as exc:
        result={'status':'failed-no-fallback','error':f'{type(exc).__name__}: {exc}','scientific_forecast_claim':False}
        (out/'smoke_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise
    (out/'smoke_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
