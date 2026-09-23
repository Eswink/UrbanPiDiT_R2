"""Explicit bounded real four-variable CPU pilot. No forecast-skill claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from data.download.wb2_surface_pilot import download_surface_pilot
from data.preprocess.r7_preflight import prepare_local
from training.r7_experiment import dataset_identity
from training.r7_local_runner import run_local_updates
from training.r7_evaluate import evaluate_local


def model_config(kind):
    base={
        "in_channels":4,
        "history_steps":2,
        "out_channels":4,
        "dim":16,
        "patch_size":2,
        "depth":1,
        "heads":4,
        "window_size":2,
    }
    if kind=="generic":
        base.update(latent_tokens=4,default_reasoning_steps=2)
    elif kind=="process":
        base.update(anchored_processes=4,free_processes=4,default_reasoning_steps=2)
    return base


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    source=out/"source"/"wb2_surface_pilot.nc"
    receipt=download_surface_pilot(source,out/"source"/"receipt.json")

    config={
        "channels":[
            {"variable":"2m_temperature","name":"t2m"},
            {"variable":"10m_u_component_of_wind","name":"u10"},
            {"variable":"10m_v_component_of_wind","name":"v10"},
            {"variable":"mean_sea_level_pressure","name":"mslp"},
        ],
        "split_years":{"train":[2018],"val":[2019],"test":[2020]},
        "history_steps":2,
        "history_interval_hours":6,
        "lead_time_hours":6,
        "sample_stride_hours":6,
        "time_chunk":3,
        "compute_process_targets":False,
    }
    preflight=prepare_local(
        source,config,write=True,
        store_path=out/"cache.zarr",
        manifest_dir=out/"manifests",
        max_raw_gib=.01,
    )
    identity,train_ds=dataset_identity(out/"manifests"/"train.jsonl")
    results={}
    for kind in ("native","generic","process"):
        checkpoint,training=run_local_updates(
            train_ds,kind=kind,model_config=model_config(kind),
            data_identity=identity,output_dir=out/f"training_{kind}",
            total_updates=2,batch_size=1,accumulation=1,steps=2,
            process_weight=0.0,device_name="cpu",
        )
        evaluation=evaluate_local(
            out/"manifests"/"test.jsonl",
            output_dir=out/f"evaluation_{kind}",
            checkpoint=checkpoint,
            lead_hours=(6,),max_samples=1,device_name="cpu",
        )
        results[kind]={
            "training_updates":training["updates_this_run"],
            "last_training_loss":training["losses"][-1]["loss"],
            "heldout_initializations":evaluation["n_evaluated"],
            "checkpoint":str(checkpoint.relative_to(out)),
        }

    final={
        "status":"real-public-multivariate-cpu-pilot-passed",
        "scientific_forecast_claim":False,
        "source_receipt":"source/receipt.json",
        "source_chunk_estimated_bytes":receipt["source_chunk_budget"]["estimated_uncompressed_bytes"],
        "local_shape":preflight["shape"],
        "windows":preflight["windows"],
        "models":results,
        "limitations":[
            "only four surface variables",
            "one transition per split year",
            "CPU integration smoke only; losses are not paper skill estimates",
            "process model uses process_weight=0 because pressure-level diagnostics are absent",
        ],
    }
    (out/"pilot_result.json").write_text(
        json.dumps(final,indent=2,ensure_ascii=False,allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(final))


if __name__=="__main__":
    main()
