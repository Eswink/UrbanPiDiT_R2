"""A separate immutable pin for the actual continuous 250-day ERA5 artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .pressure_pilot_replay import _verify_pilot_profile, _copy_verified_pair
from .earthmover_pilot import FIELDS

PINNED_CONTINUOUS_SHA256 = '0609fa38c1d88b82b985a15f93dd502c7eb7031f5d2b7452bbe971bf936dd9c1'


def verify_continuous_pilot(source, receipt):
    report = _verify_pilot_profile(source, receipt, pin=PINNED_CONTINUOUS_SHA256,
        profile='continuous-250d', origin='GitHub Actions35893397412 artifact10765553501')
    if (report.get('same_time_chunks_as_four_season_verified') is not True or
        report.get('decoded_budget_bytes') != 192*2**20 or
        report.get('decoded_charged_bytes') != 180142968 or
        report.get('cropped_state_float32_bytes') != 19008000 or
        report.get('network_body_bytes') is not None):
        raise ValueError('continuous source budget declaration mismatch')
    from .seasonal_sampling import requested_times
    times = requested_times('continuous-250d')
    expected = {str(y):dict(first=times[times.year==y][0].isoformat(),
        last=times[times.year==y][-1].isoformat(),count=1000) for y in (2018,2019,2020)}
    if report.get('time_coverage_by_year') != expected:
        raise ValueError('continuous calendar coverage mismatch')
    from data.preprocess.r7_preflight import canonical_unit, SI_UNITS, _unit
    for row in report['variables']:
        if _unit(row['units']) not in SI_UNITS[canonical_unit(row['name'])]:
            raise ValueError('continuous physical unit mismatch')
    return report


def copy_verified_continuous_pilot(source, receipt, output_source, output_receipt):
    return _copy_verified_pair(source, receipt, output_source, output_receipt,
        verifier=verify_continuous_pilot, pin=PINNED_CONTINUOUS_SHA256)


def prepare_continuous_pilot(source, receipt, output_dir):
    """Verify real bytes before writing; assign a fresh, honest cache identity."""
    from data.preprocess.r7_preflight import prepare_local
    from .seasonal_sampling import write_balanced_manifest, requested_times
    verify_continuous_pilot(source,receipt)  # fail before creating output
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    copied,audit=copy_verified_continuous_pilot(source,receipt,out/'source'/'era5_continuous250d.nc',
                                               out/'source'/'receipt.json')
    config=dict(channels=[dict(variable=n,name=n) for _,_,n in FIELDS],
        split_years=dict(train=[2018],val=[2019],test=[2020]),history_steps=2,
        history_interval_hours=6,lead_time_hours=6,sample_stride_hours=6,
        time_chunk=8,compute_process_targets=True)
    report=prepare_local(copied,config,write=True,store_path=out/'cache.zarr',
                         manifest_dir=out/'manifests',max_raw_gib=.05)
    if report['windows'] != dict(train=998,val=998,test=998):
        raise ValueError('continuous replay window counts differ')
    validation=write_balanced_manifest(out/'manifests'/'val.jsonl',requested_times('continuous-250d'))
    return out/'manifests'/'train.jsonl',validation,audit,report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--receipt',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    train,val,audit,report=prepare_continuous_pilot(args.source,args.receipt,args.out)
    print(json.dumps(dict(train=str(train),validation=str(val),windows=report['windows'],
                         source_sha256=audit['source_netcdf_sha256'],source_cloud_requests=0)))


if __name__=='__main__':
    main()
