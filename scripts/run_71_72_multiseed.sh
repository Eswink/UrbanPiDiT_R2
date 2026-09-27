#!/usr/bin/env bash
# #71/#72 round-one bounded two-arm run: 2 seeds x 2 arms on the M2 segment.
#
# One seed per GPU (the two 3090s are not NVLink-coupled, so each process sees a
# full device). Each seed freezes its protocol.json before its first optimizer
# step; the run is validation-only and never opens the test manifest.
set -e
cd /data/esw/UrbanPiDiT_R2

MD=outputs/r7_m2_segment/store/manifests
OUT=outputs/r7_71_72_m1_rwa

run_seed () {
  .venv/bin/python scripts/study_r7_71_72_spacetime_rwa.py --mode seed \
    --seed "$1" --device-index "$2" --manifests "$MD" --out "$OUT" \
    > "logs/r7_71_72_seed$1.log" 2>&1
}

run_seed 41 0 &
P0=$!
sleep 20
run_seed 42 1 &
P1=$!
wait $P0 $P1
echo R7_71_72_SEEDS_DONE
