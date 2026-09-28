#!/usr/bin/env bash
# #71 round-three bounded four-arm run: 3 seeds x 4 arms (A/B/E/P) on the M2 segment.
#
# One seed per process, one after another on a single device: the workload is small
# (~260 MiB peak, single-digit GPU utilisation in rounds one and two), so sharing a
# device buys nothing and the wall clock stays predictable. DEV selects the device.
# The actual launch history of this round is recorded in
# docs/R7_71_72_ROUND_THREE.md; an incomplete seed is never reported as a number.
set -e
cd /data/esw/UrbanPiDiT_R2

MD=outputs/r7_m2_segment/store/manifests
OUT=outputs/r7_71_72_round_three
DEV=${DEV:-1}

run_seed () {
  .venv/bin/python scripts/study_r7_71_72_round_three.py --mode seed \
    --seed "$1" --device-index "$DEV" --manifests "$MD" --out "$OUT" \
    > "logs/r7_71_72_round_three_seed$1.log" 2>&1
}

run_seed 41
run_seed 42
run_seed 43
.venv/bin/python scripts/study_r7_71_72_round_three.py --mode finalize \
  --manifests "$MD" --out "$OUT" > logs/r7_71_72_round_three_finalize.log 2>&1
echo R7_71_72_ROUND_THREE_DONE
