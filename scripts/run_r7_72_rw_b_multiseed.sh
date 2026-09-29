#!/usr/bin/env bash
# #72 M2-B bounded four-arm RW-B round: 2 seeds x 4 arms on the M2 segment.
#
# One seed per process. The two seeds go to the two local 3090s in parallel (the
# plan's GPU0/GPU1 policy): the workload is small (single-digit GPU utilisation
# and a few hundred MiB in the earlier rounds), so the seeds do not contend and
# the wall clock halves. The total GPU-hours reported by the round are the sum of
# the per-run elapsed seconds, which is the same number either way.
#
# An incomplete seed is never reported as a number: the finalize step fails if a
# declared seed has no result. The launch history of this round is recorded in
# docs/R7_72_RW_B_PILOT.md; nothing here is a scientific claim.
set -e
cd /data/esw/UrbanPiDiT_R2

MD=outputs/r7_m2_segment/store/manifests
OUT=${OUT:-outputs/r7_72_rw_b_pilot}

run_seed () {
  .venv/bin/python scripts/study_r7_72_rw_b.py --mode seed \
    --seed "$1" --device-index "$2" --manifests "$MD" --out "$OUT" \
    > "logs/r7_72_rw_b_seed$1.log" 2>&1
}

run_seed 41 0 &
SEED41=$!
run_seed 42 1 &
SEED42=$!
wait "$SEED41" "$SEED42"

.venv/bin/python scripts/study_r7_72_rw_b.py --mode finalize \
  --manifests "$MD" --out "$OUT" > logs/r7_72_rw_b_finalize.log 2>&1
echo R7_72_RW_B_DONE
