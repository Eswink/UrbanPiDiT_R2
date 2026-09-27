#!/usr/bin/env bash
# M2 bucket-expansion training run: 5 arms x 3 seeds on the 60-day segment.
#
# Seeds run one per GPU where possible (the two 3090s are not NVLink-coupled, so
# two concurrent processes each see a full GPU). The protocol is frozen inside
# the first step of each process, before any optimizer step, and each seed's
# result carries the digest.
set -e
cd /data/esw/UrbanPiDiT_R2

MD=outputs/r7_m2_segment/store/manifests
OUT=outputs/r7_m2_multiseed
NOTE="$(cat /tmp/m2_note.txt)"
LIMITS="$(cat /tmp/m2_limits.json)"

run_seed () {
  .venv/bin/python scripts/study_r7_b2_multiseed.py --mode seed --phase confirm \
    --seed "$1" --device-index "$2" --manifests "$MD" --out "$OUT" \
    --segment-note "$NOTE" --limitations "$LIMITS" > "logs/m2_seed$1.log" 2>&1
}

run_seed 41 0 &
P0=$!
sleep 20
run_seed 42 1 &
P1=$!
wait $P0 $P1
run_seed 43 0
echo M2_SEEDS_DONE
