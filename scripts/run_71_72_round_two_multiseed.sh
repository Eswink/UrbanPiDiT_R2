#!/usr/bin/env bash
# #71/#72 round-two bounded four-arm run: 3 seeds x 4 arms on the M2 segment.
#
# One seed per process, run one after another on one device: the workload is small
# (260 MiB peak, single-digit GPU utilisation in round one), so sharing a device buys
# nothing and the wall clock stays predictable. DEV selects the device.
#
# Run history (recorded because it is part of the evidence): the first launch at
# 2026-09-28T12:15 used devices 0 and 1 concurrently for seeds 41/42 and was killed by
# a session restore at ~12:17 with two runs incomplete; its partial output directory
# was deleted and no number from it is reported. This version was launched detached
# (setsid nohup) on device 1, because an unrelated job had taken device 0.
set -e
cd /data/esw/UrbanPiDiT_R2

MD=outputs/r7_m2_segment/store/manifests
OUT=outputs/r7_71_72_round_two
DEV=${DEV:-1}

run_seed () {
  .venv/bin/python scripts/study_r7_71_72_round_two.py --mode seed \
    --seed "$1" --device-index "$DEV" --manifests "$MD" --out "$OUT" \
    > "logs/r7_71_72_round_two_seed$1.log" 2>&1
}

run_seed 41
run_seed 42
run_seed 43
echo R7_71_72_ROUND_TWO_SEEDS_DONE
