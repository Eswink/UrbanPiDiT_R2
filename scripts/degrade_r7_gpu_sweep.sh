#!/usr/bin/env bash
# Ordered one-factor-at-a-time degradation from the OOM configuration (D5).
# Each run changes exactly one factor relative to the failing baseline:
#   K=8, batch=48, grid 128x128, dim=1024, depth=12, checkpointing off.
set -u
cd /data/esw/UrbanPiDiT_R2
BASE="--device cuda --dim 1024 --depth 12 --channels 11 --latent 32 --anchored 8 --free 8 --warmup 0 --measure 1 --bf16"

run() {
  tag="$1"; shift
  if [ -d "outputs/gpu_degrade/$tag" ]; then
    echo "skip $tag (exists)"
    return
  fi
  timeout 420 .venv/bin/python scripts/bench_r7_gpu_memory.py \
    --out "outputs/gpu_degrade/$tag" $BASE "$@" > "logs/degrade_$tag.log" 2>&1
  echo "done $tag rc=$?"
}

run d1_reduce_grid  --steps 8 --batch 48 --grid 64 64
run d2_reduce_batch --steps 8 --batch 24 --grid 128 128
run d3_reduce_k     --steps 4 --batch 48 --grid 128 128
run d4_reduce_scale --steps 8 --batch 48 --grid 128 128 --dim 512 --depth 8
echo "degradation series complete"
