#!/usr/bin/env bash
# Measure each sweep cell in a FRESH process so `max_memory_reserved` is not a
# carried-over allocator high-water mark from earlier cells. The in-process
# sweep is still useful for `allocated`, but `reserved` must be isolated.
set -u
cd /data/esw/UrbanPiDiT_R2
OUT="${1:-outputs/gpu_sweep_isolated}"
DIM="${2:-384}"
DEPTH="${3:-8}"
CELLS="${OUT}/cells"
mkdir -p "$CELLS"

for KIND in generic process; do
  for K in 1 2 4 8; do
    for MODE in full_bptt streamed_truncated; do
      for CK in 0 1; do
        tag="${KIND}_K${K}_${MODE}_ck${CK}"
        if [ -d "${CELLS}/${tag}" ]; then
          echo "skip $tag (exists)"
          continue
        fi
        timeout 420 .venv/bin/python scripts/bench_r7_gpu_memory.py \
          --out "${CELLS}/${tag}" --device cuda --steps "$K" \
          --kinds "$KIND" --modes "$MODE" --ckpt "$CK" \
          --batch 4 --dim "$DIM" --depth "$DEPTH" --channels 11 --grid 12 12 \
          --latent 16 --anchored 4 --free 4 --warmup 1 --measure 2 --bf16 \
          > "${OUT}/${tag}.log" 2>&1
        echo "done $tag rc=$?"
      done
    done
  done
done
echo "isolated sweep complete: ${CELLS}"
