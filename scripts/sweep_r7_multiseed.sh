#!/usr/bin/env bash
# Run the #20 multi-seed trick comparison, ONE CELL PER PROCESS.
#
# `max_memory_reserved` is the caching allocator's high-water mark and carries
# across cells inside one interpreter, so a single-process loop silently
# inflates later rows. R7_GPU_BRINGUP.md documents that artefact; this driver
# exists so the multi-seed table cannot inherit it.
#
# The plan step and the per-cell steps MUST receive the same measurement flags.
# If they diverge, protocol.json freezes a description of settings that were not
# the ones measured, which silently invalidates the frozen record.
#
# Usage: bash scripts/sweep_r7_multiseed.sh <out-dir>
set -u
cd /data/esw/UrbanPiDiT_R2
OUT="${1:?usage: sweep_r7_multiseed.sh <out-dir>}"
SCRIPT="scripts/compare_r7_gpu_multiseed.py"

# One shared flag set for the protocol and for every measured cell.
MEASURE_ARGS=(
  --data "${R7_MULTISEED_DATA:-real}"
  --device cuda
  --seeds ${R7_MULTISEED_SEEDS:-41 42 43}
  --steps ${R7_MULTISEED_STEPS:-1 4 8}
  --kinds ${R7_MULTISEED_KINDS:-generic process}
  --modes full_bptt streamed_truncated
  --ckpt 0 1
  --batch "${R7_MULTISEED_BATCH:-4}"
  --grid 12 12
  --channels 11
  --dim "${R7_MULTISEED_DIM:-384}"
  --depth "${R7_MULTISEED_DEPTH:-8}"
  --heads 4 --window 8 --latent 16 --anchored 4 --free 4
  --warmup "${R7_MULTISEED_WARMUP:-1}"
  --measure "${R7_MULTISEED_MEASURE:-2}"
)
if [ "${R7_MULTISEED_NO_BF16:-0}" != "1" ]; then
  MEASURE_ARGS+=(--bf16)
fi

if [ ! -f "${OUT}/cells.jsonl" ]; then
  .venv/bin/python "${SCRIPT}" --plan --out "${OUT}" "${MEASURE_ARGS[@]}" \
    || { echo "plan failed"; exit 1; }
fi

CELLS="${OUT}/cells"
mkdir -p "$CELLS"

# Skip the digest header; every other line is one cell.
tail -n +2 "${OUT}/cells.jsonl" | while IFS= read -r CELL; do
  [ -z "$CELL" ] && continue
  TAG=$(.venv/bin/python -c '
import json, sys
c = json.loads(sys.argv[1])
print("{}_K{}_{}_ck{}_s{}".format(c["kind"], c["k"], c["training_mode"],
      int(c["activation_checkpointing"]), c["seed"]))
' "$CELL")
  if [ -d "${CELLS}/${TAG}" ]; then
    echo "skip ${TAG} (exists)"
    continue
  fi
  timeout 420 .venv/bin/python "${SCRIPT}" \
    --cell "$CELL" --cell-out "${CELLS}/${TAG}" \
    "${MEASURE_ARGS[@]}" \
    > "${OUT}/${TAG}.log" 2>&1
  echo "done ${TAG} rc=$?"
done

.venv/bin/python "${SCRIPT}" --aggregate --out "${OUT}"
echo "multiseed sweep complete: ${OUT}"
