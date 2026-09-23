# Parameter-comparable compact controls (#51)

This diagnostic uses the already verified four-season #49 source. No new source
archive extraction, GPU or paid service is needed. Widths were fixed by parameter
count BEFORE viewing baseline errors: U-Net80210, ConvLSTM81965, AFNO-inspired78803,
nonrecursive window84307, generic83222, process83319. All lie within10%of83319.
These compact implementations are architecture-inspired controls, not pretrained
FourCastNet/SOTA reproductions. Parameter counts are not FLOPs or wall-clock cost.

One seed42,200updates,batch2,LR2e-4, same120training windows. Evaluation uses
24fixed season-balanced cases in each validation/test split, leads6/12/24/72h.
Final200 is predeclared, not selected with test metrics. Native models optimize
final forecast loss; recursive models use deep supervision and process optionally
uses auxiliary weight0.1. Same sample/update budget is not a claim that recipes
are equally optimized for each architecture. Every variable/horizon is retained.

Each trained checkpoint is profiled in a NEW CPU subprocess on the same resident
batch1, FP32,two threads,5excluded warmups and20actual timing repetitions. Reads,
transfers and metrics are outside timed forward regions. Input hashes/platform,
actual medians/parameter counts and complete paired case sets are checked before
a combined table is written. The child profiler also prohibits source networking.
CPU results contain null CUDA memory fields. A single resident input with twenty
repetitions is a diagnostic, not a production throughput or scaling benchmark.

```bash
python scripts/study_r7_baselines_cpu.py --source <era5_four_season.nc> --receipt <receipt.json> --out outputs/new_baselines
```

Output must be new. The CPU workflow first retrieves the existing artifact,
then prohibits outbound sockets during training/evaluation/profiling. Runtime
is capped at20minutes. Code, original source, receipts, checkpoints, all metrics
and individual timing samples are retained, including failures and poor results.
This is supplementary one-seed evidence, not general superiority or significance.
