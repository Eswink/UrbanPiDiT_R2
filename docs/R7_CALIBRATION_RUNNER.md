# Local controller fitting and adaptive rollout (#31)

```bash
python calibrate_r7_local.py --checkpoint outputs/process/update_0000010.pt --train-manifest /path/manifests/train.jsonl --out outputs/controller --updates 10 --max-steps 4
python evaluate_r7_local.py --checkpoint outputs/process/update_0000010.pt --controller outputs/controller/controller.pt --manifest /path/manifests/test.jsonl --out outputs/adaptive --max-samples 32
python evaluate_r7_local.py --checkpoint outputs/process/update_0000010.pt --controller outputs/controller/controller.pt --manifest /path/manifests/test.jsonl --out outputs/forced_full --max-samples 32 --force-full-depth
```

These are explicit bounded local commands, not runs performed on real ERA5 here.
The parent model must be a matching R7 process checkpoint. Fitting refuses val/test
records and uses future weather only as training labels for the existing signed
gain objective. The entire forecaster is frozen/eval, and its tensor digest is
checked unchanged before publication. Only controller parameters are optimized.

Controller artifacts bind the exact parent checkpoint SHA256, training data/
normalization identity, model code, optimizer-update count and policy thresholds.
A different parent or code revision is rejected. This initial runner does not
resume controller optimizers or perform automatic threshold search. Thresholds
must be selected using validation only, then frozen before a test comparison.

Evaluation reports the same timestamp-exact held-out sample set, per-case MSE,
and **cumulative reasoning steps at each requested forecast horizon**. These
counts are not wall-clock speedup. Timings include IO and metric computation;
a separate warm-up/synchronized hardware experiment is needed for model latency.
The default 32-sample cap is an engineering check, not the paper test set.

The immediate next-step gain target is greedy and may miss benefits which emerge
only after two or more extra steps. Parent #7 must compare this with fixed-depth,
look-ahead/oracle and validation-calibrated controls on real data. Physical
process/forecast consistency is not inferred from input-time scalar proxy signs.
No calibration, SOTA or causal-explanation claim follows from the synthetic tests.
