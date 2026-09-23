# Delayed-gain oracle diagnostic (#39)

```bash
python diagnose_r7_gain.py --checkpoint outputs/process/update_0000010.pt --manifest data/manifests/r7/val.jsonl --out outputs/oracle.json --max-steps 4 --max-samples 32 --step-cost 0.001
```

This command is RETROSPECTIVE validation analysis, not an inference policy. Both
oracles access target-based errors after forecasts. No oracle action is fed back
into deployed routing, no model is retrained, and no checkpoint is modified.
Do not place these numbers in a main performance table as a deployable method.

The fixed process checkpoint produces E1..EK for one +6h transition using the
same normalized, equal-channel latitude-weighted MSE as the gain controller.
Errors are streamed with no autograd and only compact [cases,K] error arrays are
retained. The model input whitelist still excludes targets/diagnostic labels.

The myopic oracle continues only while the next measured error decrease exceeds
step_cost. The retrospective optimum minimizes E_k + step_cost*k over the whole
computed trajectory, choosing the earliest exact tie. Objective regret and cases
where the myopic oracle stops before a beneficial later depth are reported.
For E=[1,1.1,0.4] and cost=0.01, myopic stops at 1 while retrospective chooses 3.
This analytic test is not evidence that trained weather forecasts behave so.

Costs have normalized-MSE-per-reasoning-step units; they are not GPU seconds.
The optimum is conditional on this checkpoint, trajectory and declared cost.
It is not a free-running 72h rollout comparison: changing depth at an earlier
forecast transition can change all later states. Use the ordinary frozen-policy
rollout evaluation for held-out forecasts and the isolated profiler for timing.

Only validation manifests are accepted; output is exclusive, with checkpoint/
training/manifest identity and source declaration. The small default cap is an
engineering budget, not statistical acceptance. Parent #7 needs real validation
results to decide whether a look-ahead training redesign is justified; this
addition does not change the frozen primary algorithm based on speculation.
