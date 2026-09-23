# Continuous-data offline CPU control (#56)

The #55 source is now a reusable19,052,672-byte local NetCDF, with immutable
SHA256 `0609fa38c1d88b82b985a15f93dd502c7eb7031f5d2b7452bbe971bf936dd9c1`.
Its own verifier retains the January/four-season pins/caps unchanged and checks
all11field hashes, exact3000time points, physicalunits, levels and source identity.
It cannot authenticate arbitrary self-written receipts.

Prepare a NEW local cache without Icechunk or another source-cloud download:

```bash
python -m data.download.continuous_pilot_replay --source <source.nc> --receipt <receipt.json> --out outputs/new_cache
```

The cache has998training windows and its own honest data identity. Never force
an old seasonal checkpoint to match this changed normalization/data.

The fixed control trains generic/process/process-no-aux from seeds41/42/43 for
800updates each, batch2,lr2e-4,K3. It tests24fixed balanced VALIDATION cases at
6/12/24/72h with bothK3 andsame-checkpointK1; persistence uses identical cases.
There is no test evaluation, controller fit, threshold selection, model change
or extension-until-success. Allvariables andseeds, including degradations, stay.

```python
from training.r7_continuous_control import run_continuous_control
run_continuous_control("source.nc", "receipt.json", "outputs/new_control")
```

The GitHub workflow usesCPU2threads,20minhardlimit anddenies outbound sockets
after fetching the existing22MBartifact. It archives source,checkpoints,code,
environment andresults; dependency/setup time is not model latency.
Equal optimizer updates are not equal FLOPs;800updates are not proof of convergence.
The24cases were previously inspected, so they remain exploratory validation.
