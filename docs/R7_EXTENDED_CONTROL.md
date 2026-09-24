# Extended optimization control (#54)

The initial seasonal CPU experiment did not establish benefits from anchored
supervision or deeper reasoning. This fixed control tests whether that observation
persists after additional optimization; it changes NO model or training objective.

Predeclared: original #50 generic/process/process-no-aux, seeds41/42/43; restore
exact cache/checkpoint identities, resume original optimizer/RNG from update200
to800,600 additional updates each. Preserve batch2,lr2e-4,K3,auxiliary weights,
sample order and code hashes. No earlier checkpoint is overwritten. All nine
parent contracts are validated before training and recorded in protocol.json.

Evaluate24 balanced VALIDATION cases at6/12/24/72h for original200 K3,final800 K3
and same-final-checkpoint K1. No test evaluation, controller fitting, policy
retuning or extension-until-success. All variables and seeds are retained.
Training/validation plots remain exploratory, not a pristine final benchmark.

Run after obtaining the #50 artifact:

```python
from training.r7_extended_control import run_extended_control
run_extended_control('input_study', 'outputs/new_extended_control')
```

The workflow uses CPU2threads and a20minute hard limit. It forbids outbound
sockets during the experiment, archives checkpoints/code/environment/results,
and preserves per-variable seed mean/SD without mixing physical units. Equal
updates are not equal FLOPs; neither800updates nor improvement proves convergence.

## Model-source digest requirement

This workflow loads checkpoints produced by an earlier commit, so it must run
under the model implementation those checkpoints recorded. `load_checkpoint`
(`training/r7_experiment.py`) compares the checkpoint's `model_code_sha256`
against the digest of every non-legacy `.py` under `model/` and fails closed on
mismatch. Reformatting any `model/` file changes that digest.

Current pinned digest after the Q-001/long-line clean-up: `d9fb07f2d38d41d681b45c0b4d539edb8f0f9620f44ba0c426c40b43d672f665`
(previously `20196c64...`). To replay an older artifact, run it with the
`code.zip` archived beside that artifact — never by bypassing the identity check.
See `docs/rules/CHANGELOG.md` (2026-09-24, second pass) and
`docs/rules/OPEN_QUESTIONS.md` Q-009.
