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
