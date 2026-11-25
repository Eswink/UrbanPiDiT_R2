"""G1-B's frozen constants (decision 0119), kept apart from the code that reads them.

Split out of ``r7_s3_visit_gain_protocol.py`` when that module crossed the R-021 400-line soft
cap. Nothing here computes: these are the values a reading may not move, and separating them
from the helpers makes the frozen set easy to diff against a previous round.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORMAT = "r7-s3-visit-gain-v1"
STAGE = "S3-VISIT-GAIN"

VISIT_DAMPED, UNIFORM_DAMPED = "visit_damped", "uniform_damped"
ARMS = ((VISIT_DAMPED, VISIT_DAMPED), (UNIFORM_DAMPED, UNIFORM_DAMPED))
ARM_NAMES = tuple(name for name, _ in ARMS)
REFERENCE_ARM = "incumbent"

SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 48, 72)
STEP_HOURS = 6

#: The round's visit count. ``uniform_damped``'s constant is a function of it, so it is frozen
#: here as well as in the spec, and the two are asserted equal before training.
REASONING_STEPS = 4

#: 2,400 updates. This is the dose the same cell was last priced at and the rung the G0
#: headroom reading is about; it is not the 32,132-update p4000 rung, which is a different
#: question and a different cost.
UPDATES = 2_400
BATCH_SIZE = 2
LR = 2e-5
WARMUP = 10
WEIGHT_DECAY = 1e-4
CLIP = 1.0
PHYSICAL_WEIGHTS = (1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
CHECKPOINT_EVERY = 1_200

#: ``evaluate_local`` writes ``boundary_rmse.csv`` only when ``boundary_margins`` is given. On
#: this 65x65 instance ``(32,)`` would leave a single cell, so the margins are ``(1,)`` and the
#: region read is ``full``, which is the wide instance's ``interior_32`` (see the module docstring).
BOUNDARY_MARGINS = (1,)
REGION = "full"
REGION_NOTE = ("full(65x65) == interior_32(129x129): docs/R7_S3_WIDE_REGION_FULL.md records the "
               "wide instance's interior_32 as 4,225 points = 65x65 with t2m climatology RMSE "
               "matching the narrow v3-D2 instance to ~1e-10 per lead, so this region is that "
               "interior and not a substitute for it")

TRAIN_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/train.jsonl"
VAL_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/val.jsonl"
STORE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"

#: The registered in-protocol round whose ``incumbent`` arm is the parent this branch starts
#: from: three seeds at 32,132 updates, each with its own archived receipt and state digest.
PARENT_RUN = ROOT / "outputs/r7_s3_typed_evidence_in_protocol_20261120_attempt02"
PARENT_RECORD = "s3-typed-evidence-in-protocol"
PARENT_UPDATES = 32_132
PARENT_TEMPLATE = "seed{seed}/training/incumbent/update_{updates:07d}.pt"

#: The parent checkpoints' own SHA256, copied from each seed's archived
#: ``seed{N}_incumbent_receipt.json``. Pinning them here means a drifted parent is refused twice:
#: against the receipt and against this table.
PARENT_CHECKPOINT_SHA256 = {
    41: "8c154bd901f8d6a05f30476c2dc931ca253f99ff692d8ed9258f163e5e88e3e2",
    42: "9cb93456bc2ea244e69bdfe70b90219f5e61e853929886eed87360c514fd1831",
    43: "aacdeb59c9f101b11523f366ccc9fdcbfecfe36867bb9710136bd836ea482e19",
}

#: The archived model-code digest the parents were trained under (decision 0118's identity gate).
#: Adding ``visit_gain`` moves the live digest away from it, which is why a model-only migration
#: is required rather than a relaxed guard.
PARENT_MODEL_CODE_SHA = "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324"

#: The treated rate, now **measured** rather than extrapolated (decision 0120). The damped arm
#: ran at 1.075/1.098/1.110/1.109 s/update across the four quarters of a 400-update run on an
#: idle machine, and 1.0796/1.0921 with one worker per card, so the rate is flat from the start
#: and concurrency across cards costs nothing. 1.30 is 18 percent above every measurement: it
#: covers ``uniform_damped``, whose rate was not measured separately, and normal jitter.
#:
#: The attempt01 probe measured 2.48-2.65 s/update because a 6,040-test suite was running on the
#: same host; the trainer's DataLoader uses four workers, so the loader starved and the GPU idled.
#: That was a contaminated measurement, not a slow implementation -- see decision 0120.
TREATED_SECONDS_PER_UPDATE = 1.30
#: Five-lead scoring per arm, measured 2,987.30-3,176.62 s at 32,132 updates. This round's arms
#: are the same size and the same leads, so the same bound applies.
MEASURED_SECONDS_PER_EVALUATION = 3_200.0
ENVELOPE_MARGIN = 2.0

#: One seed runs two arms and five leads each: 2 x (2,400 x 1.30 + 3,200) = 12,640 s, plus the
#: 400-update probe pair. 25,280 s is exactly 2.0x that. The guard in :func:`envelope` recomputes
#: the requirement from the treated rate instead of comparing the envelope to the constant it was
#: derived from, which is why an earlier shortfall surfaced here rather than after a truncated seed.
PER_SEED_SECONDS = 25_280.0
PLANNED_SECONDS = 118_800.0
HARD_CAP_SECONDS = 216_000.0

ESTIMATED_PEAK_BYTES = 9_728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
DEVICE_MODE = "dual"
#: The round's declared primary card. ``protocol_fields`` takes every visible card with this one
#: first, and the parent scheduler rotates the frozen list across the three seeds.
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"

#: The exclusions, verbatim, so a later reader does not have to reconstruct them.
CLOSED_AXES = {
    "capacity_0048": ("dim and depth stay 192/4, the state_dict key set is identical and the "
                      "parameter count is bit-identical across A, B and C; this is a signal-"
                      "propagation contrast, not a capacity one"),
    "mixing_0077": "global_mixing_mode is untouched and the receptive field is unchanged",
    "reasoning_depth": ("record:s3-reasoning-depth stays unresolved and is not re-run; K remains 4 "
                        "and is not swept"),
    "supervision": "the deep-supervision organization is untouched; both supervision branches are closed",
    "tier3": "no loop-index embedding, no per-loop LoRA, no spectral norm, no zero-init head",
    "physics_loss": "no physical penalty term is added; C1/C2 stay closed",
}

ATTRIBUTION_BOUNDARY = (
    "A-B isolates whether the gain depends on the visit index, but it is not a clean visit-"
    "alignment contrast: at K=4 A's gains average 0.696114 while B multiplies by 0.632456, so B "
    "damps 9.14 percent more on average (1/sqrt is convex, so a constant at the mean index sits "
    "below the mean of the gains). A-B must be read as 'at B's damping level, is visit dependence "
    "better'. B-C has a zero parameter difference and is descriptive only; B is a treated arm, not "
    "the parent, so B-C is not a causal reading of damping either.")

EXITS = {
    "one": ("72h full(65x65) t2m: A-B is negative on all three seeds. Sign is per seed; a single "
            "flip makes this false and closes the branch"),
    "two": ("6h and 12h: the seed mean of A-B is <= 0 at each of the two leads. A single seed may "
            "flip; the mean is what is judged"),
    "three_a": ("off-path identity: a cell built with visit_gain=None reproduces the pre-switch "
                "expression bit for bit, installs the archived 155 tensors exactly, and its "
                "state_dict key set and every tensor equal the archived parent's"),
    "three_b": ("k=0 bridge: visit_damped at visit 0 equals visit_gain=None on the same input bit "
                "for bit, because g_0 = 1/sqrt(1+0) is exactly 1.0. Holds for A only; B's "
                "constant is 0.632456 at every visit including the first"),
    "four": ("every seed's parameter key set and parent state_digest are bit-identical: the "
             "migration receipt proves the p32,132 state went in unchanged"),
    "reading_rule": ("an exit with no measurement is null, never false and never true. Three seeds "
                     "are consistency evidence, not a significance test. r = 0, so nothing here is "
                     "a confirmatory claim about the 2023 test split"),
    "closure": ("any exit false closes this branch: no added seed, no sweep over gain families, no "
                "change to K, no post-hoc threshold edit"),
}

LIMITATIONS = [
    "screening at 2,400 updates on one 65x65 instance (2000-2021 train / 2022 val, 17 channels); "
    "the 129x129 wide instance is a different store with different parents and is not covered",
    "A-B is not an equal-mean-damping contrast: B's constant is 9.14 percent below A's mean gain at "
    "K=4, so a negative A-B cannot be attributed to visit alignment alone",
    "the damping constants come from a post-LN DeepNorm paper (arXiv:2607.13491); this trunk is "
    "pre-LN, so importing the form is an extrapolation and none of that paper's numbers is an "
    "expected value here",
    "C is the registered incumbent at 32,132 updates, not retrained, so B-C compares a 2,400-update "
    "run against a longer one and is descriptive only",
    "the region read is full(65x65), which equals the wide instance's interior_32; that equivalence "
    "rests on a ~1e-10 climatology match recorded in docs/R7_S3_WIDE_REGION_FULL.md, not on a "
    "bitwise identity",
    "the 2023 test split stays sealed; r = 0; test_read = false",
    "three seeds are consistency evidence, not a significance test; no interval or multiplicity "
    "correction is computed",
    "GPU runs are co-resident, so latency and memory observations include neighbour load; "
    "reproducibility is config-level with no bitwise GPU promise",
    "the exits are predeclared screening gates on the +6h-centred objective family, not a skill "
    "claim: a branch that stays open still needs its own full-dose confirmation round",
]


