"""Stage B point 3: 4 epoch = 32,132 updates at batch 4 -- the curve's third ordinate.

One arm, one seed 41, batch fixed at 4, from the pinned migrated parent. The only difference
from point 1 (0.25 epoch) is the update count: this is a calibration axis, not a factor
comparison, so nothing else may move. The point is its own independent attempt -- there is no
checkpoint resume from point 1 -- and the saturation judgement belongs to the completed
four-point ladder, not to this point.

This file is a thin named binding because decision 0067 requires the execution inventory to
start from a named entry file; the machinery is in ``r7_s3_curve_point_large_driver``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_curve_point_large_driver as curve
from scripts import r7_s3_curve_point_large_protocol as proto

POINT = "p4000"
UPDATES = proto.updates_for(POINT)
SEEDS = proto.SEEDS
_TUNED = proto.budgets(POINT)
PER_SEED_SECONDS = _TUNED["per_seed_seconds"]
PLANNED_SECONDS = _TUNED["planned_seconds"]
HARD_CAP_SECONDS = _TUNED["hard_cap_seconds"]


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(POINT, output, started=started, deadline=deadline,
                        code_archive=code_archive, code_commit=code_commit,
                        code_sha256=code_sha256)


def main(argv=None):
    curve.main(sys.argv[1:] if argv is None else argv, POINT)


if __name__ == "__main__":
    sys.exit(main())
