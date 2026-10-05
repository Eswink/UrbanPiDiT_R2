"""#78 R-B: per-channel objective weights from the published change-scale sidecar.

Split out of ``r7_change_scale`` (R-021: that module is near the 400-line
target) so the derivation has its own small, testable home. It reads only the
validated sidecar metadata and never the store, the val split or the test
split.
"""
from __future__ import annotations

import numpy as np

from .r7_change_scale import RB_LOSS_RULE, RB_WEIGHT_RULE, validate_change_scale


def rb_loss_weights(meta):
    """Derive the frozen R-B channel weights from a validated sidecar.

    The issue's R-B formula is ``sum_c w_c * mean(((forecast - target) / d_c)^2)``
    in physical units. The training pipeline stores states divided by ``s_c``
    and the sidecar publishes ``runtime_ratio_c = d_c / s_c``, so the same
    objective in the stored space is ``mean_c w_c * ((err_norm_c) / ratio_c)^2``
    - i.e. per-channel weights ``w_c = (1 / ratio_c)^2``, here normalized to
    mean one so the objective's overall scale (and therefore the effective
    learning rate) matches the incumbent's equal-channel loss. Degenerate
    channels keep the incumbent weight: their runtime ratio is exactly 1.0, so
    ``(1/1)^2`` is the equal-channel unit weight. Fixed from train-only
    statistics; never tuned on val/test.
    """
    validate_change_scale(meta)
    ratio = np.asarray(meta['runtime_ratio'], dtype=np.float64)
    weights = 1.0 / np.square(ratio)
    weights = weights / float(weights.mean())
    return {'weights': weights.tolist(),
            'rule': RB_WEIGHT_RULE,
            'loss_rule': RB_LOSS_RULE,
            'degenerate_channels': [name for name, flag in
                                    zip(meta['channels'], meta['degenerate']) if flag],
            'source_change_scale_identity': meta['change_scale_identity']}
