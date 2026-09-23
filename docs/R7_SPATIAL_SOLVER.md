# Opt-in spatial feedback to the shared forecast solver (#57)

The recorded #53 correction diagnosis found nearly parallel later updates on
several variables. #54 showed that simply increasing optimization did not fix
all depth-induced validation degradation. Neither observation proves a cause.

In the old equation, the draft reached the reasoner but the correction decoder
only received the fixed spatial context plus a pooled latent summary. The new
**opt-in** `spatial_solver_feedback=True` adds existing draft tokens aligned
patch-by-patch, without creating any new parameters or quadratic attention:

    P_next = Reason(P, concat(C, E(Y)))
    delta = Decode(C + Project(mean(P_next)) + E(Y))
    Y_next = Y + delta

Default False retains the original decoder equation bit-for-bit. The same
mechanism is available to generic recursion for a fair control. Disabling
`use_forecast_feedback` on the process model also disables direct draft feedback.
Fixed forward, streamed backward, adaptive inference and controller calibration
use one shared solver-conditioning function and are equivalence-tested.

All old checkpoint identities remain immutable. Model-source hashes legitimately
change; evaluate old artifacts using their archived `code.zip`/commit rather
than bypassing model-code checks. New baseline and candidate are trained from
scratch in the SAME new revision, with exact matched within-kind parameter counts.

The preregistered CPU experiment uses the pinned continuous250day source,
998train windows, generic/process x off/on x seeds41/42/43,400updates,batch2,K3,
unchanged objective/learning rate. It evaluates same-checkpointK1/K3 on24balanced
VALIDATION cases at6/12/24/72h; no controller fitting, test or threshold tuning.
The20minute CPU2thread workflow forbids outbound networking after artifact
retrieval, preserves all results/code/checkpoints, and never extends until a win.
An input route is not proof of reasoning or weather skill; empirical negatives
must remain visible. Do not compare these400-update scores as a fair contest
against earlier800-update experiments.
