# Explicit cases and cross-profile comparison (#58)

The original seasonal and continuous studies each paired all models internally.
They did NOT select identical cases across profiles. With continuous history,
Apr/Jul/Sep1 at00UTC becomes available; the first-six-complete rule then removes
the corresponding second day of that month at12UTC. Their24-case aggregate tables therefore cannot
be treated as a paired training-data-quantity effect.

The original results/default selector are not altered. A new explicit timestamp
selector either selects every declared complete72h case or fails, inspecting no
target values or forecasts.

The retrospective comparator joins the original validation reports by timestamp,
checks units/lead/valid-time/model/seed/budget metadata, retains the intersection
and BOTH exclusion lists, and recomputes physical RMSE from stored per-case MSE.
For #54 vs #56,21cases intersect; three cases per side are excluded based only
on timestamp availability, never prediction quality. It retains both manifest
and training identities rather than pretending different normalization is equal.

No model/forecast/training/test/threshold is run or modified by this analysis.
Archived original model source files must match, and original resource/checkpoint
ownership is verified. Newly changed current model code is not used to load old
weights. Raw truth-array identity is not established by comparing error JSONs.

    from training.r7_profile_comparison import compare_original_studies
    compare_original_studies("unpacked54", "unpacked56", "fresh_comparison.json")

This is an exploratory common-case subset audit, not a pristine final test,
independent-event significance claim, or clean isolation of a single cause:
coverage, normalization and sample order all change under fixed800updates.
