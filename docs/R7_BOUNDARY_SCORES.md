# Regional boundary error stratification (#37)

```bash
python evaluate_r7_local.py --checkpoint outputs/process/update_0000010.pt --manifest data/manifests/r7/test.jsonl --out outputs/boundary --boundary-margins 2 4 --max-samples 32
```

Always use predeclared cell margins and the identical initialization set for all
models. The optional additional boundary_rmse.csv scores the SAME full-domain
forecasts: full, interior_m and the complementary edge_m band for every margin.
The ordinary rmse.csv/acc.csv and model predictions are unchanged. Margins are
in native grid cells, not constant physical kilometers. A margin must leave a
nonempty interior. Latitude weights, gridpoint counts and full-area fractions
are recorded; latitude/grid geometry cannot change between accumulated batches.

For each variable/horizon, full_MSE equals area_fraction_interior*interior_MSE
plus area_fraction_edge*edge_MSE (not an average of RMSEs). Every initialization
has equal weight and each region uses cosine-latitude area weighting. Frozen
training-only standard deviations restore physical units; no score is formed
across incompatible variable units. Missing/nonfinite cases are not dropped.

This is an error-stratification diagnostic, NOT a boundary-condition experiment.
It does not inject future reanalysis or numerical forecast boundary fields, nor
change the input halo. It cannot identify the cause of edge error or establish
that a different boundary treatment improves forecasting. Parent #8 still needs
real-data sensitivity controls. Do not publish only interior scores to hide edge
failures. No real multi-year or GPU run is launched by adding this option.

Tests cover analytic edge-only errors, exact MSE decomposition, latitude reversal,
uneven batches, units, immutable grids, nonfinite rejection and no-overwrite output.
Local evaluation/CLI tests verify the primary full-domain metrics and cases are
unchanged when boundary reporting is enabled.
