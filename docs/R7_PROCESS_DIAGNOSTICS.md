# Input-time process diagnostic contract (#14, #34)

The eight anchored values are deterministic summaries of physical-unit input
weather fields, not human CoT annotations, causal truth or future process labels.
They are computed before channel normalization; their separate mean/std use
training years only. The dataset selects history_indices[-1], never target_index.

Standalone derivatives require finite strictly monotone 1-D geographic axes and
valid latitude/longitude ranges. Irregular monotone spacing remains supported
by finite differences; archive builders additionally require the regular 0.25
distribution grid. Wrapped/duplicate seams and near-pole divisions are rejected,
not silently repaired. Scalar regional summaries require a single finite 2-D
field; accidental leading dimensions cannot silently multiply an area mean.

| Proxy | Definition / unit |
|---|---|
| pressure-gradient strength | area mean magnitude of MSLP horizontal gradient, Pa/m |
| divergence / vorticity | spherical 850-hPa wind diagnostics, regional RMS, 1/s |
| temperature advection | area mean -u*dT/dx-v*dT/dy, K/s |
| moisture advection | area mean -u*dq/dx-v*dq/dy, (kg/kg)/s |
| moisture convergence | area mean -div(q*u,q*v), (kg/kg)/s |
| static stability proxy | theta500-theta850, K, not Brunt-Vaisala frequency |
| vertical shear proxy | magnitude of wind500-wind850, m/s, not dV/dz |

Their signs/values alone do not close atmospheric tendency budgets. Warm advection
can coexist with net cooling from other terms; input-time process summaries must
not be forced into simplistic future-temperature sign constraints. Forecast-process
physical consistency remains an explicitly separate research question.
