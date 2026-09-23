# R7 local data publication and migration

Issue #24 introduces version-1 completion checks. Input files are never modified.
Both NPZ and Zarr builders refuse existing destination/manifest directories,
even if empty. Output paths must be distinct and non-nested. Cooperative local
reservation files prevent two builders using the same destination; this is not
a distributed locking/transaction protocol.

On failure, partial output remains available for inspection. It is not marked
complete and cannot be opened by the production R7 loaders. Choose a **new**
destination to retry. No directory is silently removed to make a retry succeed.
The only automatic removals are reservation files owned by that invocation.

On success, the manifest directory contains BUILD_COMPLETE.json. Zarr also has
schema_version=1, time_unit=ns and build_complete=true. Readers require both
publication and schema invariants. Older unversioned R7 caches must be rebuilt
in new destinations from their source, not relabeled by adding a marker. V6
loaders and the synthetic model fixtures remain separate.

Timestamps are explicitly converted to nanoseconds before storing time_ns;
integer timestamp resolution is not assumed. Required frames must agree with
manifest timestamps/lead/split. No nearest-time substitution or teacher forcing.

Year partitions must satisfy max(train)<min(val) and max(val)<min(test).
Zarr statistics use float64 centered mergeable population moments over training
frames only. Validation/test frames never update atmospheric or process
normalization. Process labels come only from the last input-history frame.

Raw physical fields are stored without spatial interpolation. Units are copied
from source attributes; a missing unit is written as 'unknown', not guessed.
The source label is a caller declaration, **not proof of ERA5 provenance**.
Synthetic xarray tests validate software only. Real-data acceptance still needs
source checksums/access records, variable/unit audit and a real trained run.

References for API semantics: pandas DatetimeIndex.as_unit and Zarr open_group
(mode w-). The implementation uses the repository's optional data dependencies;
no remote data download is performed by these builders.
