# R7 public-data source decision log

This document records bounded, evidence-driven source choices for the CPU-first
real-data phase. It does **not** certify a scientific benchmark or authorize a
large archive mirror.

## Verified WeatherBench2 / ARCO metadata

Public anonymous ERA5 sources:

- WeatherBench2 0.25°, 6-hour:
  `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr`
- ARCO ERA5 0.25°, hourly:
  `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`

Metadata-only probes show:

| source | surface chunk | pressure chunk | implication |
| --- | --- | --- | --- |
| WeatherBench2 | `[1,721,1440]` | `[1,13,721,1440]` | surface pilot is bounded; pressure tiny-ROI reads still touch all 13 levels/global grid |
| ARCO full_37 | `[1,721,1440]` | `[1,37,721,1440]` | pressure tiny-ROI reads are even more expensive |

Therefore R7 refuses to treat a visually small ROI as a small transfer when the
source chunk is a whole-global pressure slab. `data/download/chunk_budget.py`
implements the fail-closed preflight used before field reads.

## Verified real four-variable surface pilot

Issue #43 / workflow run 35871357066 used the public WeatherBench2 source and
selected exactly:

- 2m temperature
- 10m u wind
- 10m v wind
- mean sea-level pressure
- 9 timestamps: 00/06/12 UTC on 2018-01-01, 2019-01-01, 2020-01-01
- native 0.25° ROI 38.25–42.0°N, 114.0–119.75°E (16×24 points)

The preflight estimated 149,506,560 uncompressed source-chunk bytes under a
fixed 192 MiB cap. No interpolation was performed. The published workflow
artifact retained the source receipt and a 71,684-byte cropped NetCDF with
SHA256 `b200256a6dea3475444cacb90be0bab514b7353bd4cd60c0745fd2c39c9855d2`.

Native, generic-recursive and process-recursive R7 paths each completed two CPU
optimizer updates and one held-out +6 h forward/evaluation. The process model
used `process_weight=0`: a four-surface-variable bundle cannot supervise the
pressure-level process diagnostics. These runs prove real-data integration only,
not forecast skill.

## Pressure-level fallback under investigation

Issue #41 probes the public NSF/NCAR ERA5 AWS NetCDF archive using anonymous S3
plus HDF5 metadata/range reads. Known pressure-analysis files are organized by
variable and day, e.g. parameter codes:

- geopotential: 128_129_z
- temperature: 128_130_t
- u wind: 128_131_u
- v wind: 128_132_v
- specific humidity: 128_133_q

A pressure subset will be attempted only if internal NetCDF/HDF5 chunk geometry
makes the selected time/level/ROI transfer bounded under an explicit cap. If the
files are contiguous or whole-global chunks, the source is rejected for this
low-budget phase rather than brute-force downloaded.

## Scientific limits

The current real surface artifact is intentionally tiny. It is not the
multi-year multivariate East-Asia dataset needed for journal results. The next
meaningful milestone is a real pressure-level pilot that can exercise anchored
process diagnostics, followed by a larger but still budgeted temporal sample.
