"""Driver for the urban static-covariate data-preparation round (decision 0063 / plan 0027).

The orchestrating workflow can only pass compile-time literals, so every runtime
value is read from disk: the layer catalog is
``data/download/urban_static_sources.py``, the frozen plan is the attempt
directory's ``protocol.json``, and the attempt directory is resolved from the
``outputs/`` tree. Subcommands: ``verify-endpoints`` / ``freeze`` / ``download`` /
``aggregate`` / ``recheck`` / ``publish``.

Phase 1 (``verify-endpoints``) is the hard gate and is read-only: HEAD/range GET per
named layer, recording HTTP status, content-length, licence URL and access date. An
unreachable layer becomes a ``failed-no-fallback`` record and is dropped (R-008);
nothing is synthesised or substituted by another layer.

This round is 0 GPU-h. It never touches ``model/**``, never rebuilds a store and
never writes ``data/raw|interim|processed``. The static fields it publishes are
**derived statistics of real observations**, not a truth field, and the manifest
says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download import urban_static_sources as sources          # noqa: E402
from data.preprocess import urban_static_aggregate as aggregate    # noqa: E402
from data.preprocess import urban_static_recheck as rechecker       # noqa: E402
from data.preprocess.contracts import fresh_outputs                # noqa: E402

STEM = "r7_urban_static_covariates"
OUTPUTS = ROOT / "outputs"
LAYER_ORDER = [layer["name"] for layer in sources.layer_catalog()]

# Budget constants, frozen before any download (decision 0063, plan 0027 §7).
PLANNED_SECONDS = 21600
HARD_CAP_SECONDS = 86400
NETWORK_BYTES_CAP = 40 * 2**30          # 40 GiB, hard stop for the whole round
DISK_HEADROOM_BYTES = 60 * 2**30        # must stay free after the round
SINGLE_TILE_BYTES_CAP = 512 * 2**20     # one tile can never exceed 512 MiB
# Out-of-scope layers of decision 0063, recorded as dropped with their evidence.
DROPPED_LAYERS = [
    {"name": "landscan_global", "role": "population_cross_check",
     "reason": "not anonymously downloadable: ORNL requires an applicant account",
     "evidence": f"{sources.LANDSAN_ROOT} reachable as a website but the data "
                 "catalogue is behind a registration wall; decision 0063 lists "
                 "LandScan as optional and forbids account-gated sources",
     "failed_no_fallback": True},
    {"name": "copernicus_dem_glo30", "role": "terrain", "reason": "not a named layer",
     "evidence": "the AWS Open Data mirror copernicus-dem-30m is anonymous "
                 "(HTTP 200, image/tiff), but decision 0063 does not authorise a DEM "
                 "layer this round; recorded as a finding, not fetched",
     "failed_no_fallback": False},
]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, body) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(body) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _attempt_dirs() -> list[Path]:
    return sorted(p for p in OUTPUTS.glob(f"{STEM}_attempt*") if p.is_dir())


def new_attempt() -> Path:
    numbers = [int(p.name.rsplit("attempt", 1)[-1]) for p in _attempt_dirs()
               if p.name.rsplit("attempt", 1)[-1].isdigit()]
    path = OUTPUTS / f"{STEM}_attempt{max(numbers, default=0) + 1:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_attempt(artifact: str | None = None) -> Path:
    for path in reversed(_attempt_dirs()):
        if artifact is None or (path / artifact).exists():
            return path
    raise SystemExit(f"no attempt directory carrying {artifact!r} under {OUTPUTS}")


def _layer_by_name(name: str) -> dict:
    for layer in sources.layer_catalog():
        if layer["name"] == name:
            return layer
    raise SystemExit(f"unknown layer {name!r}")


# --- phase 1: read-only endpoint verification (the hard gate) -----------------

def _unreachable_reason(probes) -> str:
    """A non-empty, measurement-backed reason for dropping a named layer (R-008).

    The probe journal is the evidence; this turns it into one sentence a reader can
    audit. Never returns an empty string: ``require_dropped_reasons`` fails closed.
    """
    observed = []
    for probe in probes:
        status = probe.get("http_status")
        detail = probe.get("error")
        if status is not None:
            observed.append(f"HTTP {status}")
        elif detail:
            observed.append(str(detail))
        else:
            observed.append("no response")
    codes = ", ".join(sorted(set(observed)))
    return (f"no tile of the layer answered a read-only probe "
            f"({len(probes)} probed: {codes}); dropped as failed-no-fallback, "
            f"no synthetic substitute was written")


def require_dropped_reasons(dropped) -> None:
    """Fail closed unless every dropped named layer carries a non-empty reason.

    R-008 requires an auditable ``failed-no-fallback`` record; a record whose
    ``reason`` is missing or blank cannot be audited, because a reader cannot tell
    "unreachable" from "excluded by authorisation" from "download failed". This is a
    guard, not a convention: it raises rather than publishing an unexplained gap.
    """
    for entry in dropped:
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SystemExit(
                f"refusing to continue: dropped layer {entry.get('name')!r} has no "
                f"reason (R-008). Every dropped named layer must say why it was "
                f"dropped, in words, before the protocol is frozen or published.")


def verify_endpoints() -> dict:
    attempt = new_attempt()
    layers, dropped = [], []
    for layer in sources.layer_catalog():
        probes = [sources.probe(url) for url in layer["urls"]]
        reachable = [p for p in probes if p["reachable"]]
        absent = [p for p in probes if not p["reachable"]]
        lengths = [p["content_length"] for p in reachable if p.get("content_length")]
        # A published product may omit a tile by design (ESA WorldCover publishes no
        # all-ocean 3 deg tile). That is a *per-tile* failed-no-fallback record: the
        # layer is dropped only when it has no reachable tile at all, and the absent
        # tiles are carried into the protocol so a coverage array can expose them.
        verdict = "failed-no-fallback" if not reachable else (
            "reachable-with-absent-tiles" if absent else "reachable")
        record = {
            "name": layer["name"], "role": layer["role"], "source": layer["source"],
            "version": layer["version"], "native_resolution_m": layer["native_resolution_m"],
            "crs": layer["crs"], "licence": layer["licence"],
            "licence_url": layer["licence_url"], "source_page": layer["source_page"],
            "tile_count": len(probes), "reachable_tiles": len(reachable),
            "absent_tiles": [p["url"] for p in absent],
            "absent_failed_no_fallback": bool(absent),
            "total_content_length_bytes": sum(lengths) if lengths else None,
            "reachable_urls": [p["url"] for p in reachable],
            "probes": probes, "verdict": verdict,
        }
        if verdict == "failed-no-fallback":
            # R-008: a dropped layer must carry an auditable reason. The observed probe
            # failures are the measurement; "no tile answered" alone is not a reason.
            record["reason"] = _unreachable_reason(probes)
            record["failed_no_fallback"] = True
        (dropped if verdict == "failed-no-fallback" else layers).append(record)
    licence_pages = [sources.probe(url) for url in (
        sources.WORLDCOVER_SOURCE_PAGE, sources.GHSL_COPYRIGHT_URL,
        sources.LICENCE_URL_CC_BY_4)]
    report = {
        "schema": "r7-urban-static-endpoints-v1", "accessed_utc": _now(),
        "roi": {"south": sources.ROI_SOUTH, "north": sources.ROI_NORTH,
                "west": sources.ROI_WEST, "east": sources.ROI_EAST,
                "edge_south": sources.EDGE_SOUTH, "edge_north": sources.EDGE_NORTH,
                "edge_west": sources.EDGE_WEST, "edge_east": sources.EDGE_EAST,
                "points": [sources.GRID_HEIGHT, sources.GRID_WIDTH],
                "spacing_deg": sources.GRID_SPACING_DEG},
        "gate_items": {
            "ghsl_resolution": "resolved by directory listing: GHSL R2023A publishes "
                               "3 arcsec (~100 m) and 30 arcsec (~1 km) variants of the "
                               "same product; this round takes the 4326_3ss (~100 m) one",
            "copernicus_dem_aws_mirror": "measured anonymous: copernicus-dem-30m S3 "
                                         "returns HTTP 200 image/tiff; not a named layer, "
                                         "so not fetched this round",
        },
        "layers": layers, "dropped": dropped + DROPPED_LAYERS,
        "licence_pages": licence_pages,
        "scientific_claim": False,
        "limitations": [
            "reachability is what this report measures; it is not a licence opinion",
            "search summaries are not evidence (R-050); every status here was dialled",
        ],
    }
    _write_json(attempt / "endpoint_report.json", report)
    return {"attempt": str(attempt), "layers_reachable": [r["name"] for r in layers],
            "layers_dropped": [r["name"] for r in report["dropped"]],
            "layers_with_absent_tiles": {r["name"]: len(r["absent_tiles"]) for r in layers
                                         if r["absent_tiles"]},
            "status": "ok" if layers else "no-reachable-layer"}


# --- phase 2: freeze the acquisition protocol ---------------------------------

def _layer_plan(record: dict) -> dict:
    catalog = _layer_by_name(record["name"])
    return {
        "name": record["name"], "role": record["role"], "source": record["source"],
        "version": record["version"], "epoch": catalog.get("epoch"),
        "year": catalog.get("year"),
        "native_resolution_m": record["native_resolution_m"],
        "crs": record["crs"], "licence": record["licence"],
        "licence_url": record["licence_url"], "source_page": record["source_page"],
        "aggregation": catalog["aggregation"], "variables": catalog["variables"],
        "tile_count": record["tile_count"],
        "absent_tiles": record.get("absent_tiles", []),
        "absent_failed_no_fallback": record.get("absent_failed_no_fallback", False),
        "expected_bytes": record["total_content_length_bytes"],
        "urls": record["reachable_urls"],
        "url_sha256": _digest(record["reachable_urls"]),
    }


def freeze() -> dict:
    attempt = latest_attempt("endpoint_report.json")
    report = _read_json(attempt / "endpoint_report.json")
    require_dropped_reasons(report["dropped"])
    body = {
        "schema": "r7-urban-static-protocol-v1", "frozen_utc": _now(),
        "frozen_before_any_download": True, "scientific_claim": False,
        "roi": report["roi"],
        "grid": {"points": [sources.GRID_HEIGHT, sources.GRID_WIDTH],
                 "spacing_deg": sources.GRID_SPACING_DEG,
                 "lat_order": "descending (row 0 is northernmost)",
                 "lon_order": "ascending", "dtype": "float32"},
        "spatial_resampling": aggregate.SPATIAL_RESAMPLING,
        "area_basis": {"sphere": "WGS84 authalic", "radius_m": aggregate.AUTHALIC_RADIUS_M,
                       "weight": "R^2 * dlon_rad * (sin(lat_top) - sin(lat_bottom))"},
        "derived_statistic_not_truth": (
            "every field is a derived statistic of real native observations aggregated "
            "to the 0.25 deg cell edges; it is NOT interpolation and MUST NOT be used "
            "as an observational truth field"),
        "layers": [_layer_plan(record) for record in report["layers"]],
        "dropped_layers": report["dropped"],
        "output_layout": {
            "native_tiles": "native/<layer>/<file> (kept at native resolution for a "
                            "deferred 1 km round)",
            "static_fields": "published/fields/static_fields.npz",
            "manifest": "published/manifest/manifest.json + BUILD_COMPLETE.json",
            "static_identity": "sha256 over source fingerprints + aggregation spec + "
                               "output bytes; bound to protocol_sha256 at wiring time",
        },
        "budget": {"gpu_hours": 0.0, "planned_seconds": PLANNED_SECONDS,
                   "hard_cap_seconds": HARD_CAP_SECONDS,
                   "network_bytes_cap": NETWORK_BYTES_CAP,
                   "disk_headroom_bytes": DISK_HEADROOM_BYTES,
                   "single_tile_bytes_cap": SINGLE_TILE_BYTES_CAP},
        "limitations": [
            "GHSL multi-epoch layers are five-year steps synthesised across sensor "
            "generations, not annual observations",
            "building_height_m weights ANBH by native cell area over cells that carry "
            "a value; cells with no built surface carry no weight",
            "0.25 deg built-up is a *fraction*; it does not resolve intra-city structure",
        ],
    }
    body["protocol_sha256"] = _digest(body)
    _write_json(attempt / "protocol.json", body)
    return {"attempt": str(attempt), "protocol_sha256": body["protocol_sha256"],
            "layers": [layer["name"] for layer in body["layers"]],
            "expected_bytes": sum(layer["expected_bytes"] or 0 for layer in body["layers"])}


# --- phase 3: download --------------------------------------------------------

def _free_bytes() -> int:
    import shutil as _shutil
    return _shutil.disk_usage(OUTPUTS).free


def _extract_zip_member(archive: Path) -> Path:
    """Extract the single GeoTIFF member of a GHSL tile zip next to the archive.

    The archive stays: it is the sha256-verifiable downloaded artefact, and the
    extracted GeoTIFF is the native-resolution tile the aggregator reads.
    """
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist()
                   if name.lower().endswith((".tif", ".tiff"))]
        if len(members) != 1:
            raise SystemExit(f"{archive.name}: expected exactly one GeoTIFF, got {members}")
        target = archive.with_name(Path(members[0]).name)
        with bundle.open(members[0]) as source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle)
    return target


def download() -> dict:
    attempt = latest_attempt("protocol.json")
    protocol = _read_json(attempt / "protocol.json")
    native = attempt / "native"
    records, total = [], 0
    for layer in protocol["layers"]:
        for url in layer["urls"]:
            destination = native / layer["name"] / url.rsplit("/", 1)[-1]
            if destination.exists() and destination.stat().st_size > 0:
                entry = {"url": url, "path": str(destination.relative_to(attempt)),
                         "bytes": destination.stat().st_size,
                         "sha256": _sha256_file(destination), "reused": True}
            else:
                entry = sources.download(url, destination, byte_cap=SINGLE_TILE_BYTES_CAP)
                entry["path"] = str(destination.relative_to(attempt))
                entry["reused"] = False
            if destination.suffix == ".zip":
                tile = _extract_zip_member(destination)
                entry["extracted"] = str(tile.relative_to(attempt))
                entry["extracted_sha256"] = _sha256_file(tile)
            total += entry["bytes"]
            if total > protocol["budget"]["network_bytes_cap"]:
                raise SystemExit(f"network byte cap exceeded at {url}")
            if _free_bytes() < protocol["budget"]["disk_headroom_bytes"]:
                raise SystemExit(f"disk headroom breached at {url}")
            records.append(entry)
    receipt = {"schema": "r7-urban-static-download-v1", "downloaded_utc": _now(),
               "files": len(records), "total_bytes": total,
               "protocol_sha256": protocol["protocol_sha256"], "records": records}
    _write_json(attempt / "download_receipt.json", receipt)
    return {"attempt": str(attempt), "files": len(records), "total_bytes": total}


# --- phase 4: aggregate -------------------------------------------------------

def _tile_paths(attempt: Path, layer_name: str) -> list[Path]:
    directory = attempt / "native" / layer_name
    return sorted(path for path in directory.glob("*")
                  if path.suffix.lower() in (".tif", ".tiff"))


def _coverage(paths) -> np.ndarray:
    """Coverage of the target grid by the downloaded tiles (1.0 where complete)."""
    import rasterio
    bounds = []
    for path in paths:
        with rasterio.open(path) as dataset:
            box = dataset.bounds
            bounds.append((box.left, box.bottom, box.right, box.top))
    return aggregate.coverage_fraction(bounds).astype(np.float32)


def aggregate_fields() -> dict:
    attempt = latest_attempt("protocol.json")
    protocol = _read_json(attempt / "protocol.json")
    arrays, built_epochs, pop_epochs, receipts = {}, [], [], []

    for year, version in sorted(sources.WORLDCOVER_VERSIONS.items()):
        name = f"worldcover_{version}_{year}"
        paths = _tile_paths(attempt, name)
        result = aggregate.aggregate_land_cover(paths)
        for code, field in (("50", "built"), ("10", "tree"), ("40", "crop"),
                            ("80", "water"), ("60", "bare")):
            arrays[f"lc_{field}_frac_{year}"] = result["fractions"][int(code)].astype(np.float32)
        coverage = _coverage(paths)
        arrays[f"land_cover_coverage_frac_{year}"] = coverage
        arrays[f"land_cover_mapped_frac_{year}"] = result["mapped_fraction"].astype(np.float32)
        receipts.append({"layer": name,
                         "cells_with_data": int((result["mapped_fraction"] > 0).sum()),
                         "cells_below_full_coverage": int((coverage < 1.0).sum()),
                         "min_coverage": float(coverage.min())})

    for layer in protocol["layers"]:
        name, role = layer["name"], layer["role"]
        paths = _tile_paths(attempt, name)
        if role == "built_surface":
            result = aggregate.aggregate_built_surface(paths)
            arrays[f"built_surface_frac_{layer['epoch']}"] = result["fraction"].astype(np.float32)
            built_epochs.append(layer["epoch"])
        elif role == "building_height":
            result = aggregate.aggregate_building_height(paths)
            arrays["building_height_m"] = result["mean"].astype(np.float32)
        elif role == "population":
            result = aggregate.aggregate_population(paths)
            arrays[f"pop_density_km2_{layer['epoch']}"] = result["density"].astype(np.float32)
            pop_epochs.append(layer["epoch"])
        else:
            continue
        receipts.append({"layer": name, "role": role, "epoch": layer.get("epoch"),
                         "cells_with_data": int((np.asarray(list(result.values())[0]) > 0).sum())})

    # Contract names of docs/DATA_SPEC.md: the primary WorldCover year is 2021 (v200)
    # and the multi-epoch GHSL layers are stacked with an explicit epoch axis. The
    # two products keep their own epoch axes: GHS-BUILT-H is 2018 only and must not
    # leak into either stack.
    built_epochs, pop_epochs = sorted(set(built_epochs)), sorted(set(pop_epochs))
    for field in ("built", "tree", "crop", "water", "bare"):
        arrays[f"lc_{field}_frac"] = arrays[f"lc_{field}_frac_2021"]
    arrays["built_surface_frac"] = np.stack(
        [arrays[f"built_surface_frac_{e}"] for e in built_epochs]).astype(np.float32)
    arrays["pop_density_km2"] = np.stack(
        [arrays[f"pop_density_km2_{e}"] for e in pop_epochs]).astype(np.float32)
    arrays["built_surface_frac_epochs"] = np.array(built_epochs, dtype=np.int32)
    arrays["pop_density_km2_epochs"] = np.array(pop_epochs, dtype=np.int32)
    arrays["latitude"] = np.array([sources.ROI_NORTH - sources.GRID_SPACING_DEG * i
                                   for i in range(sources.GRID_HEIGHT)], dtype=np.float32)
    arrays["longitude"] = np.array([sources.ROI_WEST + sources.GRID_SPACING_DEG * j
                                    for j in range(sources.GRID_WIDTH)], dtype=np.float32)

    np.savez(attempt / "static_fields.npz", **arrays)
    downloaded_under = _read_json(attempt / "download_receipt.json")["protocol_sha256"]
    receipt = {"schema": "r7-urban-static-aggregate-v1", "aggregated_utc": _now(),
               "protocol_sha256": protocol["protocol_sha256"],
               "protocol_sha256_at_download": downloaded_under,
               "protocol_amendment": (
                   "after the download the layer plans gained the bookkeeping keys "
                   "`epoch`/`year` that the aggregator needs; acquisition-relevant "
                   "content (urls, tile counts, expected bytes, aggregation rules, "
                   "budget, roi, grid) is unchanged and the digest recorded at download "
                   "time is `protocol_sha256_at_download`"),
               "spatial_resampling": aggregate.SPATIAL_RESAMPLING,
               "area_basis": protocol["area_basis"],
               "arrays": sorted(arrays), "layer_receipts": receipts,
               "fields_sha256": _sha256_file(attempt / "static_fields.npz"),
               "derived_statistic_not_truth": protocol["derived_statistic_not_truth"]}
    _write_json(attempt / "aggregate_receipt.json", receipt)
    return {"attempt": str(attempt), "arrays": len(arrays),
            "fields_sha256": receipt["fields_sha256"]}


# --- phase 5: independent re-derivation ---------------------------------------

def recheck() -> dict:
    attempt = latest_attempt("static_fields.npz")
    protocol = _read_json(attempt / "protocol.json")
    with np.load(attempt / "static_fields.npz") as loaded:
        fields = {name: loaded[name] for name in loaded.files}
    samples = [(7, 64), (32, 32), (55, 5)]
    comparisons, worst = [], 0.0
    for row, col in samples:
        checks = [
            ("worldcover_v200_2021", "land_cover", "lc_built_frac",
             fields["lc_built_frac"][row, col], 50),
            (f"ghsl_built_h_{sources.GHSL_BUILT_H_EPOCH}", "building_height",
             "building_height_m", fields["building_height_m"][row, col], None),
            ("ghsl_built_s_2020", "built_surface", "built_surface_frac",
             fields["built_surface_frac"][-1][row, col], None),
            ("ghsl_pop_2020", "population", "pop_density_km2",
             fields["pop_density_km2"][-1][row, col], None),
        ]
        for layer_name, role, field, published, code in checks:
            derived = rechecker.independent_cell(attempt, layer_name, row, col, role=role)
            value = derived["fraction"][code] if role == "land_cover" else (
                derived["mean"] if role == "building_height" else
                derived["fraction"] if role == "built_surface" else derived["density"])
            delta = abs(float(value) - float(published))
            scale = max(abs(float(published)), 1e-12)
            relative = delta / scale
            worst = max(worst, relative)
            comparisons.append({"row": row, "col": col, "layer": layer_name,
                                "field": field, "independent": float(value),
                                "published": float(published), "abs_delta": delta,
                                "relative_delta": relative})
    receipt = {"schema": "r7-urban-static-recheck-v1", "checked_utc": _now(),
               "protocol_sha256": protocol["protocol_sha256"],
               "samples": samples, "comparisons": comparisons,
               "max_relative_delta": worst, "tolerance": 1e-5,
               "passed": worst <= 1e-5,
               "method": "raw tiles re-read and reduced independently of the "
                         "aggregation accumulator; the tolerance is float32 storage "
                         "precision, not an algorithmic allowance"}
    _write_json(attempt / "recheck_receipt.json", receipt)
    return {"attempt": str(attempt), "max_relative_delta": worst,
            "passed": receipt["passed"]}


# --- phase 6: publish ---------------------------------------------------------

@fresh_outputs("fields_dir", "manifest_dir")
def _publish(fields_dir, manifest_dir, *, attempt, protocol, receipt, recheck_receipt):
    require_dropped_reasons(protocol["dropped_layers"])
    fields_dir, manifest_dir = Path(fields_dir), Path(manifest_dir)
    fields_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = fields_dir / "static_fields.npz"
    target.write_bytes((attempt / "static_fields.npz").read_bytes())
    download = _read_json(attempt / "download_receipt.json")
    identity = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_sha256": _digest({entry["url"]: entry["sha256"]
                                  for entry in download["records"]}),
        "aggregation_spec": {"spatial_resampling": aggregate.SPATIAL_RESAMPLING,
                             "area_basis": protocol["area_basis"],
                             "grid": protocol["grid"]},
        "output_sha256": _sha256_file(target),
        "recheck_max_relative_delta": recheck_receipt["max_relative_delta"],
    }
    identity["static_identity"] = _digest(identity)
    manifest = {
        "schema": "r7-urban-static-manifest-v1", "published_utc": _now(),
        "scientific_claim": False, "roi": protocol["roi"], "grid": protocol["grid"],
        "spatial_resampling": aggregate.SPATIAL_RESAMPLING,
        "derived_statistic_not_truth": protocol["derived_statistic_not_truth"],
        "static_identity": identity["static_identity"], "identity_components": identity,
        "source_sha256": identity["source_sha256"],
        "layers": [{k: layer[k] for k in ("name", "source", "version", "licence",
                                          "licence_url", "native_resolution_m",
                                          "aggregation", "variables", "tile_count")}
                   for layer in protocol["layers"]],
        "dropped_layers": protocol["dropped_layers"],
        "aggregate_receipt_sha256": receipt["fields_sha256"],
        "recheck": {"max_relative_delta": recheck_receipt["max_relative_delta"],
                    "passed": recheck_receipt["passed"]},
        "limitations": protocol["limitations"] + [
            "static_identity is meant to be bound into protocol_sha256 at model "
            "wiring time, never into data_identity",
        ],
    }
    _write_json(manifest_dir / "manifest.json", manifest)
    return manifest


def publish() -> dict:
    attempt = latest_attempt("recheck_receipt.json")
    protocol = _read_json(attempt / "protocol.json")
    receipt = _read_json(attempt / "aggregate_receipt.json")
    recheck_receipt = _read_json(attempt / "recheck_receipt.json")
    # Fail closed (decision 0072 §③). Publishing a static_identity asserts that the
    # fields carry the identity they claim; an independent re-derivation that
    # disagrees with them is exactly the case where that assertion is unproven.
    # Recording `passed: false` inside the manifest is not a gate: attempt03 published
    # at max_relative_delta 9.1e+05 that way.
    if recheck_receipt.get("passed") is not True:
        failed = [c for c in recheck_receipt.get("comparisons", [])
                  if c.get("relative_delta", 0.0) > recheck_receipt.get("tolerance", 1e-5)]
        detail = ", ".join(f"{c['field']}@({c['row']},{c['col']})="
                           f"{c['relative_delta']:.3e}" for c in failed) or "no per-cell detail"
        raise SystemExit(
            f"refusing to publish {attempt.name}: the independent re-derivation did not "
            f"pass (max_relative_delta={recheck_receipt.get('max_relative_delta')!r}, "
            f"tolerance={recheck_receipt.get('tolerance')!r}). Failing comparisons: "
            f"{detail}. Fix the aggregation or the recheck; do not publish.")
    manifest = _publish(fields_dir=attempt / "published" / "fields",
                        manifest_dir=attempt / "published" / "manifest",
                        attempt=attempt, protocol=protocol, receipt=receipt,
                        recheck_receipt=recheck_receipt)
    return {"attempt": str(attempt), "static_identity": manifest["static_identity"],
            "fields": str(attempt / "published" / "fields" / "static_fields.npz")}


COMMANDS = {"verify-endpoints": verify_endpoints, "freeze": freeze, "download": download,
            "aggregate": aggregate_fields, "recheck": recheck, "publish": publish}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="urban static-covariate round driver")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    result = COMMANDS[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
