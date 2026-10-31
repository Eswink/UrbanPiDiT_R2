"""Phase 1 of the urban static round: read-only endpoint verification (decision 0063).

Extracted from ``scripts/r7_urban_static_round.py`` when that driver crossed the
R-051 file-size cap (decision 0073). This is the first hard gate of the round: it
dials every named tile, records HTTP status / content length / licence page URL /
access date, and separates three outcomes that used to be conflated:

* ``reachable`` — at least one tile answered and none is known absent;
* ``failed-no-fallback`` — every tile answered with a 4xx, i.e. positive evidence
  that the object is not there (R-008);
* ``unreliable`` — every tile failed transiently (timeout / 5xx). The layer is
  neither present nor absent: the *measurement* failed. Freezing it would publish
  its holes as zeroes, so ``freeze`` refuses.

The distinction exists because ``attempt03`` recorded three real layers as
``failed-no-fallback`` on nothing but ``read timeout``; a live re-probe found 45/45
tiles answering HTTP 200 (decision 0073 §①).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from data.download import urban_static_sources as sources

__all__ = ["verify_endpoints", "require_dropped_reasons", "require_no_unreliable_layers",
           "DROPPED_LAYERS"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path, body) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _transient_reason(probes) -> str:
    """A reason for a layer whose probes all failed in a way that is *not* an answer.

    Decision 0073 §①: this is deliberately worded as an unreliable measurement, not
    as "the layer does not exist". Keeping the two apart is what stops a throttling
    episode from being remembered as a data gap.
    """
    observed = sorted({(str(p.get("http_status")) if p.get("http_status") is not None
                        else str(p.get("error") or "no response")) for p in probes})
    return (f"every probe was transient, not a definitive answer ({len(probes)} probed: "
            f"{', '.join(observed)}); the layer is NOT established as absent — "
            f"the measurement failed and must be retried, no substitute was written")


def _endpoint_verdict(probes) -> str:
    """Classify a layer from its probes (decision 0073 §①).

    ``failed-no-fallback`` requires positive evidence of absence: every tile answered
    with a 4xx. A layer whose tiles merely timed out is ``unreliable`` — it must not
    be published (its holes would become zeroes) and it must not be recorded as a
    data gap either.
    """
    if any(p["reachable"] for p in probes):
        return ("reachable-with-absent-tiles"
                if any(p.get("failure_class") == "absent" for p in probes) else "reachable")
    if all(p.get("failure_class") == "absent" for p in probes):
        return "failed-no-fallback"
    return "unreliable"


def verify_endpoints(new_attempt) -> dict:
    attempt = new_attempt()
    layers, dropped, unreliable = [], [], []
    for layer in sources.layer_catalog():
        probes = [sources.probe_with_retry(url) for url in layer["urls"]]
        reachable = [p for p in probes if p["reachable"]]
        absent = [p for p in probes if p.get("failure_class") == "absent"]
        lengths = [p["content_length"] for p in reachable if p.get("content_length")]
        # A published product may omit a tile by design (ESA WorldCover publishes no
        # all-ocean 3 deg tile). That is a *per-tile* gap: the layer is dropped only
        # when there is positive evidence of absence, and the absent tiles are carried
        # into the protocol so a coverage array can expose them.
        verdict = _endpoint_verdict(probes)
        record = {
            "name": layer["name"], "role": layer["role"], "source": layer["source"],
            "version": layer["version"], "native_resolution_m": layer["native_resolution_m"],
            "crs": layer["crs"], "licence": layer["licence"],
            "licence_url": layer["licence_url"], "source_page": layer["source_page"],
            "tile_count": len(probes), "reachable_tiles": len(reachable),
            "absent_tiles": [p["url"] for p in absent],
            "absent_failed_no_fallback": bool(absent),
            "transient_tiles": [p["url"] for p in probes
                                if p.get("failure_class") == "transient"],
            "total_content_length_bytes": sum(lengths) if lengths else None,
            "reachable_urls": [p["url"] for p in reachable],
            "probes": probes, "verdict": verdict,
        }
        if verdict == "failed-no-fallback":
            # R-008: a dropped layer must carry an auditable reason. The observed probe
            # failures are the measurement; "no tile answered" alone is not a reason.
            record["reason"] = _unreachable_reason(absent)
            record["failed_no_fallback"] = True
            dropped.append(record)
        elif verdict == "unreliable":
            record["reason"] = _transient_reason(probes)
            record["failed_no_fallback"] = False
            unreliable.append(record)
        else:
            layers.append(record)
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
        "unreliable": unreliable,
        "licence_pages": licence_pages,
        "scientific_claim": False,
        "limitations": [
            "reachability is what this report measures; it is not a licence opinion",
            "search summaries are not evidence (R-050); every status here was dialled",
            "a transient failure (timeout/5xx) is recorded as unreliable, never as an "
            "absent layer: one timeout is not evidence that a tile does not exist",
        ],
    }
    _write_json(attempt / "endpoint_report.json", report)
    return {"attempt": str(attempt), "layers_reachable": [r["name"] for r in layers],
            "layers_dropped": [r["name"] for r in report["dropped"]],
            "layers_unreliable": [r["name"] for r in unreliable],
            "layers_with_absent_tiles": {r["name"]: len(r["absent_tiles"]) for r in layers
                                         if r["absent_tiles"]},
            "status": "ok" if layers else "no-reachable-layer"}




def require_no_unreliable_layers(unreliable) -> None:
    """Fail closed when a layer's reachability could not be measured (decision 0073 §①).

    An ``unreliable`` layer is one whose tiles all failed transiently. Freezing it
    into a protocol would mean downloading a partial layer and publishing its holes
    as zeroes, which is exactly the defect §② of the same decision describes. The
    layer is neither present nor absent, so the round must stop and retry rather than
    commit to either story.
    """
    if not unreliable:
        return
    detail = "; ".join(f"{entry['name']}: {entry['reason']}" for entry in unreliable)
    raise SystemExit(
        f"refusing to freeze: {len(unreliable)} layer(s) could not be measured "
        f"(every probe transient). Retry the endpoint verification; do not freeze a "
        f"layer whose holes would be published as zeros. {detail}")
