"""Named urban static-covariate sources: catalog, tile grids and guarded HTTP.

Decision 0063 / plan 0027 / goal ``docs/goals/s3-urban-static-covariates.md``.
This is the only new network surface of the round and it lives under
``data/download/`` (R-016). Every request goes through
:func:`data.download.http_public.open_public`: the host is resolved once, every
answer must be global, the socket is connected to the validated addresses, every
redirect hop is re-validated and a timeout is applied. The downloader keeps its
own byte cap on top of that.

The catalog is data-only. Per layer it records the URL template, release/version,
licence, native resolution, the tile grid that covers the R7 frozen ROI and the
aggregation rule. Nothing here interpolates, synthesises or substitutes: an
unreachable endpoint becomes a ``failed-no-fallback`` record (R-008) and the
layer is dropped, never replaced by another layer or by synthetic values.

The GHSL grid origin below was **measured**, not assumed: tile ``R6_C28`` was
fetched and its GeoTIFF reported ``lon 89.992083..99.992083`` / ``lat
29.099583..39.099583`` at 3 arcsec (12000x12000 px). The row index grows
southward and the column index eastward, both at 10 deg spacing.
"""
from __future__ import annotations

import hashlib
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from .http_public import open_public

# --- R7 frozen ROI (data/download/read_plan_frozen.py:34-36) ------------------
ROI_SOUTH, ROI_NORTH, ROI_WEST, ROI_EAST = 27.0, 43.0, 107.0, 123.0
GRID_SPACING_DEG = 0.25
GRID_HEIGHT = GRID_WIDTH = 65
# Aggregation uses cell *edges*, not centres.
EDGE_SOUTH = ROI_SOUTH - GRID_SPACING_DEG / 2   # 26.875
EDGE_NORTH = ROI_NORTH + GRID_SPACING_DEG / 2   # 43.125
EDGE_WEST = ROI_WEST - GRID_SPACING_DEG / 2     # 106.875
EDGE_EAST = ROI_EAST + GRID_SPACING_DEG / 2     # 123.125

WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
GHSL_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
COPERNICUS_DEM_30M_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
LANDSAN_ROOT = "https://landscan.ornl.gov"

WORLDCOVER_VERSIONS = {2020: "v100", 2021: "v200"}
WORLDCOVER_RESOLUTION_M = 10.0
GHSL_BUILT_S_EPOCHS = (2000, 2005, 2010, 2015, 2020)
GHSL_POP_EPOCHS = (2000, 2005, 2010, 2015, 2020)
GHSL_BUILT_H_EPOCH = 2018
GHSL_BUILT_H_VARIANT = "ANBH"       # Average Net Building Height, metres
GHSL_NATIVE_RESOLUTION = "3 arcsec (~100 m), EPSG:4326"

# Measured GHSL 4326_3ss tile grid (see module docstring).
GHSL_TILE_SPAN_DEG = 10.0
GHSL_TILE_ORIGIN_LON = 89.99208295968086
GHSL_TILE_ORIGIN_COL = 28
GHSL_TILE_ORIGIN_LAT = 29.099583419121082
GHSL_TILE_ORIGIN_ROW = 6

WORLDCOVER_TILE_SPAN_DEG = 3.0

# WorldCover class codes (ESA WorldCover product user manual, v200).
WORLDCOVER_CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "herbaceous_wetland", 95: "mangrove", 100: "moss_lichen",
}

LICENCE_CC_BY_4 = "CC-BY-4.0"
LICENCE_URL_CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
WORLDCOVER_SOURCE_PAGE = "https://registry.opendata.aws/esa-worldcover/"
GHSL_SOURCE_PAGE = f"{GHSL_BASE}/GHS_BUILT_S_GLOBE_R2023A"
GHSL_COPYRIGHT_URL = (
    f"{GHSL_BASE}/GHS_BUILT_S_GLOBE_R2023A/"
    "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss/copyright.txt"
)
DEM_SOURCE_PAGE = "https://registry.opendata.aws/copernicus-dem/"
LANDSAN_SOURCE_PAGE = f"{LANDSAN_ROOT}/"


def _format_lat(value: int) -> str:
    return f"{'N' if value >= 0 else 'S'}{abs(value):02d}"


def _format_lon(value: int) -> str:
    return f"{'E' if value >= 0 else 'W'}{abs(value):03d}"


def worldcover_tile_id(lat: int, lon: int) -> str:
    """3 deg WorldCover tile id from its south-west corner, e.g. ``N39E114``."""
    return _format_lat(lat) + _format_lon(lon)


def worldcover_tiles() -> list[str]:
    """Every 3 deg tile whose extent intersects the ROI edge box."""
    first_lat = math.floor(EDGE_SOUTH / WORLDCOVER_TILE_SPAN_DEG) * 3
    last_lat = math.floor(EDGE_NORTH / WORLDCOVER_TILE_SPAN_DEG) * 3
    first_lon = math.floor(EDGE_WEST / WORLDCOVER_TILE_SPAN_DEG) * 3
    last_lon = math.floor(EDGE_EAST / WORLDCOVER_TILE_SPAN_DEG) * 3
    return [
        worldcover_tile_id(lat, lon)
        for lat in range(first_lat, last_lat + 1, 3)
        for lon in range(first_lon, last_lon + 1, 3)
    ]


def worldcover_url(tile: str, year: int) -> str:
    version = WORLDCOVER_VERSIONS[year]
    return (f"{WORLDCOVER_BASE}/{version}/{year}/map/"
            f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")


def ghsl_tile_id(row: int, col: int) -> str:
    return f"R{row}_C{col}"


def ghsl_tiles() -> list[tuple[int, int]]:
    """Every 10 deg GHSL tile whose extent intersects the ROI edge box."""
    first_col = GHSL_TILE_ORIGIN_COL + math.floor(
        (EDGE_WEST - GHSL_TILE_ORIGIN_LON) / GHSL_TILE_SPAN_DEG)
    last_col = GHSL_TILE_ORIGIN_COL + math.floor(
        (EDGE_EAST - GHSL_TILE_ORIGIN_LON) / GHSL_TILE_SPAN_DEG)
    first_row = GHSL_TILE_ORIGIN_ROW - math.floor(
        (EDGE_NORTH - GHSL_TILE_ORIGIN_LAT) / GHSL_TILE_SPAN_DEG)
    last_row = GHSL_TILE_ORIGIN_ROW - math.floor(
        (EDGE_SOUTH - GHSL_TILE_ORIGIN_LAT) / GHSL_TILE_SPAN_DEG)
    return [(row, col) for row in range(first_row, last_row + 1)
            for col in range(first_col, last_col + 1)]


def ghsl_built_s_url(epoch: int, row: int, col: int) -> str:
    product = f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss"
    stem = f"{product}_V1_0_{ghsl_tile_id(row, col)}"
    return f"{GHSL_BASE}/GHS_BUILT_S_GLOBE_R2023A/{product}/V1-0/tiles/{stem}.zip"


def ghsl_built_h_url(row: int, col: int) -> str:
    product = f"GHS_BUILT_H_{GHSL_BUILT_H_VARIANT}_E{GHSL_BUILT_H_EPOCH}_GLOBE_R2023A_4326_3ss"
    stem = f"{product}_V1_0_{ghsl_tile_id(row, col)}"
    return f"{GHSL_BASE}/GHS_BUILT_H_GLOBE_R2023A/{product}/V1-0/tiles/{stem}.zip"


def ghsl_pop_url(epoch: int, row: int, col: int) -> str:
    product = f"GHS_POP_E{epoch}_GLOBE_R2023A_4326_3ss"
    stem = f"{product}_V1_0_{ghsl_tile_id(row, col)}"
    return f"{GHSL_BASE}/GHS_POP_GLOBE_R2023A/{product}/V1-0/tiles/{stem}.zip"


def _tile_rows(prefix: str, urls: list[str], *, epoch: int | None) -> list[dict]:
    rows = []
    for url in urls:
        stem = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        rows.append({"tile": stem.replace(prefix + "_V1_0_", ""), "url": url,
                     "epoch": epoch})
    return rows


def layer_catalog() -> list[dict]:
    """The named layers of decision 0063 that are in scope for this round.

    Each row carries its own tile list, so the endpoint report and the protocol
    can both be rebuilt from this single source of truth.
    """
    ghsl = ghsl_tiles()
    layers = []
    for year in sorted(WORLDCOVER_VERSIONS):
        tiles = worldcover_tiles()
        layers.append({
            "name": f"worldcover_{WORLDCOVER_VERSIONS[year]}_{year}",
            "role": "land_cover",
            "source": "ESA WorldCover", "version": WORLDCOVER_VERSIONS[year],
            "year": year, "native_resolution_m": WORLDCOVER_RESOLUTION_M,
            "crs": "EPSG:4326", "licence": LICENCE_CC_BY_4,
            "licence_url": LICENCE_URL_CC_BY_4, "source_page": WORLDCOVER_SOURCE_PAGE,
            "urls": [worldcover_url(tile, year) for tile in tiles],
            "tile_prefix": f"ESA_WorldCover_10m_{year}_{WORLDCOVER_VERSIONS[year]}",
            "aggregation": "class area fraction per cell edge box",
            "variables": [f"lc_{name}_frac" for name in
                          ("built", "tree", "crop", "water", "bare")],
        })
    for epoch in GHSL_BUILT_S_EPOCHS:
        layers.append({
            "name": f"ghsl_built_s_{epoch}", "role": "built_surface",
            "source": "GHSL GHS-BUILT-S R2023A", "version": "R2023A",
            "epoch": epoch, "native_resolution_m": 100.0, "crs": "EPSG:4326",
            "licence": LICENCE_CC_BY_4, "licence_url": LICENCE_URL_CC_BY_4,
            "source_page": GHSL_SOURCE_PAGE,
            "unit": "built square metres per native cell",
            "urls": [ghsl_built_s_url(epoch, r, c) for r, c in ghsl],
            "tile_prefix": f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss",
            "aggregation": "sum(built m^2) / cell area",
            "variables": ["built_surface_frac"],
        })
    layers.append({
        "name": f"ghsl_built_h_{GHSL_BUILT_H_EPOCH}", "role": "building_height",
        "source": "GHSL GHS-BUILT-H R2023A (ANBH)", "version": "R2023A",
        "epoch": GHSL_BUILT_H_EPOCH, "native_resolution_m": 100.0,
        "crs": "EPSG:4326", "licence": LICENCE_CC_BY_4,
        "licence_url": LICENCE_URL_CC_BY_4, "source_page": GHSL_SOURCE_PAGE,
        "unit": "metres, average net building height of the built surface",
        "urls": [ghsl_built_h_url(r, c) for r, c in ghsl],
        "tile_prefix": f"GHS_BUILT_H_ANBH_E{GHSL_BUILT_H_EPOCH}_GLOBE_R2023A_4326_3ss",
        "aggregation": "area-weighted mean over native cells with a value",
        "variables": ["building_height_m"],
    })
    for epoch in GHSL_POP_EPOCHS:
        layers.append({
            "name": f"ghsl_pop_{epoch}", "role": "population",
            "source": "GHSL GHS-POP R2023A", "version": "R2023A",
            "epoch": epoch, "native_resolution_m": 100.0, "crs": "EPSG:4326",
            "licence": LICENCE_CC_BY_4, "licence_url": LICENCE_URL_CC_BY_4,
            "source_page": GHSL_SOURCE_PAGE,
            "unit": "population counts per native cell",
            "urls": [ghsl_pop_url(epoch, r, c) for r, c in ghsl],
            "tile_prefix": f"GHS_POP_E{epoch}_GLOBE_R2023A_4326_3ss",
            "aggregation": "sum(counts) / cell area",
            "variables": ["pop_density_km2"],
        })
    return layers


# --- guarded HTTP -------------------------------------------------------------

#: In-process audit journal of probe failures. Every unreachable endpoint is
#: journalled here as well as returned to the caller, so a failed acquisition
#: leaves a trace even if the caller drops the returned record (R-008).
PROBE_FAILURES: list[dict] = []


def _as_int(value):
    text = "" if value is None else str(value).strip()
    return int(text) if text.lstrip("+-").isdigit() else None


def _journal_failure(record: dict) -> dict:
    """Journal one failed probe and hand the same record back to the caller."""
    PROBE_FAILURES.append(record)
    return record


def _classify_failure(record: dict) -> str:
    """Say whether a failed probe means *absent* or merely *transient*.

    Decision 0073 §①: a transport timeout is not evidence that a tile does not
    exist. ``attempt03`` recorded ``ghsl_built_s_2000/2005/2010`` as
    ``failed-no-fallback`` on nothing but ``read timeout``; a live re-probe found
    45/45 tiles answering HTTP 200. Only a 4xx (the server positively says the
    object is not there, or not permitted) may drop a named layer. 5xx, timeouts,
    DNS and TLS failures are transient and must be retried, not buried.
    """
    status = record.get("http_status")
    if status is None:
        return "transient"
    if 400 <= status < 500:
        return "absent"
    return "transient"


#: Timeouts for successive probe attempts. Widening them is the cheap half of the
#: fix for 0073 §①: the JRC endpoint throttles under load, and a single 60 s
#: timeout used to condemn a whole layer.
PROBE_TIMEOUTS = (60.0, 90.0, 120.0)
PROBE_RETRY_SLEEP = (2.0, 4.0)


def probe_with_retry(url: str, *, range_bytes: int | None = None,
                     timeouts=PROBE_TIMEOUTS, sleep=None) -> dict:
    """Probe once per timeout in ``timeouts``; stop at the first non-transient answer.

    The returned record carries ``attempts`` and ``failure_class`` of ``"none"``,
    ``"absent"`` or ``"transient"``. Only ``absent`` is grounds for dropping a
    layer (decision 0073 §①). A ``transient`` final result means the measurement
    itself is unreliable, so the caller must fail closed rather than publish a
    hole as a zero.
    """
    if sleep is None:
        sleep = time.sleep
    record: dict = {}
    for attempt, timeout in enumerate(timeouts, 1):
        record = probe(url, range_bytes=range_bytes, timeout=timeout)
        record["attempts"] = attempt
        if record["reachable"]:
            record["failure_class"] = "none"
            return record
        record["failure_class"] = _classify_failure(record)
        # A positive 4xx is an answer: retrying cannot change it.
        if record["failure_class"] == "absent":
            return record
        if attempt < len(timeouts):
            sleep(PROBE_RETRY_SLEEP[min(attempt - 1, len(PROBE_RETRY_SLEEP) - 1)])
    return record


def _request(url: str, *, method: str, range_bytes: int | None = None):
    headers = {}
    if range_bytes is not None:
        headers["Range"] = f"bytes=0-{range_bytes - 1}"
    return urllib.request.Request(url, headers=headers, method=method)


def probe(url: str, *, range_bytes: int | None = None, timeout: float = 60.0) -> dict:
    """Read-only HEAD (or range GET) through the pinned public opener.

    An HTTP error status is a *result*, not an exception: the caller records the
    code so an unreachable layer can be dropped with a ``failed-no-fallback``
    record (R-008). Only transport-level failures raise, and they are reported
    with the same shape.
    """
    request = _request(url, method="HEAD" if range_bytes is None else "GET",
                       range_bytes=range_bytes)
    try:
        with open_public(request, timeout=timeout) as response:
            return {
                "url": url, "method": request.get_method(), "http_status": response.status,
                "content_length": _as_int(response.headers.get("Content-Length")),
                "content_type": response.headers.get("Content-Type"),
                "content_range": response.headers.get("Content-Range"),
                "reachable": 200 <= response.status < 300,
            }
    except urllib.error.HTTPError as error:
        return _journal_failure(
            {"url": url, "method": request.get_method(), "http_status": error.code,
             "content_length": _as_int(error.headers.get("Content-Length")),
             "content_type": error.headers.get("Content-Type"),
             "reachable": False, "error": f"HTTP {error.code}"})
    except Exception as error:  # SSRF refusal, DNS, timeout, TLS, ...
        return _journal_failure(
            {"url": url, "method": request.get_method(), "http_status": None,
             "content_length": None, "content_type": None, "reachable": False,
             "error": f"{type(error).__name__}: {error}"})


def download(url: str, destination: Path, *, byte_cap: int, timeout: float = 300.0) -> dict:
    """Stream one tile to ``destination`` through the pinned opener, byte-capped.

    The digest is computed while streaming, so a partially written file never
    yields a valid sha256 and a cap overrun removes the file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    request = _request(url, method="GET")
    try:
        with open_public(request, timeout=timeout) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > byte_cap:
                    raise RuntimeError(f"download exceeded the byte cap {byte_cap}: {url}")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"url": url, "path": destination.name, "bytes": total,
            "sha256": digest.hexdigest()}
