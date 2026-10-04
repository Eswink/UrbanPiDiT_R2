"""Frozen season-plan arithmetic for the S1 four-season acquisition.

Split out of ``data/download/earthmover_spatial_s1.py`` (R-021): the plan is
pure arithmetic over explicit UTC windows, so it is testable without any
network access, and the downloader keeps only the read/attest/merge machinery.
"""
from __future__ import annotations

from .earthmover_pilot import DecodedBudget  # noqa: F401 - re-exported for the downloader

SEASON_MONTHS = {"winter": 1, "spring": 4, "summer": 7, "autumn": 10}
SEASON_DAYS = 30
STAMPS_PER_DAY = 4
SEASON_STAMPS = SEASON_DAYS * STAMPS_PER_DAY
DEFAULT_PART_DAYS = 30
DEFAULT_DEADLINE_SECONDS = 1800.0
MAX_SEASONS_PER_YEAR = 4
MAX_YEARS_PER_BATCH = 4


def season_plan(years, *, season_names=tuple(SEASON_MONTHS), days=SEASON_DAYS):
    """Explicit four-season plan: contiguous UTC windows on the four init hours.

    A season window starts at 00:00 UTC on the 1st of its month and runs
    ``days`` days, so every stamp lands on 00/06/12/18 and the window is exactly
    6-hourly continuous. Windows never span a month boundary for the frozen
    30-day length and the chosen months.
    """
    import pandas as pd

    years = [int(year) for year in years]
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be a nonempty set of unique integers")
    if len(years) > MAX_YEARS_PER_BATCH:
        raise ValueError(f"at most {MAX_YEARS_PER_BATCH} years per batch")
    if not 1940 <= min(years) and max(years) <= 2026:
        raise ValueError("years must fall inside the pinned source range 1940..2026")
    if days != SEASON_DAYS:
        raise ValueError(f"the frozen season length is {SEASON_DAYS} days")
    names = list(season_names)
    if not names or len(set(names)) != len(names) or not set(names) <= set(SEASON_MONTHS):
        raise ValueError(f"season_names must be a nonempty subset of {sorted(SEASON_MONTHS)}")
    blocks = []
    for year in years:
        for name in names:
            start = pd.Timestamp(year=year, month=SEASON_MONTHS[name], day=1)
            steps = days * 24 // 6
            stamps = [start + pd.Timedelta(hours=6 * step) for step in range(steps)]
            if len(stamps) != SEASON_STAMPS:
                raise ValueError("season window must hold exactly 120 stamps")
            if sorted({stamp.hour for stamp in stamps}) != [0, 6, 12, 18]:
                raise ValueError("season stamps must cover the four UTC init hours")
            blocks.append({"season": name, "year": year,
                           "first_time": stamps[0].isoformat(),
                           "last_time": stamps[-1].isoformat(),
                           "stamps": [stamp.isoformat() for stamp in stamps]})
    return {"years": sorted(years), "season_names": names, "days": days,
            "stamps_per_season": SEASON_STAMPS, "total_stamps": SEASON_STAMPS * len(blocks),
            "blocks": blocks}


def _planned_stamps(plan):
    import pandas as pd

    stamps = [pd.Timestamp(value) for block in plan["blocks"] for value in block["stamps"]]
    if len(stamps) != plan["total_stamps"]:
        raise ValueError("plan stamp count mismatch")
    if len(set(stamps)) != len(stamps):
        raise ValueError("plan contains duplicate stamps")
    return stamps


def plan_report(years, *, season_names=tuple(SEASON_MONTHS)):
    """Read-only plan summary: stamps, estimated network and decoded bytes."""
    plan = season_plan(years, season_names=season_names)
    stamps = plan["total_stamps"]
    per_stamp_network = 7252462353 / 240  # measured on the M2 acquisition
    per_stamp_decoded = 9482837880 / 120  # measured on the M2 part A
    return {"plan": plan,
            "estimated_network_bytes": int(round(stamps * per_stamp_network)),
            "estimated_decoded_bytes": int(round(stamps * per_stamp_decoded)),
            "estimate_source": ("per-stamp M2 measured network 7252462353/240 and decoded "
                                "9482837880/120; planning figures only, not a guarantee"),
            "limitations": ["estimates reuse the M2 per-stamp rates; the actual rate depends on "
                            "S3/icechunk latency at run time"]}
