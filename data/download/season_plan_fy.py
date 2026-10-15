"""Frozen full-year plan arithmetic for the FY (2000-2021) acquisition.

Derived from ``season_plan_s1`` without touching it: the S1 plan is frozen to the
four months Jan/Apr/Jul/Oct at 30 days, while this path tiles **every calendar
month** of a year so the train span gets uniform full-year coverage. The plan is
pure arithmetic over explicit UTC windows, so it is testable without network.

Each block is one calendar month, so its stamp count varies with the month length
(28..31 days x 4 init hours = 112..124 stamps); the downloader's per-part decoded
budget is sized for the longest month.
"""
from __future__ import annotations

import calendar

STAMPS_PER_DAY = 4
DEFAULT_DEADLINE_SECONDS = 1800.0
MONTHS = tuple(range(1, 13))
MAX_YEARS_PER_BATCH = 8
#: Longest month part: 31 days x 4 init hours.
MAX_STAMPS_PER_PART = 31 * STAMPS_PER_DAY


def _month_stamps(year, month):
    """The exact 6-hourly UTC stamps of one calendar month."""
    import pandas as pd

    days = calendar.monthrange(int(year), int(month))[1]
    start = pd.Timestamp(year=int(year), month=int(month), day=1)
    return [start + pd.Timedelta(hours=6 * step) for step in range(days * STAMPS_PER_DAY)]


def full_year_plan(years, *, months=MONTHS):
    """Calendar-month plan: contiguous 6-hourly UTC windows covering each month.

    Unlike the S1 season plan this spans every month, so a year's blocks
    concatenate into one contiguous 6-hourly axis from Jan 1 00:00 to Dec 31 18:00
    (the 18:00 UTC of the last day). Leap years are handled by ``calendar``.
    """
    years = [int(year) for year in years]
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be a nonempty set of unique integers")
    if len(years) > MAX_YEARS_PER_BATCH:
        raise ValueError(f"at most {MAX_YEARS_PER_BATCH} years per batch")
    if not 1940 <= min(years) and max(years) <= 2026:
        raise ValueError("years must fall inside the pinned source range 1940..2026")
    names = [int(month) for month in months]
    if not names or len(set(names)) != len(names) or not set(names) <= set(MONTHS):
        raise ValueError(f"months must be a nonempty subset of {list(MONTHS)}")
    blocks = []
    for year in years:
        for month in names:
            stamps = _month_stamps(year, month)
            if sorted({stamp.hour for stamp in stamps}) != [0, 6, 12, 18]:
                raise ValueError("month stamps must cover the four UTC init hours")
            if len(stamps) > MAX_STAMPS_PER_PART:
                raise ValueError("month window exceeds the longest frozen part")
            blocks.append({"month": month, "year": year,
                           "first_time": stamps[0].isoformat(),
                           "last_time": stamps[-1].isoformat(),
                           "stamps": [stamp.isoformat() for stamp in stamps]})
    total = sum(len(block["stamps"]) for block in blocks)
    return {"years": sorted(years), "months": names, "stamps_per_day": STAMPS_PER_DAY,
            "total_stamps": total, "parts": len(blocks), "blocks": blocks}


def planned_stamps(plan):
    """Every planned stamp, as a validated increasing DatetimeIndex."""
    import pandas as pd

    stamps = [pd.Timestamp(value) for block in plan["blocks"] for value in block["stamps"]]
    if len(stamps) != plan["total_stamps"]:
        raise ValueError("plan stamp count mismatch")
    if len(set(stamps)) != len(stamps):
        raise ValueError("plan contains duplicate stamps")
    return pd.DatetimeIndex(stamps)


def year_stamp_counts(years, *, months=MONTHS):
    """Per-year planned stamp counts, for the frozen budget arithmetic."""
    counts = {}
    for year in years:
        counts[int(year)] = sum(len(_month_stamps(year, month)) for month in months)
    return counts


def plan_report(years, *, months=MONTHS):
    """Read-only plan summary: stamps, estimated network and decoded bytes.

    The per-stamp rates are the registered batch-3 measurements (network) and the
    M2 measurement (decoded), so the figures stay comparable with the batches
    whose receipts the campaign already cites.
    """
    plan = full_year_plan(years, months=months)
    stamps = plan["total_stamps"]
    per_stamp_network = 57084203564 / 1920  # registered batch-3 network measurement
    per_stamp_decoded = 9482837880 / 120  # measured on the M2 part A
    return {"plan": plan,
            "year_stamp_counts": year_stamp_counts(years, months=months),
            "estimated_network_bytes": int(round(stamps * per_stamp_network)),
            "estimated_decoded_bytes": int(round(stamps * per_stamp_decoded)),
            "estimate_source": ("per-stamp network from the registered batch-3 receipt "
                                "(57,084,203,564 B / 1920 stamps) and decoded from the M2 "
                                "part A; planning figures only, not a guarantee"),
            "limitations": ["estimates reuse registered per-stamp rates; the actual rate "
                            "depends on S3/icechunk latency at run time"]}
