"""Separate immutable pin for the actual #49 four-season source, not a fallback."""
from __future__ import annotations
from .pressure_pilot_replay import _verify_pilot_profile, _copy_verified_pair

PINNED_SEASONAL_SHA256 = '3b5d7df8973a46d522508a3b483a52394a07c0f9d2db6e07894adb1bcb41e5bb'


def verify_seasonal_pilot(source, receipt):
    return _verify_pilot_profile(source, receipt, pin=PINNED_SEASONAL_SHA256,
        profile='four-season', origin='GitHub Actions run 35884443083, artifact 10762805966')


def copy_verified_seasonal_pilot(source, receipt, output_source, output_receipt):
    return _copy_verified_pair(source, receipt, output_source, output_receipt,
        verifier=verify_seasonal_pilot, pin=PINNED_SEASONAL_SHA256)
