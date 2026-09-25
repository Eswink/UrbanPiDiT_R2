"""SSRF boundary tests for the download-layer public opener (CWE-918 guard).

Introduced with decision-free hardening after the sealed Mimosa deep scan
flagged two SSRF findings in the download layer. Offline-safe: every case
resolves locally (IP literals, localhost) — no DNS or connection is needed
for a refusal, and the allow case uses a public IP literal.
"""
from __future__ import annotations

import pytest

from data.download.http_public import reject_non_public_host

REJECTED_URLS = [
    ("loopback", "http://127.0.0.1/beijing.zip"),
    ("ipv6 loopback", "http://[::1]/beijing.zip"),
    ("private ten", "http://10.0.0.8/beijing.zip"),
    ("private 192", "http://192.168.1.5/beijing.zip"),
    ("private 172", "http://172.16.0.9/beijing.zip"),
    ("cloud metadata", "http://169.254.169.254/latest/meta-data/"),
    ("localhost name", "http://localhost/beijing.zip"),
    ("non http scheme", "ftp://archive.ics.uci.edu/beijing.zip"),
    ("no scheme", "archive.ics.uci.edu/beijing.zip"),
]

ALLOWED_URLS = [
    ("public ip literal", "https://93.184.216.34/static/public/381/beijing+pm2.5.zip"),
]


@pytest.mark.parametrize("label,url", REJECTED_URLS,
                         ids=[c[0].replace(" ", "-") for c in REJECTED_URLS])
def test_non_public_targets_are_refused(label, url):
    with pytest.raises(ValueError) as excinfo:
        reject_non_public_host(url)
    assert "拒绝" in str(excinfo.value) or "仅允许" in str(excinfo.value), label


@pytest.mark.parametrize("label,url", ALLOWED_URLS,
                         ids=[c[0].replace(" ", "-") for c in ALLOWED_URLS])
def test_public_targets_pass_validation(label, url):
    reject_non_public_host(url)
