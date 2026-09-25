"""Public-host HTTP opener for the download layer (SSRF guard, CWE-918).

The download layer is the only place allowed to open network clients (R-016).
Every request through :data:`PUBLIC_OPENER` targets a named public host over
http(s): the initial URL and every redirect hop are validated (scheme, host
resolution) and localhost/loopback/private/link-local/reserved targets are
refused before they are dialed. Redirects are therefore both limited and
re-validated per hop, which also mitigates redirect-based SSRF and narrows
DNS-rebinding windows (each hop re-resolves and re-checks).

Introduced 2026-09-25 after the sealed Mimosa deep scan flagged two SSRF
findings on the direct convenience-opener calls here (scan
`scan-2026-09-25T10-46-34.545Z-e148e037d7f7`; see
docs/R7_SECURITY_SCAN_TRIAGE.md). Mirrors the opener pattern that
`arco_tiny_bounded.py` already used.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request

_ALLOWED_SCHEMES = ("https", "http")
_DEFAULT_PORTS = {"https": 443, "http": 80}


def reject_non_public_host(url: str) -> None:
    """Refuse URLs that are not http(s) to hosts resolving to public addresses.

    Raises ValueError before anything is dialed. Numeric-IP hosts and
    /etc/hosts names resolve locally, so callers stay offline-safe when the
    address is refused or public-by-literal.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"仅允许 http/https 下载地址，得到 scheme={parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("下载地址缺少主机名")
    port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 80)
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise ValueError(
                f"下载主机 {host!r} 解析到非公网地址 {address}，已拒绝（SSRF 防护）"
            )


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        reject_non_public_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


PUBLIC_OPENER = urllib.request.build_opener(_ValidatedRedirectHandler)


def open_public(request: urllib.request.Request, *, timeout: int = 60):
    """Open ``request`` on the validated, redirect-limiting public opener."""
    reject_non_public_host(request.full_url)
    try:
        return PUBLIC_OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"下载请求被拒绝或失败：{exc}") from exc
