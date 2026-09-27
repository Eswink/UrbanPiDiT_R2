"""Public-host HTTP opener for the download layer (SSRF guard, CWE-918).

The download layer is the only place allowed to open network clients (R-016).
Every request through :data:`PUBLIC_OPENER` targets a named public host over
http(s), and the initial URL plus every redirect hop are validated.

**Since #68 the validation constrains the connection, not just the URL.** The
implementation lives in :mod:`data.download.http_pinned`: the host is resolved
once, every answer must be global, and the socket is connected to those already
validated addresses, so a name that re-resolves between the check and the dial
(DNS rebinding) cannot redirect the connection to a private address. Redirects
are limited and re-validated per hop, the proxy policy is declared rather than
inherited from the environment, and HTTPS keeps SNI and certificate
verification on the default verifying context.

This module keeps its historical import path and public names and re-exports the
pinned implementation, so every existing caller gains the stronger behaviour
without changing its imports. The precise boundary is unchanged: this provides
address validation, not an allowlist, not content inspection, and no guarantee
about what a public origin serves.

Introduced 2026-09-25 after the sealed Mimosa deep scan flagged two SSRF
findings on the direct convenience-opener calls here (scan
`scan-2026-09-25T10-46-34.545Z-e148e037d7f7`; see
docs/R7_SECURITY_SCAN_TRIAGE.md). Mirrors the opener pattern that
`arco_tiny_bounded.py` already used. Pinned to the validated address under #68.
"""
from __future__ import annotations

from .http_pinned import (  # noqa: F401  (re-exported public surface)
    MAX_REDIRECTS,
    PINNED_OPENER,
    NonPublicAddressRefused,
    PinnedHTTPConnection,
    PinnedHTTPSConnection,
    build_public_opener,
    open_public,
    reject_non_public_host,
    resolve_public_addresses,
)

# Historical name kept for callers and docs that refer to it.
PUBLIC_OPENER = PINNED_OPENER
