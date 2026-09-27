"""Connection-address-constrained HTTP opener for the download layer (#68).

Why this exists next to :mod:`data.download.http_public`. The earlier guard
checked a URL's host with ``getaddrinfo`` and then let urllib dial the *same
hostname* again. That leaves a check/use window: the name can resolve to a
public address during validation and to a private one during the connection's
own resolution (classic DNS rebinding), and the validation is therefore a
**pre-resolution check**, not a constraint on the address actually dialled.

This module closes that window by construction rather than by narrowing it:

- the host is resolved **once**, every returned address is required to be
  global, and the socket is connected to those exact ``sockaddr`` values - the
  same objects that were checked. No second resolution happens between the
  check and the connect, because the connect no longer takes a hostname.
- every redirect hop is validated for scheme, host **and port** before it is
  followed.
- the proxy policy is explicit. ``build_opener`` would otherwise inherit
  ``*_proxy`` environment variables silently, so the opener installs its own
  :class:`urllib.request.ProxyHandler` with a declared mapping.
- HTTPS keeps the URL hostname as the TLS SNI name and keeps certificate and
  hostname verification on: the ``ssl`` context is the default verifying one,
  and no caller-facing switch relaxes it or treats a literal address as
  trusted.

What is *not* claimed: a pinned connection stops the connection from landing on
a non-public address, but a host that legitimately resolves to a public address
can still be an attacker-controlled origin, and a *redirect* to a hostname that
resolves publicly at validation time is followed normally. This module does not
provide allowlisting, content inspection, or a guarantee about what a public
origin serves. Those are separate controls.
"""
from __future__ import annotations

import errno
import http.client
import ipaddress
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request

_ALLOWED_SCHEMES = ("https", "http")
_DEFAULT_PORTS = {"https": 443, "http": 80}
MAX_REDIRECTS = 5


class NonPublicAddressRefused(ValueError):
    """The name resolved to at least one non-global address."""


def _require_scheme_and_host(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"仅允许 http/https 下载地址，得到 scheme={parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("下载地址缺少主机名")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("下载地址不得携带 userinfo 凭据")
    port = parsed.port or _DEFAULT_PORTS.get(parsed.scheme, 80)
    if not 1 <= port <= 65535:
        raise ValueError(f"端口超出范围：{port}")
    return parsed.scheme, host, port


def resolve_public_addresses(host, port):
    """Return the *validated* ``(family, socktype, proto, sockaddr)`` tuples.

    Every address the resolver returns must be global, so a name that answers
    with a public address and a private one is refused outright rather than
    connected to optimistically.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise NonPublicAddressRefused(f"无法解析下载主机 {host!r}：{exc}") from exc
    if not infos:
        raise NonPublicAddressRefused(f"下载主机 {host!r} 没有解析结果")
    validated = []
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise NonPublicAddressRefused(
                f"下载主机 {host!r} 解析到非公网地址 {address}，已拒绝（SSRF 防护）")
        validated.append(info)
    return validated


def reject_non_public_host(url: str) -> None:
    """Pre-resolution check. Kept for callers that want an early refusal.

    This function alone does **not** constrain the connected address: it
    resolves a hostname and discards the result. The connection classes in this
    module are what bind the check to the dialled address.
    """
    _scheme, host, port = _require_scheme_and_host(url)
    resolve_public_addresses(host, port)


def _connect_validated(infos, timeout, source_address):
    """Connect to one of the addresses that were already validated."""
    errors = []
    for info in infos:
        family, socktype, proto, _canonname, sockaddr = info
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            errors.append(exc)
            if sock is not None:
                sock.close()
    if errors:
        raise errors[0]
    raise OSError("no validated address could be connected")


class PinnedAddressMixin:
    """Replace hostname dialling with connecting to the validated addresses."""

    def _pinned_socket(self):
        infos = resolve_public_addresses(self.host, self.port)
        return _connect_validated(infos, self.timeout, self.source_address)


def _set_nodelay(sock):
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as exc:
        if exc.errno != errno.ENOPROTOOPT:
            raise


class PinnedHTTPConnection(PinnedAddressMixin, http.client.HTTPConnection):
    """HTTP connection whose socket target is the validated address."""

    def connect(self):
        self.sock = self._pinned_socket()
        _set_nodelay(self.sock)
        if self._tunnel_host:
            self._tunnel()


class PinnedHTTPSConnection(PinnedAddressMixin, http.client.HTTPSConnection):
    """HTTPS connection: pinned address, but SNI and cert checks stay on the host.

    ``server_hostname`` is the URL hostname, so the TLS handshake presents the
    same SNI as an ordinary connection and the certificate is verified against
    the name the caller asked for - connecting to a literal address never
    relaxes verification.
    """

    def connect(self):
        sock = self._pinned_socket()
        server_hostname = self._tunnel_host or self.host
        # ``self._context`` is always set by HTTPSConnection.__init__ (a
        # verifying default when the caller passes none), so the TLS identity is
        # carried by the context, never relaxed here.
        self.sock = self._context.wrap_socket(sock, server_hostname=server_hostname)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(PinnedHTTPConnection, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(PinnedHTTPSConnection, req, context=self._context)


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate scheme, host, port **and resolved addresses** of every hop.

    Failing here is an early refusal for a hop that could not be dialled
    anyway; the connection classes remain the place that binds the check to the
    address actually used, so this is defence in depth rather than the only
    guard.
    """

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _scheme, host, port = _require_scheme_and_host(newurl)
        resolve_public_addresses(host, port)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_public_opener(proxies=None, context=None):
    """An opener with pinned connections and a declared proxy policy.

    ``proxies`` is the explicit mapping. The default is an empty mapping, which
    means "no proxy": environment variables are deliberately *not* inherited,
    because a silently inherited proxy would make the validated address a
    request to the proxy rather than the destination. An explicit mapping is
    honoured, so a site that genuinely needs an egress proxy can declare one
    instead of relying on ambient environment state.
    """
    handlers = [
        _PinnedHTTPHandler(),
        _PinnedHTTPSHandler(context=context),
        _ValidatedRedirectHandler(),
    ]
    if proxies:
        handlers.append(urllib.request.ProxyHandler(dict(proxies)))
    else:
        # An explicit "no proxy" decision. Passing the *class* rather than an
        # empty instance is what makes build_opener install a ProxyHandler that
        # overrides the environment-derived default instead of skipping it.
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


PUBLIC_OPENER = build_public_opener()


def open_public(request, *, timeout=60, proxies=None, context=None, opener=None):
    """Open ``request`` on a pinned, redirect-validating, proxy-declared opener.

    ``timeout`` is passed to the socket layer, so a hung peer cannot hold the
    caller indefinitely. Callers keep their own byte caps.
    """
    _require_scheme_and_host(request.full_url)
    active = opener
    if active is None:
        active = PUBLIC_OPENER if proxies is None and context is None \
            else build_public_opener(proxies=proxies, context=context)
    try:
        return active.open(request, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"下载请求被拒绝或失败：{exc}") from exc
