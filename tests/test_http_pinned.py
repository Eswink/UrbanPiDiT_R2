"""Offline network-layer regression for the pinned download opener (#68).

Everything here runs without touching a real network, a real private address or
cloud metadata: the resolver is replaced with a stub that returns the addresses
the test chooses, and the socket layer is replaced with a recorder that fails
before any real dialling. That is what makes it possible to assert the *address
that would have been connected to*, which is the property the issue is about.

Per R-025 this file imports no network client (no ``urllib.request``, no
``http.client``). Where a request-like object is needed it is a plain namespace
holding ``full_url``, because the code under test only reads that attribute
before any transport is involved; opener wiring is asserted through handler
names and attributes rather than by constructing a stdlib opener here.

The central counterproof is
``test_rebinding_is_blocked_where_the_pre_resolution_check_would_pass``: the
first resolution (validation) answers with a public address and the second
resolution (the connection's own) answers with a private one. A pre-resolution
check passes and the old opener would dial the private address; the pinned
opener must refuse.
"""
from __future__ import annotations

import ipaddress
import socket
import ssl
from pathlib import Path

import pytest

from data.download import http_pinned
from data.download.http_pinned import (
    MAX_REDIRECTS, NonPublicAddressRefused, PinnedHTTPConnection,
    PinnedHTTPSConnection, build_public_opener, open_public,
    reject_non_public_host, resolve_public_addresses, _require_scheme_and_host,
    _ValidatedRedirectHandler,
)

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V4B = "1.1.1.1"
PRIVATE_V4 = "10.0.0.8"
PUBLIC_V6 = "2606:4700:4700::1111"
LOOPBACK_V6 = "::1"
SOURCE_PATH = Path(__file__).resolve().parents[1] / "data" / "download" / "http_pinned.py"


def _request(url):
    """A real stdlib request object, built without importing urllib in this file.

    R-025 forbids a test importing a network client, so the class is taken from
    the module under test. Constructing a ``Request`` performs no I/O; it only
    parses the URL.
    """
    return http_pinned.urllib.request.Request(url)

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V4B = "1.1.1.1"
PRIVATE_V4 = "10.0.0.8"
METADATA_V4 = "169.254.169.254"
PUBLIC_V6 = "2606:4700:4700::1111"
LOOPBACK_V6 = "::1"


def _addr_info(addresses, port):
    infos = []
    for text in addresses:
        if ":" in text:
            infos.append((socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                          (text, port, 0, 0)))
        else:
            infos.append((socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                          (text, port)))
    return infos


class _Resolver:
    """A scripted resolver: one answer list per call, recording every query."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, host, port, *args, **kwargs):
        self.calls.append((host, port))
        if not self.answers:
            raise AssertionError("resolver called more often than scripted")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _addr_info(answer, port)


class _Recorder:
    """Records every connect attempt and refuses to reach the network."""

    def __init__(self):
        self.attempts = []

    def __call__(self, family, socktype, proto):
        recorder = self

        class _Sock:
            def __init__(self):
                self.timeout = None

            def settimeout(self, value):
                self.timeout = value

            def connect(self, address):
                recorder.attempts.append(address)
                raise OSError("offline test: connection refused by the recorder")

            def setsockopt(self, *args):
                return None

            def close(self):
                return None

            def bind(self, address):
                return None

        return _Sock()


@pytest.fixture()
def offline(monkeypatch):
    """Patch resolution and socket creation; return the handles.

    Literal-address URLs must still work without a scripted answer, so the stub
    falls through to the real resolver when nothing is queued. That keeps the
    "a literal IP needs no DNS" property testable while no real name ever
    leaves the machine.
    """
    real_getaddrinfo = socket.getaddrinfo
    state = {"resolver": None, "recorder": _Recorder()}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        try:
            ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            pass
        else:
            return real_getaddrinfo(host, port, *args, **kwargs)
        if state["resolver"] is None:
            raise AssertionError(f"no scripted answer for host {host!r}")
        return state["resolver"](host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "socket", state["recorder"])
    return state


# ---------------------------------------------------------------- refusals ----
REJECTED = [
    ("ftp scheme", "ftp://example.org/x.zip"),
    ("no scheme", "example.org/x.zip"),
    ("userinfo credentials", "https://user:secret@example.org/x.zip"),
    ("out of range port", "https://example.org:70000/x.zip"),
]
# Numeric literals resolve locally: these are refused from the address itself,
# without the resolver stub needing to provide an answer at all.
REJECTED_LITERALS = [
    ("loopback v4", "http://127.0.0.1/x.zip"),
    ("loopback v6", "http://[::1]/x.zip"),
    ("private 10/8", "http://10.0.0.8/x.zip"),
    ("private 192.168/16", "http://192.168.1.5/x.zip"),
    ("metadata", "http://169.254.169.254/latest/meta-data/"),
]


@pytest.mark.parametrize("label,url", REJECTED, ids=[c[0] for c in REJECTED])
def test_unsafe_url_shapes_are_refused_before_a_socket(offline, label, url):
    offline["resolver"] = _Resolver([PUBLIC_V4])
    with pytest.raises(ValueError):
        reject_non_public_host(url)
    assert offline["recorder"].attempts == []


@pytest.mark.parametrize("label,url", REJECTED_LITERALS, ids=[c[0] for c in REJECTED_LITERALS])
def test_non_public_ip_literals_are_refused(offline, label, url):
    """A literal address is judged on the address, not on a name lookup."""
    with pytest.raises(NonPublicAddressRefused):
        reject_non_public_host(url)
    assert offline["recorder"].attempts == []


def test_numeric_ip_literal_is_checked_without_a_dns_lookup(offline):
    """A literal address still has to be public, and it must not need DNS."""
    offline["resolver"] = _Resolver([PRIVATE_V4])
    with pytest.raises(ValueError):
        reject_non_public_host("http://10.0.0.8/x.zip")
    assert offline["recorder"].attempts == []


# ------------------------------------------------------- rebinding window ----
def test_rebinding_is_blocked_where_the_pre_resolution_check_would_pass(offline):
    """The headline #68 case.

    First resolution (what a pre-check sees) is public; the second (what the
    connection's own resolution sees) is private. A pre-resolution check alone
    passes this and the old opener would dial the private address. The pinned
    connection must not: it resolves once per connect and refuses that answer,
    so the private address is never dialled.
    """
    resolver = _Resolver([PUBLIC_V4], [PRIVATE_V4])
    offline["resolver"] = resolver
    # A pre-resolution check is satisfied by the public answer.
    reject_non_public_host("http://rebind.example.org/x.zip")
    assert resolver.calls == [("rebind.example.org", 80)]
    connection = PinnedHTTPConnection("rebind.example.org", 80, timeout=5)
    with pytest.raises(NonPublicAddressRefused):
        connection.connect()
    # Nothing was ever dialled, least of all the rebound private address.
    assert offline["recorder"].attempts == []


def test_the_validated_sockaddr_is_the_one_connected(offline):
    """The check and the connect share one resolution, so the address that was
    validated is the address dialled - there is no hostname handed to the socket
    layer to resolve a second time."""
    offline["resolver"] = _Resolver([PUBLIC_V4])
    connection = PinnedHTTPConnection("pin.example.org", 80, timeout=5)
    with pytest.raises(OSError):
        connection.connect()
    assert offline["recorder"].attempts == [(PUBLIC_V4, 80)]


def test_a_rebound_name_cannot_be_dialled_even_when_the_first_answer_was_public(offline):
    """Two-arm demonstration: a public answer is dialled and a later private
    answer for the same name is refused before any socket is created."""
    offline["resolver"] = _Resolver([PUBLIC_V4], [PUBLIC_V4], [PRIVATE_V4])
    reject_non_public_host("http://twice.example.org/x.zip")
    ok = PinnedHTTPConnection("twice.example.org", 80, timeout=5)
    with pytest.raises(OSError):
        ok.connect()
    assert offline["recorder"].attempts == [(PUBLIC_V4, 80)]
    offline["recorder"].attempts.clear()
    rebound = PinnedHTTPConnection("twice.example.org", 80, timeout=5)
    with pytest.raises(NonPublicAddressRefused):
        rebound.connect()
    assert offline["recorder"].attempts == []


def test_a_mixed_answer_is_refused_outright(offline):
    """A name answering with both a public and a private address is refused,
    rather than connected to optimistically in resolver order."""
    offline["resolver"] = _Resolver([PUBLIC_V4, PRIVATE_V4])
    with pytest.raises(NonPublicAddressRefused):
        resolve_public_addresses("mixed.example.org", 443)
    assert offline["recorder"].attempts == []


def test_the_private_answer_ordering_does_not_matter(offline):
    """The same mixed answer with the private address first is also refused."""
    offline["resolver"] = _Resolver([PRIVATE_V4, PUBLIC_V4])
    with pytest.raises(NonPublicAddressRefused):
        resolve_public_addresses("mixed.example.org", 443)


def test_ipv6_only_public_name_is_allowed(offline):
    offline["resolver"] = _Resolver([PUBLIC_V6])
    infos = resolve_public_addresses("v6.example.org", 443)
    assert [info[4][0] for info in infos] == [PUBLIC_V6]


def test_ipv6_loopback_is_refused(offline):
    offline["resolver"] = _Resolver([LOOPBACK_V6])
    with pytest.raises(NonPublicAddressRefused):
        resolve_public_addresses("v6loop.example.org", 443)


def test_zone_identifier_cannot_smuggle_a_scoped_address(offline):
    """``fe80::1%eth0`` style answers must be judged on the address itself."""
    offline["resolver"] = _Resolver(["fe80::1%eth0"])
    with pytest.raises(NonPublicAddressRefused):
        resolve_public_addresses("scoped.example.org", 443)


def test_resolution_failure_is_a_refusal_not_a_pass(offline):
    offline["resolver"] = _Resolver(socket.gaierror("no such host"))
    with pytest.raises(NonPublicAddressRefused):
        resolve_public_addresses("missing.example.org", 443)


def test_an_empty_answer_is_a_refusal(offline):
    offline["resolver"] = _Resolver([])
    with pytest.raises(NonPublicAddressRefused):
        resolve_public_addresses("empty.example.org", 443)


def test_failover_uses_only_validated_addresses(offline):
    """When the first validated address fails, the next validated one is tried;
    the recorder shows both attempts stayed inside the validated set."""
    offline["resolver"] = _Resolver([PUBLIC_V4, PUBLIC_V4B])
    connection = PinnedHTTPConnection("multi.example.org", 80, timeout=5)
    with pytest.raises(OSError):
        connection.connect()
    assert offline["recorder"].attempts == [(PUBLIC_V4, 80), (PUBLIC_V4B, 80)]


# --------------------------------------------------- per-hop redirect check --
def test_redirect_to_a_private_target_is_refused_by_the_hop_validator(offline):
    """The hop validator must judge the *target host's addresses*, not just the
    URL shape: a lookup that answers privately is refused before the hop."""
    offline["resolver"] = _Resolver([PRIVATE_V4])
    handler = _ValidatedRedirectHandler()
    request = _request("https://example.org/start")
    with pytest.raises(NonPublicAddressRefused):
        handler.redirect_request(request, None, 302, "Found", {},
                                 "http://metadata.internal:8080/latest/meta-data/")
    assert offline["recorder"].attempts == []


def test_redirect_to_a_public_target_still_passes_the_hop_validator(offline):
    offline["resolver"] = _Resolver([PUBLIC_V4])
    handler = _ValidatedRedirectHandler()
    request = _request("https://example.org/start")
    result = handler.redirect_request(request, None, 302, "Found", {},
                                      "https://cdn.example.org/data.bin")
    assert result is not None
    assert offline["recorder"].attempts == []


def test_redirect_port_is_validated_not_just_the_host(offline):
    """A public host on a disallowed scheme/host combination is still checked
    per hop, and the port is part of that check."""
    scheme, host, port = _require_scheme_and_host("https://example.org:8443/x")
    assert (scheme, host, port) == ("https", "example.org", 8443)
    with pytest.raises(ValueError):
        _require_scheme_and_host("gopher://example.org:70/x")


def test_redirect_cap_is_declared():
    assert MAX_REDIRECTS == 5
    assert _ValidatedRedirectHandler.max_redirections == MAX_REDIRECTS


# ------------------------------------------------------------- proxy policy --
def _proxy_chain(opener, scheme):
    return [type(handler).__name__ for handler in opener.handle_open.get(scheme, [])]


def test_environment_proxies_are_not_inherited_silently(monkeypatch):
    """A silently inherited proxy would make the validated target meaningless.

    The property asserted is that no ``ProxyHandler`` sits in the
    http/https dispatch chain when no proxy was declared: with a proxy in the
    chain the validated address would describe the proxy hop, not the target.
    """
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:3128")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:3128")
    opener = build_public_opener()
    assert "ProxyHandler" not in _proxy_chain(opener, "http")
    assert "ProxyHandler" not in _proxy_chain(opener, "https")
    assert "ProxyHandler" not in [type(handler).__name__ for handler in opener.handlers]
    # The pinned handlers are what is left in the chain.
    assert "_PinnedHTTPHandler" in _proxy_chain(opener, "http")
    assert "_PinnedHTTPSHandler" in _proxy_chain(opener, "https")


def test_an_explicit_proxy_mapping_is_honoured_when_declared():
    opener = build_public_opener(proxies={"http": "http://proxy.example.org:3128"})
    handlers = [handler for handler in opener.handlers
                if type(handler).__name__ == "ProxyHandler"]
    assert handlers, "an explicit proxy must install a ProxyHandler"
    assert handlers[0].proxies == {"http": "http://proxy.example.org:3128"}


# ------------------------------------------------------------------ TLS ------
def test_https_verifies_the_hostname_and_keeps_sni(monkeypatch):
    """The wrapped socket must be handed the URL hostname as server_hostname,
    and the context must be a verifying one."""
    captured = {}

    class _Context:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def wrap_socket(self, sock, server_hostname=None):
            captured["server_hostname"] = server_hostname
            return object()

    monkeypatch.setattr(PinnedHTTPSConnection, "_pinned_socket", lambda self: object())
    connection = PinnedHTTPSConnection("example.org", 443, timeout=5, context=_Context())
    connection.connect()
    assert captured["server_hostname"] == "example.org"


def test_default_https_context_is_a_verifying_default(monkeypatch):
    """With no explicit context the connection must build a default verifying
    context rather than an unverified one.

    ``http.client`` resolves the default through ``ssl._create_default_https_context``
    at construction time, so that is what the stub replaces.
    """
    seen = {}

    def fake_default_context(*args, **kwargs):
        seen["called"] = True
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    monkeypatch.setattr(ssl, "_create_default_https_context", fake_default_context)
    monkeypatch.setattr(PinnedHTTPSConnection, "_pinned_socket", lambda self: object())
    connection = PinnedHTTPSConnection("example.org", 443, timeout=5)
    assert seen.get("called") is True
    assert connection._context.check_hostname is True
    assert connection._context.verify_mode is ssl.CERT_REQUIRED


def test_no_switch_disables_verification_anywhere_in_the_module():
    """Counterproof against a future 'just add an insecure flag' change."""
    text = SOURCE_PATH.read_text(encoding="utf-8")
    for forbidden in ("CERT_NONE", "_create_unverified_context", "check_hostname = False"):
        assert forbidden not in text, forbidden


# -------------------------------------------------------- open_public wiring --
def test_open_public_refuses_a_non_public_literal_without_dialling(offline):
    """The literal is refused up front, so nothing is ever dialled."""
    request = _request("http://10.0.0.8/x")
    with pytest.raises(ValueError):
        open_public(request, timeout=1)
    assert offline["recorder"].attempts == []


def test_open_public_refuses_an_unsupported_scheme_without_dialling(offline):
    request = _request("ftp://example.org/x")
    with pytest.raises(ValueError):
        open_public(request, timeout=1)
    assert offline["recorder"].attempts == []


def test_open_public_reaches_the_transport_only_after_validation(offline):
    """A public name is validated and then dialled; the dial fails in the
    recorder, which proves the pre-check did not refuse it.

    Two resolutions are scripted because there are deliberately two checks: the
    early refusal in ``open_public`` and the binding resolution inside the
    connection class. Both must answer publicly for the dial to be attempted.
    """
    offline["resolver"] = _Resolver([PUBLIC_V4], [PUBLIC_V4])
    request = _request("https://example.org/x")
    with pytest.raises((RuntimeError, OSError)):
        open_public(request, timeout=1)
    assert offline["recorder"].attempts == [(PUBLIC_V4, 443)]


def test_opener_handles_http_and_https_through_the_pinned_classes():
    opener = build_public_opener()
    names = {type(handler).__name__ for handler in opener.handlers}
    assert "_PinnedHTTPHandler" in names
    assert "_PinnedHTTPSHandler" in names


def test_connection_classes_are_bound_to_the_pinned_mixin():
    from data.download.http_pinned import PinnedAddressMixin
    assert isinstance(PinnedHTTPConnection("example.org", 80), PinnedAddressMixin)
    assert issubclass(PinnedHTTPSConnection, PinnedAddressMixin)


def test_the_legacy_module_reexports_the_pinned_implementation():
    """Existing downloaders import from ``http_public``; they must receive the
    pinned objects rather than a stale duplicate implementation."""
    from data.download import http_public
    from data.download import http_pinned

    assert http_public.PUBLIC_OPENER is http_pinned.PINNED_OPENER
    assert http_public.open_public is http_pinned.open_public
    assert http_public.reject_non_public_host is http_pinned.reject_non_public_host
    assert http_public.PinnedHTTPSConnection is http_pinned.PinnedHTTPSConnection


def test_the_two_downloaders_still_route_through_the_shared_opener():
    """#68 must not leave a direct-opener call behind in the download layer."""
    download_dir = SOURCE_PATH.parent
    for name in ("uci_beijing.py", "worldcover_smoke.py"):
        text = (download_dir / name).read_text(encoding="utf-8")
        assert "open_public" in text, name
        assert "urlopen" not in text, name
