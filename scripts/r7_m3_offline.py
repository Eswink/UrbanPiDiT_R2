"""Process-wide outbound denial for the bounded M3 script and its owned workers.

R-028 requires that an experiment which does not intend to reach the network *cannot*, by
monkeypatching ``socket.socket.connect`` and ``socket.create_connection`` to raise. That is
what this module does, with one measured refinement: the denial applies to **IP** sockets
(``AF_INET``/``AF_INET6``) and to name resolution, not to every socket.

The distinction is not cosmetic. A ``DataLoader`` with ``num_workers>0`` and
``pin_memory=True`` sends tensors back through ``multiprocessing``'s resource sharer, which
uses an **``AF_UNIX``** socket at a local filesystem address. A blanket denial kills that
transfer and the run dies with ``Pin memory thread exited unexpectedly`` — measured on this
repository's Stage B point 1, where the whole first seed failed for that reason. A local
``AF_UNIX`` socket is not an outbound connection: it cannot leave the host, so permitting it
weakens nothing the rule is about.
"""
from __future__ import annotations

import socket

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CREATE = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo

#: The only family permitted: a local filesystem socket, which cannot leave the host.
#: Everything else -- including the ``connect(None, ...)`` form some callers use, where the
#: family cannot be read off the socket -- is denied, so an unknown target fails closed.
_LOCAL_FAMILY = socket.AF_UNIX
_MESSAGE = "M3 is offline: outbound connections forbidden"


def _is_local(args):
    """True only when the socket being connected is provably AF_UNIX."""
    owner = args[0] if args else None
    return getattr(owner, "family", None) == _LOCAL_FAMILY


def deny_network():
    """Refuse outbound connections except on a local ``AF_UNIX`` socket."""

    def refused_connect(*args, **kwargs):
        if _is_local(args):
            return _ORIGINAL_CONNECT(*args, **kwargs)
        raise RuntimeError(_MESSAGE)

    def refused_create_connection(*args, **kwargs):
        raise RuntimeError(_MESSAGE)

    def refused_getaddrinfo(*args, **kwargs):
        raise RuntimeError(_MESSAGE)

    socket.socket.connect = refused_connect
    socket.create_connection = refused_create_connection
    socket.getaddrinfo = refused_getaddrinfo


def allow_network():
    """Undo :func:`deny_network`, restoring the originals captured at import.

    The denial replaces attributes on the stdlib ``socket`` module, so it outlives the code
    that asked for it: a test that calls it leaves every later test in the same process
    unable to resolve or open an IP connection. Callers that want the denial scoped rather
    than permanent call this in a teardown.
    """
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.create_connection = _ORIGINAL_CREATE
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO
