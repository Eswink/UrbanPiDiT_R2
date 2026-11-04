"""Process-wide outbound denial for the bounded M3 script and its owned workers."""
from __future__ import annotations

import socket

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CREATE = socket.create_connection


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("M3 is offline: outbound connections forbidden")
    socket.socket.connect = refused
    socket.create_connection = refused


def allow_network():
    """Undo :func:`deny_network`, restoring the originals captured at import.

    The denial replaces two attributes on the stdlib ``socket`` module, so it outlives the
    code that asked for it: a test that calls it leaves every later test in the same
    process unable to open a socket, including the IPC a forked DataLoader worker needs.
    Callers that want the denial scoped rather than permanent call this in a teardown.
    """
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.create_connection = _ORIGINAL_CREATE
