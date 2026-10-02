"""Process-wide outbound denial for the bounded M3 script and its owned workers."""
from __future__ import annotations

import socket


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("M3 is offline: outbound connections forbidden")
    socket.socket.connect = refused
    socket.create_connection = refused
