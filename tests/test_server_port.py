"""Regression tests for api.server port-resolution helpers.

The launcher must:
- bind to PORT (or 8010 by default) when free
- fall back to the next free port when the preferred is taken
- raise a clear error if the entire scan range is busy
- reject malformed PORT env values
"""
from __future__ import annotations

import socket
import unittest

from api import server


def _occupy(host: str, port: int) -> socket.socket:
    """Open a listening socket on host:port. Caller closes it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    return s


def _free_port() -> int:
    """Ask the OS for a free port (port 0) and return its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class IsPortFreeTests(unittest.TestCase):

    def test_free_port_reports_true(self) -> None:
        port = _free_port()
        self.assertTrue(server.is_port_free("127.0.0.1", port))

    def test_occupied_port_reports_false(self) -> None:
        port = _free_port()
        listener = _occupy("127.0.0.1", port)
        try:
            self.assertFalse(server.is_port_free("127.0.0.1", port))
        finally:
            listener.close()


class FindAvailablePortTests(unittest.TestCase):

    def test_returns_preferred_when_free(self) -> None:
        port = _free_port()
        chosen = server.find_available_port(preferred=port, max_tries=5)
        self.assertEqual(chosen, port)

    def test_falls_back_when_preferred_busy(self) -> None:
        # Find two adjacent free ports, then occupy the first one.
        # find_available_port should skip it and return the second.
        port_a = _free_port()
        port_b = port_a + 1
        # If port_a + 1 happens to be occupied by something else on this
        # machine, retry once with a different starting point.
        if not server.is_port_free("127.0.0.1", port_b):
            port_a = _free_port()
            port_b = port_a + 1
        listener = _occupy("127.0.0.1", port_a)
        try:
            self.assertFalse(server.is_port_free("127.0.0.1", port_a))
            chosen = server.find_available_port(preferred=port_a, max_tries=10)
            self.assertNotEqual(chosen, port_a,
                                "must NOT pick the occupied preferred port")
            self.assertGreaterEqual(chosen, port_a + 1)
        finally:
            listener.close()

    def test_raises_when_every_port_in_range_busy(self) -> None:
        port = _free_port()
        listener = _occupy("127.0.0.1", port)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                # max_tries=1 means only the preferred port is tried.
                server.find_available_port(preferred=port, max_tries=1)
            self.assertIn(str(port), str(ctx.exception))
        finally:
            listener.close()

    def test_max_tries_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            server.find_available_port(preferred=12345, max_tries=0)


class ResolvePortTests(unittest.TestCase):

    def test_default_when_env_empty(self) -> None:
        # Resolution against an empty env must default to 8010 (or the
        # next free port if 8010 is taken — that's the whole point of
        # the launcher and is acceptable as long as it's still free).
        port = server.resolve_port(env={})
        self.assertGreaterEqual(port, server.DEFAULT_PORT)
        self.assertLess(port, server.DEFAULT_PORT + server.MAX_PORT_TRIES)

    def test_honours_env_port(self) -> None:
        target = _free_port()
        port = server.resolve_port(env={"PORT": str(target)})
        self.assertEqual(port, target)

    def test_falls_back_when_env_port_busy(self) -> None:
        target = _free_port()
        listener = _occupy("127.0.0.1", target)
        try:
            port = server.resolve_port(env={"PORT": str(target)})
            self.assertNotEqual(port, target)
            self.assertGreater(port, target)
        finally:
            listener.close()

    def test_rejects_non_integer_env(self) -> None:
        with self.assertRaises(ValueError):
            server.resolve_port(env={"PORT": "not-a-number"})

    def test_rejects_out_of_range_env(self) -> None:
        with self.assertRaises(ValueError):
            server.resolve_port(env={"PORT": "0"})
        with self.assertRaises(ValueError):
            server.resolve_port(env={"PORT": "70000"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
