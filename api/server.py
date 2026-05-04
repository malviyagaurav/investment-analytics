"""Server launcher with dynamic port selection.

Picks the preferred port (env PORT, default 8010); if it's already
in use, falls back to the next free port within a bounded range.
Used by run.command / run.bat and by `python -m api.main`.
"""
from __future__ import annotations

import errno
import os
import socket
from typing import Optional

DEFAULT_PORT = 8010
DEFAULT_HOST = "127.0.0.1"
MAX_PORT_TRIES = 20  # scan PORT..PORT+19 before giving up


def is_port_free(host: str, port: int) -> bool:
    """Return True if a TCP listener could bind `host:port` right now.

    Uses a transient bind() with SO_REUSEADDR off — i.e., we want the
    real "is anything listening here?" answer, not the relaxed
    REUSEADDR semantics that would let two listeners coexist.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                return False
            raise
        return True


def find_available_port(
    preferred: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    max_tries: int = MAX_PORT_TRIES,
) -> int:
    """Try `preferred` first, then preferred+1, +2, ... up to max_tries.

    Returns the first port that's free. Raises RuntimeError if every
    candidate in the scan range is occupied — leaves choice of remediation
    (kill the offending process, pick a higher base port) to the operator.
    """
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")
    for offset in range(max_tries):
        port = preferred + offset
        if is_port_free(host, port):
            return port
    raise RuntimeError(
        f"No free port found in range {preferred}..{preferred + max_tries - 1} "
        f"on {host}. Set PORT to a different base or free one of these ports."
    )


def resolve_port(env: Optional[dict] = None) -> int:
    """Resolve the actual port to bind based on env PORT (default 8010),
    falling back if the requested port is taken."""
    env = env if env is not None else os.environ
    preferred_raw = env.get("PORT") or str(DEFAULT_PORT)
    try:
        preferred = int(preferred_raw)
    except ValueError:
        raise ValueError(
            f"Invalid PORT={preferred_raw!r}; expected an integer."
        )
    if preferred < 1 or preferred > 65535:
        raise ValueError(f"PORT={preferred} out of range 1..65535")
    return find_available_port(preferred=preferred)


def run() -> None:
    """Entry point: resolve the port, log the resolution, hand off to uvicorn."""
    import uvicorn  # imported here so test imports of this module are dep-free

    preferred = int(os.environ.get("PORT") or DEFAULT_PORT)
    port = resolve_port()
    if port != preferred:
        print(
            f"Port {preferred} is busy. Falling back to {port}.\n"
            f"Open: http://{DEFAULT_HOST}:{port}",
            flush=True,
        )
    else:
        print(
            f"Investment Analytics Engine starting on http://{DEFAULT_HOST}:{port}",
            flush=True,
        )

    # Try to open the browser non-blockingly. Failing to open is fine —
    # the server still starts and prints the URL.
    if os.environ.get("OPEN_BROWSER", "1") == "1":
        try:
            import webbrowser
            webbrowser.open(f"http://{DEFAULT_HOST}:{port}")
        except Exception:
            pass

    uvicorn.run(
        "api.main:app",
        host=DEFAULT_HOST,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()
