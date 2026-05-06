"""`python -m backend.jobs <subcommand>` dispatcher.

Forwards to backend.jobs.watchlist for now; new jobs can be wired
in here as additional subcommand groups.
"""
from __future__ import annotations

import sys

from backend.jobs.watchlist import _main


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
