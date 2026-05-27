"""Entry point: ``python -m backend.research_artifacts.llm_assist``."""
from __future__ import annotations

import sys

from backend.research_artifacts.llm_assist.cli import main


if __name__ == "__main__":
    sys.exit(main())
