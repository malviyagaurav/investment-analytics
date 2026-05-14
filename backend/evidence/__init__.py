"""Evidence layer — by-reference artifacts referenced from the
audit chain.

Per the architecture decision (2026-05-15), heavy payloads (ranking
snapshots, portfolio-health snapshots) live in this package's store
under ``data/evidence/<kind>/<run_id>.json``. The audit record carries
``evidence_ref: {path, sha256, size_bytes}`` pointing at them.

Future modules in this package:
  - ``store.py``     — write_evidence + read_evidence (step 4).
  - ``replay.py``    — semantic replay tool (step 7).
  - ``drift.py``     — snapshot-to-snapshot drift analysis (future).
  - ``experiment.py``— experiment runner (future).
"""
