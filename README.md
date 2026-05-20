# Investment Analytics Engine

Analytics-only MVP foundation for deterministic portfolio diagnostics, benchmark comparisons, and user-driven scenarios.

## Positioning

This system is designed to enforce regulatory-aware constraints, subject to legal validation.

It does not produce personalized investment actions, product selection, rankings, targets, or implied preferences.

## Current Scope

- Deterministic JSON insight templates
- Compile-time schema enforcement
- Advisory-language linter with fail-closed behavior
- Jurisdiction whitelist gate
- Data license lineage propagation
- Standard and user-authored scenario guardrails
- Append-only hash-chained audit log
- Minimal portfolio diagnostics endpoint
- CSV-backed Mutual Fund Analyzer

## Mutual Fund Analyzer

The first real analyzer emits only template-driven analytics:

- Separate fund and benchmark series with common-date alignment
- Rolling return distribution and excess-return statistics
- Maximum drawdown comparison with start, trough, recovery, and duration fields
- Hypothetical TER drag calculation under explicit assumptions
- Dirty-data handling for duplicates, unsorted rows, non-positive values, missing dates, and outlier flags

Input contract:

```json
{
  "fund": [{ "date": "YYYY-MM-DD", "nav": 100.0 }],
  "benchmark": [{ "date": "YYYY-MM-DD", "value": 100.0 }]
}
```

Demo endpoint:

```text
GET /analytics/mutual-fund/demo
```

The included CSV is synthetic sample data for local testing:

```text
data/sample/mutual_fund_nav.csv
```

## Supported platforms

- macOS — officially supported.
- Linux — officially supported.
- Windows — **not officially supported for daily operational use.** CI does run the full test suite on `windows-latest` to catch portability regressions in the locking adapter and byte-stability surface, but end-to-end behavior (server, scheduler, registry refresh under Windows file semantics) is not validated. `run.bat` is retained as a deprecated launcher. See `BOOTSTRAP.md` for the calibrated platform stance.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8010 --reload
```

Then open:

```text
http://127.0.0.1:8010
```

The equivalent invocation used by `run.command` / `run.bat` is `python -m api.server`, which additionally handles port fallback (env `PORT`, default 8010) and opens a browser (env `OPEN_BROWSER=1`).

## Test

Every test under `tests/` is a `unittest.TestCase` subclass; both runners are equivalent:

```bash
python3 -m unittest discover -s tests -v          # stdlib only
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/ -v                       # equivalent; matches CI
```

CI uses pytest for richer output.

## Setting up a fresh clone

See [`BOOTSTRAP.md`](BOOTSTRAP.md) for the canonical clean-clone sequence, the Python version requirement (3.9), the operational state taxonomy (which directories under `data/` are tracked vs lazily created vs gitignored), and the registry-refresh procedure.
