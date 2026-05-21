# Bootstrap

This document describes how to bring a fresh clone of the
repository to a working state. It complements `README.md` — the
README covers product positioning and the canonical run/test
commands; this file covers what a new machine actually needs.

It is intentionally narrow: it documents the operational
contract a clean clone is expected to reproduce. It does NOT
describe how to reproduce the maintainer's audit history,
scheduler cadence state, registry snapshot, or local cache —
those are per-machine artifacts by design (see "Operational
state taxonomy" below).


## Canonical clean-clone sequence

```bash
git clone <repo>
cd investment-analytics
python3.9 -m venv venv
source venv/bin/activate              # macOS / Linux
# venv\Scripts\activate                Windows (see "Platform support")
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only if you intend to run tests

python -m unittest discover -s tests -v   # full suite, no pytest needed
# OR
python -m pytest tests/ -v                # equivalent; needs requirements-dev.txt

python -m api.server                  # starts the HTTP API on 127.0.0.1:8010
```

That sequence reaches Tier 1 ("runnable") and Tier 2
("development-complete") of the cloneability model used in the
2026-05-20 reproducibility audit. It does not reach Tier 3 or
Tier 4 — those tiers are architecturally unchosen non-goals,
not gaps (see "Operational state taxonomy").


## Python version

Python 3.9 is the supported interpreter. The reasons are:

  * the cross-platform CI matrix in
    `.github/workflows/cross-platform-parity.yml` pins
    `python-version: "3.9"` on all three runners (ubuntu, macos,
    windows);
  * the canonical-JSON byte-stability test at
    `tests/test_cross_platform.py` pins a SHA-256 reference hash
    against output produced by CPython 3.9.6 (the comment in
    that test says "PINNED on macOS (darwin, CPython 3.9.6)
    2026-05-16. Do NOT update this value casually.") — running
    on a different minor Python version risks a silent failure
    if json/encoding behavior drifts.

The repo root carries a `.python-version` file containing `3.9`
so pyenv-compatible shells will select the right interpreter
automatically when present.


## Dependencies

Two requirements files:

  * `requirements.txt` — runtime dependencies. Used by
    `run.command`, `run.bat`, and CI. Pinned exactly except for
    `httpx`, which uses a `>=` constraint (legacy; left alone in
    this pass).
  * `requirements-dev.txt` — development dependencies (currently
    only `pytest`). Not required to start the server or to run
    the test suite via `python -m unittest discover`.

Both files are intentionally narrow. Everything else is either
in the Python stdlib or pulled in transitively by FastAPI.


## Running the server

```bash
python -m api.server
```

`api.server` resolves a free TCP port (env `PORT`, default 8010;
scans `PORT..PORT+19` if the preferred port is busy), prints the
chosen URL, optionally opens a browser (env `OPEN_BROWSER`,
default `1`), and hands off to uvicorn. No other environment
variables are read by the application.

The README also shows an equivalent uvicorn invocation:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8010 --reload
```

That form is fine for development but bypasses `api.server`'s
port-fallback and browser-open logic.


## Running tests

Every test under `tests/` is a `unittest.TestCase` subclass, so
both runners work and produce equivalent coverage:

```bash
python -m unittest discover -s tests -v    # stdlib only
python -m pytest tests/ -v                 # needs requirements-dev.txt
```

CI uses pytest for richer output. The pinned byte-stability
test in `tests/test_cross_platform.py` is the most likely source
of cross-platform divergence; if it fails after a clean clone,
verify Python version (`python --version`) before anything else.


## Operational state taxonomy

`data/` mixes three categories of state. The boundary is not
encoded by directory naming alone; this section documents it
explicitly:

| Category | Contents | Tracked? |
|----------|----------|----------|
| Canonical seeds — repository-authoritative reference data | `data/reference/`, `data/watchlist/categories.json` | tracked |
| Reproducible sample fixtures — synthetic data for local testing and demos | `data/sample/*.csv` | tracked |
| Mutable operational runtime state — machine-local lineage, opportunistic caches, append-only chains, by-reference evidence files | `data/audit/`, `data/cache/`, `data/evidence/`, `data/registry/`, `data/snapshots/`, `data/scheduler/` | gitignored, lazily created |

The runtime-state directories are auto-created on demand by the
code that writes them (`mkdir(parents=True, exist_ok=True)` in
`backend/investment_analytics/audit.py`,
`backend/data_discovery/cache.py`, and
`backend/scheduler/runner.py`). A fresh clone does NOT need
them pre-populated for the server to start or for the test
suite to pass — every test that touches these surfaces creates
its own tmpdir fixture.

What the fresh clone DOES need, to escape the "everything works
but produces empty results" state for discovery / watchlist /
calibration endpoints, is a populated registry. See next
section.


## Populating the registry (optional but usually wanted)

`data/registry/schemes.json` is the AMFI scheme metadata
snapshot (~5 MB on the maintainer machine). It is gitignored
because AMFI publishes a new copy daily and committing it would
introduce stale-state semantics. A fresh clone has no registry
file, which means `load_registry()` returns `[]` and the
search / watchlist / calibration endpoints serve empty results
without raising.

To populate it, run the server and hit the refresh endpoint:

```bash
python -m api.server &
curl -X POST http://127.0.0.1:8010/discover/refresh-registry
```

That fetches `https://portal.amfiindia.com/spages/NAVAll.txt`,
parses it, and writes `data/registry/schemes.json`. There is
currently no standalone CLI for this; the endpoint is the only
documented entry point.

Network access to `portal.amfiindia.com` is required. Offline
clones cannot populate the registry; they remain in the
empty-result state until they regain connectivity.


## Scheduler installation

The daily evidence cadence is installed via:

```bash
python scripts/install_scheduler.py
```

The script emits the OS-appropriate artifact (launchd plist on
macOS, crontab line on Linux, Task Scheduler XML on Windows)
but does NOT activate the scheduler — that step requires a
deliberate operator action (`launchctl load`, `crontab -e`,
`schtasks /Create`). The activation commands are printed at the
end of `install_scheduler.py`'s output.

Known issue (tracked by the 2026-05-20 audit as Phase C work,
not yet authorized): the committed
`scripts/com.investment-analytics.scheduler.plist` contains
literal `/Users/snapdevio/...` paths on four lines. An operator
who installs that plist verbatim on a different machine will
get a broken job. The plist's own header comment instructs the
operator to "Edit the WorkingDirectory + Program Arguments
below to match your install path before installing." Until the
plist is converted to a placeholder-substituted template (Phase
C), that manual edit is a required step on macOS.


## Platform support

The repository targets macOS and Linux as the supported
operational platforms. Windows occupies a deliberately narrow
position:

  * CI runs the full test suite on `windows-latest` to catch
    portability regressions (e.g., a future `import fcntl`
    slipping back into a module top-level).
  * Two test classes in `tests/test_locking_posix.py` are
    `@unittest.skipIf(sys.platform.startswith("win"), ...)` —
    documented, justified, outside the parity test surface.
  * `run.bat` is retained as a deprecated launcher.
  * End-to-end Windows behavior (server boot, scheduler
    cadence, watchlist writes, registry refresh under Windows
    file semantics) is NOT validated by CI.

In practice: a Windows clone can pass tests today; it has not
been validated as a daily operational environment, and the
README's "Windows is not officially supported" line reflects
the operational stance even though CI proves cross-platform
parity at the test-suite level. The two surfaces are
deliberately calibrated to different fidelity levels.


## What this document does NOT cover

  * How to reproduce a specific `run_id`'s replay across
    machines — replay identity is intra-chain by design; a
    fresh clone starts a new chain at epoch 1.
  * How to import another machine's audit chain — not a
    supported operation; the chain is append-only and locally
    rooted.
  * How to migrate scheduler cadence state between machines —
    same reasoning; cadence state is local lineage.

These are not bootstrap gaps. They are the chosen contract.
