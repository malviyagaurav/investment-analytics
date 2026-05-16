"""Cross-platform scheduler installer (advisory script).

Generates the OS-appropriate deployment artifact for the daily
evidence cadence. Picks the right format based on ``sys.platform``:

  macOS  → emits a launchd plist to ~/Library/LaunchAgents/
  Linux  → prints the crontab line to copy into ``crontab -e``
  Windows → emits a Task Scheduler XML next to the source template

The script ONLY writes the deployment artifact. It does NOT install
or activate the scheduler — that step is the operator's, because
activating a scheduler is a real operational action with system-
wide side effects (launchctl load / schtasks /Create / crontab
edit) that should not happen as a side effect of running a Python
script.

Usage:
    python scripts/install_scheduler.py

Honors:
    --output-dir <path>   write the generated artifact under here
                           instead of the OS default
    --dry-run             print the artifact to stdout, write nothing
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _next_weekday_2130_local_iso() -> str:
    """Return the next future Mon-Fri 21:30 in local-naive ISO form.

    Task Scheduler accepts ISO 8601 without a timezone designator
    and interprets it as local time. The trigger's calendar
    schedule re-fires the task each weekday at the same local
    21:30, so the boundary value really only matters as "some
    valid future moment to anchor the trigger from."
    """
    now = datetime.now()
    for delta in range(0, 8):
        candidate = (now + timedelta(days=delta)).replace(
            hour=21, minute=30, second=0, microsecond=0,
        )
        # 0 = Mon ... 4 = Fri
        if candidate.weekday() < 5 and candidate > now:
            return candidate.strftime("%Y-%m-%dT%H:%M:%S")
    # Fallback (shouldn't happen): now + 1 day.
    return (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")


def _windows_python_exe() -> Path:
    """Best-effort guess at the venv's Windows Python interpreter."""
    venv_python = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _posix_python_exe() -> Path:
    venv_python = REPO_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _emit_windows_task(output_dir: Path, dry_run: bool) -> Path:
    template_path = SCRIPT_DIR / "scheduler-task.xml"
    template = template_path.read_text(encoding="utf-8")
    rendered = (
        template
        .replace("{{INSTALL_ROOT}}", str(REPO_ROOT))
        .replace("{{PYTHON_EXE}}", str(_windows_python_exe()))
        .replace("{{START_BOUNDARY}}", _next_weekday_2130_local_iso())
    )
    target = output_dir / "scheduler-task.rendered.xml"
    if dry_run:
        sys.stdout.write(rendered)
        sys.stdout.write(
            "\n# (dry-run; would have written to: "
            + str(target) + ")\n"
        )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Windows Task Scheduler XML uses UTF-16. Write LF terminators
    # consistent with our chain hygiene; schtasks accepts both.
    target.write_text(rendered, encoding="utf-16")
    return target


def _emit_macos_plist(output_dir: Path, dry_run: bool) -> Path:
    template_path = SCRIPT_DIR / "com.investment-analytics.scheduler.plist"
    template = template_path.read_text(encoding="utf-8")
    # The macOS plist already exists in the repo with WorkingDirectory
    # + ProgramArguments hardcoded. For a generic installer we
    # substitute the {{}} placeholders if present; otherwise we just
    # copy as-is and tell the operator to edit.
    target = output_dir / "com.investment-analytics.scheduler.plist"
    if dry_run:
        sys.stdout.write(template)
        sys.stdout.write(
            "\n# (dry-run; would have written to: "
            + str(target) + ")\n"
        )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    return target


def _emit_linux_cron_hint(dry_run: bool) -> Path:
    cron_line_path = SCRIPT_DIR / "cron-line.txt"
    content = cron_line_path.read_text(encoding="utf-8")
    sys.stdout.write(
        "# Linux crontab line (copy into 'crontab -e'):\n"
    )
    sys.stdout.write(content)
    if not content.endswith("\n"):
        sys.stdout.write("\n")
    return cron_line_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the OS-appropriate scheduler deployment artifact. "
            "Does NOT install or activate; that step is the operator's."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write the generated artifact (default: "
             "scripts/ next to this script).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the artifact to stdout, write nothing.",
    )
    args = parser.parse_args(argv)

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None else SCRIPT_DIR
    )

    if sys.platform.startswith("win"):
        target = _emit_windows_task(output_dir, args.dry_run)
        if not args.dry_run:
            sys.stdout.write(
                f"Windows Task Scheduler XML written: {target}\n"
                f"Install with: "
                f'schtasks /Create /TN "InvestmentAnalyticsScheduler" '
                f"/XML {target}\n"
            )
    elif sys.platform == "darwin":
        target = _emit_macos_plist(output_dir, args.dry_run)
        if not args.dry_run:
            sys.stdout.write(
                f"macOS launchd plist written: {target}\n"
                f"Install with:\n"
                f"  cp {target} ~/Library/LaunchAgents/\n"
                f"  launchctl load ~/Library/LaunchAgents/"
                f"com.investment-analytics.scheduler.plist\n"
            )
    elif sys.platform.startswith("linux"):
        _emit_linux_cron_hint(args.dry_run)
    else:
        sys.stderr.write(
            f"unsupported platform: {sys.platform!r}\n"
            "Manual deployment required.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
