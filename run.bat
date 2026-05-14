@echo off
REM ============================================================
REM DEPRECATED — Windows is not officially supported.
REM This script is retained for legacy use but the project's
REM file-system primitives (atomic writes, file locking) are
REM tested against POSIX semantics only. Use macOS or Linux for
REM supported operation. See README "Supported platforms".
REM ============================================================
cd /d %~dp0

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate
pip install -q -r requirements.txt 2>nul

echo.
echo Starting Investment Analytics Engine...
echo (set PORT to override the default 8010; falls back if busy)
echo.

REM Server picks its own port (env PORT, default 8010; scans up if busy),
REM prints the chosen URL, and opens the browser.
python -m api.server
pause
