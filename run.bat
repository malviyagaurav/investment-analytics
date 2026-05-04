@echo off
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
