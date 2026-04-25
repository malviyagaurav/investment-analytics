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
echo Open http://127.0.0.1:8010 in your browser
echo.

start http://127.0.0.1:8010
uvicorn api.main:app --host 127.0.0.1 --port 8010
pause
