#!/bin/bash
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null

echo ""
echo "Starting Investment Analytics Engine..."
echo "(set PORT to override the default 8010; falls back if busy)"
echo ""

# Server picks its own port (env PORT, default 8010; scans up if busy),
# prints the chosen URL, and opens the browser. No hardcoded port here
# so the launcher and the running server can never disagree.
exec python -m api.server
