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
echo "Open http://127.0.0.1:8010 in your browser"
echo ""

open http://127.0.0.1:8010 &
uvicorn api.main:app --host 127.0.0.1 --port 8010
