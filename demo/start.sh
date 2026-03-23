#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting agentkit Real Estate Demo..."
exec uvicorn demo.main:app --host 0.0.0.0 --port 8000 --reload
