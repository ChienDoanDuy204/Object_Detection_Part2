#!/bin/bash
# Script to launch the Faster R-CNN FastAPI backend
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate study_env python
PYTHON_BIN="/home/chiendoan/miniconda3/envs/study_env/bin/python"

# Kill any existing process on port 8000
fuser -k 8000/tcp 2>/dev/null || true

echo "Starting Faster R-CNN FastAPI Backend on http://localhost:8000..."
exec "$PYTHON_BIN" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
