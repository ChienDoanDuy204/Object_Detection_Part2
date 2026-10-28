#!/bin/bash
# Start both Backend and Frontend concurrently with clean shutdown on CTRL+C
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Free ports 8000 and 3000
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true

cleanup() {
    echo ""
    echo "Stopping Faster R-CNN Backend and Frontend..."
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 3000/tcp 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start Backend
./run_backend.sh &
BACKEND_PID=$!

# Wait 2 seconds for backend to start
sleep 2

# Start Frontend
./run_frontend.sh &
FRONTEND_PID=$!

echo ""
echo "🚀 Faster R-CNN Studio is running!"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "Press CTRL+C at any time to stop both servers."
echo ""

wait
