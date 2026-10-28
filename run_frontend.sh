#!/bin/bash
# Script to launch the Next.js Frontend
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

# Ensure node and npm are in PATH
export PATH="/home/chiendoan/.local/node/bin:$PATH"

# Kill any existing process on port 3000
fuser -k 3000/tcp 2>/dev/null || true

echo "Starting Next.js Frontend on http://localhost:3000..."
exec npm run dev -- -p 3000
