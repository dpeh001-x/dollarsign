#!/usr/bin/env bash
# Update the live predict.drdarylpeh.com deployment to the latest code.
#
# Run this ON THE VPS, from the repo root:
#     ./deploy/update.sh
#
# It pulls the current branch, rebuilds the container, and waits for the
# health check to pass so you know the new version is actually live.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "→ Updating from branch: $BRANCH"

git pull origin "$BRANCH"

echo "→ Rebuilding container..."
docker compose up -d --build

echo "→ Waiting for health check..."
for i in $(seq 1 30); do
    if curl -fs http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
        echo "✓ Live and healthy: https://predict.drdarylpeh.com"
        exit 0
    fi
    sleep 2
done

echo "✗ Health check did not pass in 60s. Check: docker logs -f dollarsign-predict"
exit 1
