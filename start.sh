#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
APP="$SCRIPT_DIR/quant/app.py"

# Create venv on first run
if [ ! -f "$VENV/bin/activate" ]; then
    echo "First run — setting up virtual environment..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$SCRIPT_DIR/quant/requirements.txt"
    echo "Setup complete."
else
    source "$VENV/bin/activate"
fi

echo "Starting Long or Short? at http://localhost:8501"
cd "$SCRIPT_DIR/quant"
exec streamlit run app.py \
    --server.headless false \
    --server.port 8501 \
    --browser.gatherUsageStats false
