#!/usr/bin/env bash
# Launch the VLM Object Detection demo inside the conda environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting VLM Object Detection Demo..."
echo "Open http://localhost:8000 in your browser (or use the gradio.live HTTPS URL printed below for webcam)."
echo ""

conda run -n "gnd_dino+sam2_venv" --no-capture-output \
    python "$SCRIPT_DIR/app.py"
