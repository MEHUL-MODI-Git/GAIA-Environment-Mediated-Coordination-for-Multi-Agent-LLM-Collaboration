#!/bin/bash
# Run MiniWoB++ experiments using the correct Python environment with MiniWoB++ installed
VENV_PYTHON="/Users/mehulmodi/University/URECA/GAIA copy/GAIA/.venv/bin/python"
SCRIPT="/Users/mehulmodi/University/URECA/GAIA/experiments/miniwob/scripts/run_miniwob.py"
exec "$VENV_PYTHON" "$SCRIPT" "$@"
