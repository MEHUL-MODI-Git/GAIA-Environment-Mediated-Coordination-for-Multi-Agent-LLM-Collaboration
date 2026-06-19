#!/bin/zsh
set -e
cd /Users/mehulmodi/University/URECA/GAIA
echo "=== QUEUE2 START $(date) ==="
echo "--- W6 primitive-ablation ---"
python3 experiments/w6_primitive/scripts/run_w6.py 2>&1 | grep -aE "== (full|untyped|blind|none)|Saved w6" | grep -avE "INFO|timestamp" || true
echo "--- W9 failure-injection ---"
python3 experiments/w9_inject/scripts/run_w9.py 2>&1 | grep -aE "== (none|FM-)|Saved w9" | grep -avE "INFO|timestamp" || true
echo "--- W3 emergent-specialization ---"
python3 experiments/w3_emergent/scripts/run_w3.py 2>&1 | grep -aE "== W3|distinct dominant|Saved w3" | grep -avE "INFO|timestamp" || true
echo "=== QUEUE2 DONE $(date) ==="
