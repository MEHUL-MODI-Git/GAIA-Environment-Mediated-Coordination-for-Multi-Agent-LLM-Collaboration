#!/bin/zsh
# Sequential autorun queue — runs the remaining stretch experiments ONE AT A
# TIME (no parallel API contention → clean results). Safe to launch in the
# background; each blocks the next.
set -e
cd /Users/mehulmodi/University/URECA/GAIA
echo "=== QUEUE START $(date) ==="
echo "--- NX5 branch-merge (Feature F) ---"
python3 experiments/nx5_branch/scripts/run_nx5.py 18 2>&1 | grep -aE "== branch_|Recovered|Saved nx5" | grep -avE "INFO|timestamp" || true
echo "--- NX8 prompt-injection ---"
python3 experiments/nx8_injection/scripts/run_nx8.py 2>&1 | grep -aE "== (plain_blackboard|gaia)|Saved nx8" | grep -avE "INFO|timestamp" || true
echo "--- W1 chaos / agent-dropout ---"
python3 experiments/w1_chaos/scripts/run_w1.py 2>&1 | grep -aE "== p_drop|Saved w1" | grep -avE "INFO|timestamp" || true
echo "=== QUEUE DONE $(date) ==="
