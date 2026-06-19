#!/bin/zsh
set -e
cd /Users/mehulmodi/University/URECA/GAIA
echo "=== GROQ CHAIN START $(date) ==="
echo "--- C4-4 leave-one-out ---"
python3 experiments/cycle4/scripts/run_c4_4.py 2>&1 | grep -aE "\[-|\[full|Δacc|Saved c4_4" | grep -avE "INFO|timestamp" || true
echo "--- C4-5 counterfactual do-operator ---"
python3 experiments/cycle4/scripts/run_c4_5.py 2>&1 | grep -aE "\[(base|do_)|Saved c4_5" | grep -avE "INFO|timestamp" || true
echo "=== GROQ CHAIN DONE $(date) ==="
