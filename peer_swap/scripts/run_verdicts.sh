#!/usr/bin/env bash
# Ask the observer for a same-or-different verdict on every trial of every arm.
# Usage: bash run_verdicts.sh                (diners + publicgoods)
#        GAMES=publicgoods bash run_verdicts.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GAMES="${GAMES:-diners publicgoods}"
for game in $GAMES; do
  for arm in control sibling provider; do
    input="$ROOT/peer_swap/runs/raw/$game/2agents/anthropic_claude-haiku-5-5_endogenous_$arm.json"
    if [ ! -f "$input" ]; then
      echo "missing raw output: $input" >&2
      continue
    fi
    echo "=== verdicts: $game / $arm ==="
    python3 -I "$ROOT/peer_swap/scripts/verdict.py" \
      --input "$input" \
      --observer M \
      --partner Q \
      --model anthropic/claude-haiku-5-5 \
      --out "$ROOT/peer_swap/runs/verdicts_${game}_${arm}.json"
  done
done
