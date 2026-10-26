#!/usr/bin/env bash
# Run the peer-swap arms of the starter harness on the Respan gateway.
#
# Usage:
#   bash peer_swap/scripts/run_swap_experiment.sh                  # full run
#   ARMS=sibling TRIALS=1 ROUNDS=5 SWAP_AFTER=1 \
#     bash peer_swap/scripts/run_swap_experiment.sh                # pilot
#
# Environment overrides: ARMS, TRIALS, ROUNDS, SWAP_AFTER, OUT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="${HARNESS:-$ROOT/third_party/collusion_hackathon/llm-trust-breaking}"
OUT="${OUT:-$ROOT/peer_swap/runs/raw}"
BASE="anthropic/claude-haiku-5-5"
SIBLING="anthropic/claude-haiku-4-5"
PROVIDER="baseten/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"

GAME="${GAME:-diners}"
OBSERVER="${OBSERVER:-}"
TRIALS="${TRIALS:-8}"
ROUNDS="${ROUNDS:-8}"
SWAP_AFTER="${SWAP_AFTER:-3}"     # 0-based: rounds 1-4 original, 5-8 swapped
ARMS="${ARMS:-control,sibling,provider}"

mkdir -p "$OUT"
cd "$HARNESS"

run_arm() {
  local name="$1"
  local swap="$2"
  echo
  echo "================================================================"
  echo "ARM $name: Q -> $swap  (game=$GAME trials=$TRIALS rounds=$ROUNDS swap_after=$SWAP_AFTER)"
  echo "================================================================"
  EXTRA=()
  if [ -n "$OBSERVER" ]; then
    MODELS_FILE="$(mktemp)"
    printf '{"M": "%s", "Q": "%s"}' "$OBSERVER" "$BASE" > "$MODELS_FILE"
    EXTRA=(--agent-models-file "$MODELS_FILE")
  fi
  uv run python experiments/run_endogenous.py \
    --game "$GAME" \
    --agents 2 \
    --trials "$TRIALS" \
    --rounds "$ROUNDS" \
    --model "$BASE" \
    "${EXTRA[@]}" \
    --temperature 1.0 \
    --swap-after-round "$SWAP_AFTER" \
    --swap-agent Q \
    --swap-model "$swap" \
    --use-queue \
    --output-dir "$OUT" \
    --run-suffix "$name"
}

IFS=',' read -ra SELECTED <<< "$ARMS"
for arm in "${SELECTED[@]}"; do
  case "$arm" in
    control)  run_arm control  "$BASE" ;;
    sibling)  run_arm sibling  "$SIBLING" ;;
    provider) run_arm provider "$PROVIDER" ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
done

echo
echo "Raw outputs are under: $OUT"
