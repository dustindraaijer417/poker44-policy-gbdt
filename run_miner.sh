#!/bin/bash
# Poker44 miner launcher. Set WALLET_NAME / HOTKEY / AXON_PORT before running.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

NETUID="${NETUID:-126}"
NETWORK="${NETWORK:-finney}"
WALLET_NAME="${WALLET_NAME:?set WALLET_NAME}"
HOTKEY="${HOTKEY:?set HOTKEY}"
AXON_PORT="${AXON_PORT:-8091}"
PM2_NAME="${PM2_NAME:-poker44_miner}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

# The manifest must identify the code actually being served. repo_commit is
# regex-checked by the validator and a mismatch can zero the miner's score.
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:?set POKER44_MODEL_REPO_URL to the public repo}"
export POKER44_MODEL_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git rev-parse HEAD)}"
export PYTHONPATH="$REPO_DIR"

ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.info
)
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VH <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  ARGS+=(--blacklist.allowed_validator_hotkeys "${VH[@]}")
else
  ARGS+=(--blacklist.force_validator_permit)
fi

pm2 delete "$PM2_NAME" 2>/dev/null || true
pm2 start "$REPO_DIR/.venv/bin/python" --name "$PM2_NAME" --interpreter none -- \
  "$REPO_DIR/neurons/miner.py" "${ARGS[@]}"
pm2 save
echo "started $PM2_NAME on port $AXON_PORT (commit $POKER44_MODEL_REPO_COMMIT)"
