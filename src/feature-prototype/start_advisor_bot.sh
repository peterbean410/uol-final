#!/usr/bin/env bash
# Bring up the served Gemma-4 vLLM, point the advisor bot at it, and run it.
#
#   ./start_advisor_bot.sh                # port-forward + conversational bot
#   PORT=8081 ./start_advisor_bot.sh      # use a different local port
#   ./start_advisor_bot.sh --dry-run      # extra args pass through to the bot
#
# Needs the TELEGRAM_BOT_TOKEN in .env and cluster access via <forex>/kubeconfig.
# Cleans up the port-forward on exit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"      # -> .../Fintech/forex
PF_SCRIPT="$REPO_ROOT/scripts/port-forward-gemma4.sh"
PORT="${PORT:-8080}"
BASE="http://localhost:${PORT}/v1"

PF_PID=""
cleanup() {
  [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null || true
  pkill -f "port-forward.*gemma-4-31b-predictor" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

reachable() { curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$BASE/models" 2>/dev/null | grep -q '^2'; }

# 1) Port-forward the Gemma-4 vLLM (unless something already answers on the port).
if reachable; then
  echo "==> Gemma-4 already reachable at $BASE"
else
  [[ -f "$PF_SCRIPT" ]] || { echo "ERROR: $PF_SCRIPT not found" >&2; exit 1; }
  echo "==> Port-forwarding gemma-4-31b-predictor on localhost:${PORT} ..."
  bash "$PF_SCRIPT" "$PORT" >/tmp/gemma4-pf.log 2>&1 &
  PF_PID=$!
  printf "    waiting for %s " "$BASE"
  for i in $(seq 1 60); do
    if reachable; then echo " ready"; break; fi
    if ! kill -0 "$PF_PID" 2>/dev/null; then
      echo; echo "ERROR: port-forward exited, see /tmp/gemma4-pf.log:" >&2; tail -n 5 /tmp/gemma4-pf.log >&2; exit 1
    fi
    printf "."; sleep 2
    [[ "$i" -eq 60 ]] && { echo; echo "ERROR: endpoint not ready after 120s" >&2; exit 1; }
  done
fi

# 2) Point the bot at it, auto-discover the served model id from /v1/models.
MODEL="$(curl -s --max-time 5 "$BASE/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
export LLM_BASE_URL="$BASE"
[[ -n "$MODEL" ]] && export LLM_MODEL="$MODEL"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
echo "==> LLM_BASE_URL=$LLM_BASE_URL  LLM_MODEL=${LLM_MODEL:-<from .env>}"

# 3) Run the bot (foreground; Ctrl-C stops it and the cleanup tears down the port-forward).
echo "==> Starting the advisor bot, message /advice or ask a free-form question."
python3 "$SCRIPT_DIR/telegram_advisor_bot.py" "$@"
