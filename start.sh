#!/bin/sh
set -eu

API_PORT="${PORT:-8000}"

# Set bot's base URL to current hosted service (or localhost for local dev)
export CORTEX_BASE_URL="${CORTEX_BASE_URL:-"HOSTED_SERVICE_URL"}"

# Ensure required environment variables are set for both API and bot
export TELEGRAM_TOKEN="${TELEGRAM_TOKEN:?TELEGRAM_TOKEN is required}"

# Accept either API_KEYS or RAG_API_KEY, then normalize both.
# API expects API_KEYS; bot uses RAG_API_KEY.
if [ -z "${API_KEYS:-}" ] && [ -z "${RAG_API_KEY:-}" ]; then
  echo "ERROR: Either API_KEYS or RAG_API_KEY must be set" >&2
  exit 1
fi

if [ -z "${API_KEYS:-}" ] && [ -n "${RAG_API_KEY:-}" ]; then
  export API_KEYS="$RAG_API_KEY"
fi

if [ -z "${RAG_API_KEY:-}" ] && [ -n "${API_KEYS:-}" ]; then
  export RAG_API_KEY="${API_KEYS%%,*}"
fi

python -m uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID="$!"

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

python bot.py
BOT_EXIT_CODE="$?"

kill "$API_PID" 2>/dev/null || true
wait "$API_PID" 2>/dev/null || true
exit "$BOT_EXIT_CODE"