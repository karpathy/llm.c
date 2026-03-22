#!/usr/bin/env bash
# Integration test for llm-serve: happy-path smoke test.
# Starts the server, checks /health, sends a completion request,
# validates response structure, and tears down.
#
# Usage: ./test_serve_integration.sh [path/to/model.gguf]
# Requires: curl, jq

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { ((PASS++)); echo -e "  ${GREEN}✓${NC} $1"; }
fail() { ((FAIL++)); echo -e "  ${RED}✗${NC} $1"; }

# ---------------------------------------------------------------------------
# Config
MODEL="${1:-$HOME/Projects/TestInference/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf}"
PORT=18199  # high port to avoid conflicts
SERVER_PID=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Preflight checks

echo "=== llm-serve integration test ==="
echo ""

if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}ERROR: Model not found: $MODEL${NC}"
    echo "Pass a GGUF model path as the first argument."
    exit 1
fi

for cmd in curl jq; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}ERROR: $cmd not found${NC}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Build

echo "Building llm-serve..."
cd "$SCRIPT_DIR"
make llm-serve 2>&1 | tail -3
if [[ ! -x "./llm-serve" ]]; then
    echo -e "${RED}ERROR: Build failed — ./llm-serve not found${NC}"
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Start server

echo "Starting server on port $PORT..."
./llm-serve -m "$MODEL" -p "$PORT" -c 256 -t 2 &
SERVER_PID=$!

# Wait for the server to become ready (up to 30s for model load)
READY=0
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    # Check if process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "${RED}ERROR: Server exited prematurely${NC}"
        exit 1
    fi
    sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
    echo -e "${RED}ERROR: Server didn't become ready in 30s${NC}"
    exit 1
fi
echo "Server ready (PID $SERVER_PID)."
echo ""

# ---------------------------------------------------------------------------
# Test 1: GET /health

echo "Test 1: GET /health"
HEALTH=$(curl -sf "http://127.0.0.1:$PORT/health")
if echo "$HEALTH" | jq -e '.status == "ok"' >/dev/null 2>&1; then
    pass "/health returns {\"status\":\"ok\"}"
else
    fail "/health unexpected response: $HEALTH"
fi

# ---------------------------------------------------------------------------
# Test 2: POST /v1/completions — basic generation

echo "Test 2: POST /v1/completions (basic)"
RESP=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0.0
    }')

# 2a: Response is valid JSON
if echo "$RESP" | jq . >/dev/null 2>&1; then
    pass "Response is valid JSON"
else
    fail "Response is not valid JSON: $RESP"
fi

# 2b: Has choices array
if echo "$RESP" | jq -e '.choices | length > 0' >/dev/null 2>&1; then
    pass "Response has non-empty choices array"
else
    fail "Missing or empty choices array"
fi

# 2c: choices[0].text is a non-empty string
GEN_TEXT=$(echo "$RESP" | jq -r '.choices[0].text // ""')
if [[ -n "$GEN_TEXT" ]]; then
    pass "Generated text is non-empty: \"$(echo "$GEN_TEXT" | head -c 80)...\""
else
    fail "Generated text is empty"
fi

# 2d: Has finish_reason
if echo "$RESP" | jq -e '.choices[0].finish_reason' >/dev/null 2>&1; then
    pass "Response includes finish_reason"
else
    fail "Missing finish_reason"
fi

# 2e: Has usage info
if echo "$RESP" | jq -e '.usage.prompt_tokens > 0' >/dev/null 2>&1; then
    pass "Response includes usage.prompt_tokens > 0"
else
    fail "Missing or zero usage.prompt_tokens"
fi

if echo "$RESP" | jq -e '.usage.completion_tokens >= 0' >/dev/null 2>&1; then
    pass "Response includes usage.completion_tokens"
else
    fail "Missing usage.completion_tokens"
fi

# ---------------------------------------------------------------------------
# Test 3: POST /v1/completions with temperature/top_p

echo "Test 3: POST /v1/completions (with sampling params)"
RESP2=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Once upon a time",
        "max_tokens": 8,
        "temperature": 0.8,
        "top_p": 0.95
    }')

GEN_TEXT2=$(echo "$RESP2" | jq -r '.choices[0].text // ""')
if [[ -n "$GEN_TEXT2" ]]; then
    pass "Sampling generation returned text: \"$(echo "$GEN_TEXT2" | head -c 80)...\""
else
    fail "Sampling generation returned empty text"
fi

# ---------------------------------------------------------------------------
# Test 4: Error handling — missing prompt

echo "Test 4: Error handling"
ERR_RESP=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{}')

if [[ "$ERR_RESP" == "400" ]]; then
    pass "Empty body returns 400"
else
    fail "Expected 400 for missing prompt, got $ERR_RESP"
fi

# ---------------------------------------------------------------------------
# Test 5: 404 on unknown route

echo "Test 5: Unknown route"
NOT_FOUND=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/nonexistent")

if [[ "$NOT_FOUND" == "404" ]]; then
    pass "Unknown route returns 404"
else
    fail "Expected 404, got $NOT_FOUND"
fi

# ---------------------------------------------------------------------------
# Test 6: Streaming

echo "Test 6: POST /v1/completions (streaming)"
STREAM_RESP=$(curl -sf -N -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "prompt": "Hello",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": true
    }' 2>&1 || true)

if echo "$STREAM_RESP" | grep -q "data:"; then
    pass "Streaming response contains SSE data lines"
else
    fail "Streaming response missing SSE data lines"
fi

if echo "$STREAM_RESP" | grep -q '\[DONE\]'; then
    pass "Streaming response ends with [DONE]"
else
    fail "Streaming response missing [DONE] sentinel"
fi

# ---------------------------------------------------------------------------
# Summary

echo ""
echo "=== Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} ==="

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
