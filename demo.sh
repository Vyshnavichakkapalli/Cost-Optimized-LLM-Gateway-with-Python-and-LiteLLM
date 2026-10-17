#!/usr/bin/env bash
# Cost-Optimized LLM Gateway Video Demo Script
set -e

BASE_URL="http://127.0.0.1:8000"

echo "======================================================================"
echo "1. HEALTH CHECK ENDPOINT"
echo "======================================================================"
curl --noproxy "*" -s "${BASE_URL}/health" | python -m json.tool || curl --noproxy "*" -s "${BASE_URL}/health"
echo -e "\n"

echo "======================================================================"
echo "2. FUNCTIONALITY 1: LOW-TIER ROUTING (Routine query -> Groq / Llama 3)"
echo "======================================================================"
curl --noproxy "*" -s -X POST "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital city of France?",
    "user_id": "demo_user"
  }' | python -m json.tool || true
echo -e "\n"

echo "======================================================================"
echo "3. FUNCTIONALITY 2: SEMANTIC CACHING"
echo "--- Request A: Initial Prompt (Cache Miss) ---"
echo "======================================================================"
RESP_A=$(curl --noproxy "*" -s -X POST "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I reset my password?",
    "user_id": "demo_user"
  }')
echo "$RESP_A" | python -m json.tool || echo "$RESP_A"
echo -e "\n"

echo "--- Request B: Semantically Similar Prompt (Cache Hit Expected) ---"
RESP_B=$(curl --noproxy "*" -s -X POST "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the process to reset my forgotten password?",
    "user_id": "demo_user"
  }')
echo "$RESP_B" | python -m json.tool || echo "$RESP_B"
echo -e "\n"

echo "======================================================================"
echo "4. FUNCTIONALITY 3: HIGH-TIER ROUTING (Reasoning Keywords -> GPT-4o)"
echo "======================================================================"
RESP_HIGH=$(curl --noproxy "*" -s -X POST "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Please synthesize and critique this distributed microservice architecture.",
    "user_id": "demo_architect"
  }')
echo "$RESP_HIGH" | python -m json.tool || echo "$RESP_HIGH"
echo -e "\n"

echo "======================================================================"
echo "5. FUNCTIONALITY 4: SINGLE REQUEST TELEMETRY AUDIT"
echo "======================================================================"
REQ_ID=$(echo "$RESP_HIGH" | grep -o '"request_id": *"[^"]*"' | head -n1 | cut -d'"' -f4)
if [ -n "$REQ_ID" ]; then
  echo "Fetching telemetry record for Request ID: $REQ_ID"
  curl --noproxy "*" -s "${BASE_URL}/v1/analytics/requests/${REQ_ID}" | python -m json.tool || curl --noproxy "*" -s "${BASE_URL}/v1/analytics/requests/${REQ_ID}"
fi
echo -e "\n"

echo "======================================================================"
echo "6. FUNCTIONALITY 5: AGGREGATED USAGE & SAVINGS SUMMARY"
echo "======================================================================"
curl --noproxy "*" -s "${BASE_URL}/v1/analytics/summary" | python -m json.tool || curl --noproxy "*" -s "${BASE_URL}/v1/analytics/summary"
echo -e "\n"
