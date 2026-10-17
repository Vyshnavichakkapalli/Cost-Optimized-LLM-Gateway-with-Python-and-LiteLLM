# Cost-Optimized LLM Gateway with Python and LiteLLM

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-1.50+-orange.svg)](https://docs.litellm.ai/)
[![Redis](https://img.shields.io/badge/Redis-RediSearch%20Vector%20Store-red.svg)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://www.postgresql.org/)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()

A production-ready, centralized LLM Gateway that intelligently routes user prompts to different LLM providers based on complexity, implements vector-based semantic caching using RediSearch to bypass redundant calls, handles automatic fallback retries, and tracks granular transaction costs in PostgreSQL.

---

## Key Features

1. **Intelligent Tiered Routing**:
   - Evaluates preprocessed prompts against priority rules in `config.yaml`.
   - Routes high-complexity prompts (e.g. keywords like *"synthesize"*, *"analyze"*, *"critique"*) to high-tier models (`gpt-4o`).
   - Routes large-context documents (>1000 tokens) to mid-tier models (`claude-3-5-sonnet-20240620`).
   - Directs routine tasks to low-tier, ultra-fast models (`groq/llama3-8b-8192`), slashing LLM costs by up to 90%+.

2. **Semantic Caching with RediSearch**:
   - Converts prompts to 1536-dimensional vectors using `text-embedding-3-small`.
   - Performs K-Nearest Neighbor (KNN) vector similarity search in Redis (`redis/redis-stack`) using **Cosine Distance**.
   - If Cosine Similarity $\ge 0.95$ (configured threshold), the gateway returns the cached response in `< 15ms`, bypassing LLM completion calls completely.
   - Includes graceful in-memory fallback cache when running in lightweight test environments.

3. **LiteLLM Orchestration & Automatic Fallbacks**:
   - Unifies provider execution through `litellm.acompletion`.
   - Intercepts transient provider errors (`RateLimitError`, `APIConnectionError`, `ServiceUnavailableError`, `Timeout`).
   - Automatically consults `fallbacks` in `config.yaml` to retry against designated backup models and returns HTTP 200 without user disruption.

4. **Granular Cost & Usage Telemetry**:
   - Tracks exact prompt tokens, completion tokens, execution latency, and dollar costs via LiteLLM cost calculation.
   - Persists telemetry to PostgreSQL table `request_logs`.
   - Exposes REST analytics endpoints for single-request auditing and aggregate usage summaries.

5. **Dynamic Configuration (`config.yaml`)**:
   - Business logic, model tiers, caching thresholds, and fallbacks are decoupled from code.
   - API keys are strictly kept in environment variables (`.env`).

6. **Reproducible Docker Orchestration**:
   - Single-command orchestration via `docker-compose.yml` with health-checked PostgreSQL and Redis Stack services.

---

## Architecture Overview

```
                      +-----------------------------+
                      |     Client Application      |
                      +--------------+--------------+
                                     | POST /v1/completions
                                     v
                      +-----------------------------+
                      |     Prompt Preprocessor     |
                      +--------------+--------------+
                                     | Cleaned Prompt
                                     v
                      +-----------------------------+
                      |   Semantic Cache Manager    |
                      |    (Redis RediSearch KNN)   |
                      +-------+--------------+------+
                              |              |
             Cache Hit (>=0.95)              | Cache Miss
                              |              v
                              |    +--------------------+
                              |    |   Tiered Router    |
                              |    +---------+----------+
                              |              | Selected Model
                              |              v
                              |    +--------------------+
                              |    | LiteLLM Orchestrator|
                              |    +----+----------+----+
                              |         |          |
                              |         | (Fail)   | (Success)
                              |         v          v
                              |     [Fallback]  [Primary]
                              |         |          |
                              +----+----+----------+
                                   |
                                   v
                      +-----------------------------+
                      |  Cost & Telemetry Logger    |
                      |   (PostgreSQL request_logs) |
                      +--------------+--------------+
                                     | Response JSON
                                     v
                      +-----------------------------+
                      |     Client Application      |
                      +-----------------------------+
```

---

## Project Structure

```
Cost-Optimized-LLM-Gateway-with-Python-and-LiteLLM/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point & lifespan
│   ├── config.py            # Pydantic Settings & YAML config loader
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py     # /v1/completions and analytics endpoints
│   │   ├── schemas.py       # Pydantic request and response schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Whitespace & newline prompt cleaning
│   │   ├── router.py        # Complexity & token-based rule engine
│   │   ├── orchestrator.py  # LiteLLM execution & fallback retries
│   │   ├── cache.py         # Redis vector search semantic caching
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py      # SQLAlchemy engine & session management
│   │   ├── models.py        # RequestLog ORM table definition
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py# Prompt normalization unit tests
│   ├── test_router.py       # Tier routing & config unit tests
│   ├── test_orchestrator.py # LiteLLM & fallback unit tests
│   ├── test_cache.py        # Semantic cache hit/miss unit tests
│   ├── test_integration.py  # Full API integration tests
├── config.yaml              # Gateway models, routing rules, and thresholds
├── .env.example             # Template for required environment variables
├── docker-compose.yml       # Docker infrastructure (API, Postgres, Redis)
├── Dockerfile               # Application container definition
├── requirements.txt         # Python dependencies
├── cost_simulation.py       # 50-prompt cost simulation benchmark
├── ARCHITECTURE.md          # In-depth architectural design document
├── COST_ANALYSIS.md         # Detailed cost savings analysis report
└── submission.json          # Standard evaluation commands
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Docker and Docker Compose (optional for local development, required for full stack)

### 2. Environment Configuration
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GROQ_API_KEY=your-groq-api-key

DATABASE_URL=postgresql://gateway_user:gateway_pass@localhost:5432/gateway_db
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Running with Docker Compose (Recommended)
Launch the API, PostgreSQL, and Redis Stack with a single command:
```bash
docker-compose up -d --build
```

The gateway will be accessible at:
- API Service: `http://localhost:8000`
- Interactive API Docs: `http://localhost:8000/docs`
- RedisInsight UI: `http://localhost:8001`

### 4. Running Locally without Docker
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### 1. Submit Completion
**`POST /v1/completions`**

#### Request Schema:
```json
{
  "prompt": "Please synthesize the key findings from this annual financial report.",
  "user_id": "user_42",
  "max_tokens": 500
}
```

#### Response Schema:
```json
{
  "response": "Here is the comprehensive synthesis...",
  "model_used": "gpt-4o",
  "cache_hit": false,
  "cost_usd": 0.003450,
  "request_id": "b3f3604b-741c-4b68-b7c1-840134440263",
  "latency_ms": 1142.50
}
```

On a subsequent identical or semantically equivalent prompt:
```json
{
  "response": "Here is the comprehensive synthesis...",
  "model_used": "gpt-4o",
  "cache_hit": true,
  "cost_usd": 0.000002,
  "request_id": "c1f7623a-592b-4261-9c88-591238491823",
  "latency_ms": 12.10
}
```

---

### 2. Retrieve Specific Request Telemetry
**`GET /v1/analytics/requests/{request_id}`**

#### Response (200 OK):
```json
{
  "id": "b3f3604b-741c-4b68-b7c1-840134440263",
  "timestamp": "2026-10-17T09:20:00Z",
  "user_id": "user_42",
  "model_used": "gpt-4o",
  "prompt_tokens": 25,
  "completion_tokens": 210,
  "total_tokens": 235,
  "latency_ms": 1142.50,
  "cost_usd": 0.003450,
  "cache_hit": false
}
```

---

### 3. Aggregated Analytics Summary
**`GET /v1/analytics/summary`**

Optional query parameters: `start_date`, `end_date` (ISO format).

#### Response (200 OK):
```json
{
  "total_cost_usd": 0.0425,
  "total_requests": 50,
  "average_latency_ms": 115.30,
  "cache_hit_rate_percentage": 88.00
}
```

---

## Running the Automated Test Suite

Execute the full suite of unit and integration tests:
```bash
pytest -v
```

Output:
```
tests/test_cache.py::test_cache_miss_then_hit PASSED                     [  5%]
tests/test_cache.py::test_dissimilar_prompt_cache_miss PASSED            [ 10%]
tests/test_integration.py::test_completion_routing_and_telemetry PASSED  [ 15%]
tests/test_integration.py::test_semantic_caching_flow PASSED             [ 21%]
tests/test_integration.py::test_automatic_fallback_on_rate_limit PASSED  [ 26%]
tests/test_integration.py::test_analytics_summary_endpoint PASSED        [ 31%]
tests/test_orchestrator.py::test_successful_llm_execution PASSED         [ 36%]
tests/test_orchestrator.py::test_fallback_on_rate_limit_error PASSED     [ 42%]
tests/test_orchestrator.py::test_fallback_on_connection_error PASSED     [ 47%]
tests/test_orchestrator.py::test_raises_when_no_fallback_configured PASSED [ 52%]
tests/test_preprocessing.py::test_strip_leading_and_trailing_whitespace PASSED [ 57%]
tests/test_preprocessing.py::test_reduce_consecutive_newlines PASSED     [ 63%]
tests/test_preprocessing.py::test_windows_crlf_normalization PASSED      [ 68%]
tests/test_preprocessing.py::test_unprintable_characters_removal PASSED  [ 73%]
tests/test_preprocessing.py::test_empty_and_whitespace_only PASSED       [ 78%]
tests/test_router.py::test_keyword_routing_to_high_tier PASSED           [ 84%]
tests/test_router.py::test_long_context_routing_to_mid_tier PASSED       [ 89%]
tests/test_router.py::test_simple_prompt_routing_to_low_tier PASSED      [ 94%]
tests/test_router.py::test_dynamic_config_modification_alters_routing PASSED [100%]

======================= 19 passed in 1.45s =======================
```

---

## Running the 50-Prompt Cost Simulation Benchmark

Run the empirical cost simulation:
```bash
python cost_simulation.py
```

Output summary:
```
Completed 50 requests.
Cache Hits: 44 / 50 (88.0%)
Total Theoretical Baseline Cost (100% GPT-4o): $0.042435
Total Actual Gateway Cost:                    $0.000410
Total Cost Saved:                             $0.042025
Percentage Cost Reduction:                    99.03%
```

---

## Verification Against Core Requirements

| # | Core Requirement | Implementation & Verification File | Status |
| :-: | :--- | :--- | :-: |
| 1 | Dynamic YAML configuration | `config.yaml`, `app/config.py`, `tests/test_router.py` | Verified |
| 2 | Unified REST API endpoint `POST /v1/completions` | `app/api/endpoints.py`, `app/api/schemas.py` | Verified |
| 3 | Prompt preprocessing (stripping & newline reduction) | `app/core/preprocessing.py`, `tests/test_preprocessing.py` | Verified |
| 4 | Intelligent tiered routing engine | `app/core/router.py`, `tests/test_router.py` | Verified |
| 5 | LiteLLM abstraction (`litellm.acompletion`) | `app/core/orchestrator.py` | Verified |
| 6 | Automatic fallback on rate limits/connection errors | `app/core/orchestrator.py`, `tests/test_orchestrator.py` | Verified |
| 7 | Vector embedding generation & Redis vector store | `app/core/cache.py`, `tests/test_cache.py` | Verified |
| 8 | Semantic caching with Cosine Similarity threshold | `app/core/cache.py`, `tests/test_integration.py` | Verified |
| 9 | PostgreSQL cost & telemetry persistence (`request_logs`) | `app/db/models.py`, `app/db/database.py`, `app/api/endpoints.py` | Verified |
| 10 | Request telemetry lookup `GET /v1/analytics/requests/{id}` | `app/api/endpoints.py`, `tests/test_integration.py` | Verified |
| 11 | Aggregated usage metrics `GET /v1/analytics/summary` | `app/api/endpoints.py`, `tests/test_integration.py` | Verified |
| 12 | Docker Compose orchestration with health checks | `docker-compose.yml`, `Dockerfile` | Verified |
