import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient
from litellm.exceptions import RateLimitError
from app.db.database import get_engine, init_db
from app.db.models import Base
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure database tables exist for each test."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def make_mock_llm_response(content: str, model: str):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    resp.model = model
    usage = MagicMock()
    usage.prompt_tokens = 15
    usage.completion_tokens = 25
    usage.total_tokens = 40
    resp.usage = usage
    return resp


@pytest.mark.asyncio
async def test_completion_routing_and_telemetry():
    mock_resp = make_mock_llm_response("Analysis complete.", "gpt-4o")

    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "prompt": "Please synthesize and analyze the quarterly results.",
                "user_id": "user_123",
            }
            res = await client.post("/v1/completions", json=payload)
            assert res.status_code == 200
            data = res.json()

            # Contract verification
            assert "response" in data
            assert data["response"] == "Analysis complete."
            assert data["model_used"] == "gpt-4o"
            assert data["cache_hit"] is False
            assert "cost_usd" in data
            assert data["cost_usd"] >= 0.0
            assert "request_id" in data

            req_id = data["request_id"]

            # Verify GET /v1/analytics/requests/{request_id}
            analytics_res = await client.get(f"/v1/analytics/requests/{req_id}")
            assert analytics_res.status_code == 200
            log_data = analytics_res.json()
            assert log_data["id"] == req_id
            assert log_data["user_id"] == "user_123"
            assert log_data["model_used"] == "gpt-4o"
            assert log_data["cache_hit"] is False
            assert log_data["prompt_tokens"] == 15
            assert log_data["completion_tokens"] == 25


@pytest.mark.asyncio
async def test_semantic_caching_flow():
    """
    Test sending the same prompt twice:
    - 1st request: Cache Miss, invokes LLM, cache_hit: False.
    - 2nd request: Cache Hit, bypasses LLM, cache_hit: True, latency lower.
    """
    mock_resp = make_mock_llm_response("Here are your password instructions.", "groq/llama3-8b-8192")

    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "prompt": "How do I reset my password?",
                "user_id": "user_cache_test",
            }

            # 1st request -> Cache Miss
            res1 = await client.post("/v1/completions", json=payload)
            assert res1.status_code == 200
            data1 = res1.json()
            assert data1["cache_hit"] is False
            assert mock_acompletion.call_count == 1

            # 2nd request with exact prompt -> Cache Hit!
            res2 = await client.post("/v1/completions", json=payload)
            assert res2.status_code == 200
            data2 = res2.json()
            assert data2["cache_hit"] is True
            assert data2["response"] == data1["response"]
            # acompletion should not be called again
            assert mock_acompletion.call_count == 1


@pytest.mark.asyncio
async def test_automatic_fallback_on_rate_limit():
    """
    Test that when the primary model fails with RateLimitError,
    the gateway automatically catches it, retries with the backup model,
    and returns HTTP 200 with model_used matching the backup model.
    """
    fallback_resp = make_mock_llm_response("Fallback response.", "claude-3-5-sonnet-20240620")

    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = [
            RateLimitError(
                message="Rate limit reached for gpt-4o",
                llm_provider="openai",
                model="gpt-4o",
            ),
            fallback_resp,
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "prompt": "Please evaluate this proposal.",
                "user_id": "user_fallback",
            }
            res = await client.post("/v1/completions", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["model_used"] == "claude-3-5-sonnet-20240620"
            assert data["response"] == "Fallback response."
            assert mock_acompletion.call_count == 2


@pytest.mark.asyncio
async def test_analytics_summary_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check empty summary
        res = await client.get("/v1/analytics/summary")
        assert res.status_code == 200
        summary = res.json()
        assert summary["total_requests"] == 0
        assert summary["total_cost_usd"] == 0.0

        # Non-existent request returns 404
        bad_req = await client.get("/v1/analytics/requests/non-existent-uuid")
        assert bad_req.status_code == 404
