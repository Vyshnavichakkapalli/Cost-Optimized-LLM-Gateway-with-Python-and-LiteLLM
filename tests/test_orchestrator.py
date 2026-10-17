import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.exceptions import APIConnectionError, RateLimitError
from app.config import (
    CachingConfig,
    FallbackRule,
    GatewayConfig,
    ModelsConfig,
    RoutingRule,
)
from app.core.orchestrator import execute_llm_call


@pytest.fixture
def mock_config():
    return GatewayConfig(
        models=ModelsConfig(
            high_tier="gpt-4o",
            mid_tier="claude-3-5-sonnet-20240620",
            low_tier="groq/llama3-8b-8192",
            embedding="text-embedding-3-small",
        ),
        caching=CachingConfig(
            enabled=True,
            similarity_threshold=0.95,
            ttl_seconds=86400,
            embedding_dimension=1536,
        ),
        routing_rules=[
            RoutingRule(name="default", condition="catch_all", target_tier="low_tier")
        ],
        fallbacks=[
            FallbackRule(primary="gpt-4o", backup="claude-3-5-sonnet-20240620"),
            FallbackRule(primary="claude-3-5-sonnet-20240620", backup="groq/llama3-8b-8192"),
        ],
    )


def make_mock_response(content: str, model: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    resp.model = model
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp.usage = usage
    return resp


@pytest.mark.asyncio
async def test_successful_llm_execution(mock_config):
    mock_resp = make_mock_response("Paris is the capital of France.", "gpt-4o")

    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        result = await execute_llm_call(
            model_name="gpt-4o",
            prompt="What is the capital of France?",
            config=mock_config,
        )

        assert result["content"] == "Paris is the capital of France."
        assert result["model_used"] == "gpt-4o"
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 30
        assert result["cost_usd"] >= 0.0
        assert result["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_fallback_on_rate_limit_error(mock_config):
    fallback_resp = make_mock_response("Response from Claude fallback", "claude-3-5-sonnet-20240620")

    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # First call (gpt-4o) raises RateLimitError, second call (claude-3-5-sonnet-20240620) succeeds
        mock_acompletion.side_effect = [
            RateLimitError(
                message="Rate limit exceeded on gpt-4o",
                llm_provider="openai",
                model="gpt-4o",
            ),
            fallback_resp,
        ]

        result = await execute_llm_call(
            model_name="gpt-4o",
            prompt="Analyze this quantum computing circuit.",
            config=mock_config,
        )

        assert mock_acompletion.call_count == 2
        # Assert fallback model was used
        assert result["model_used"] == "claude-3-5-sonnet-20240620"
        assert result["content"] == "Response from Claude fallback"


@pytest.mark.asyncio
async def test_fallback_on_connection_error(mock_config):
    fallback_resp = make_mock_response("Response from Groq fallback", "groq/llama3-8b-8192")

    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # claude-3-5-sonnet fails with APIConnectionError, groq succeeds
        mock_acompletion.side_effect = [
            APIConnectionError(
                message="Connection timeout to anthropic",
                llm_provider="anthropic",
                model="claude-3-5-sonnet-20240620",
            ),
            fallback_resp,
        ]

        result = await execute_llm_call(
            model_name="claude-3-5-sonnet-20240620",
            prompt="Summarize this article.",
            config=mock_config,
        )

        assert mock_acompletion.call_count == 2
        assert result["model_used"] == "groq/llama3-8b-8192"
        assert result["content"] == "Response from Groq fallback"


@pytest.mark.asyncio
async def test_raises_when_no_fallback_configured(mock_config):
    with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = RateLimitError(
            message="Rate limit on groq",
            llm_provider="groq",
            model="groq/llama3-8b-8192",
        )

        with pytest.raises(RateLimitError):
            await execute_llm_call(
                model_name="groq/llama3-8b-8192",
                prompt="Tell me a joke",
                config=mock_config,
            )
