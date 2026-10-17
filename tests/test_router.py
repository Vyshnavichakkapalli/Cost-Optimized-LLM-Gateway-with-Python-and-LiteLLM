import pytest
from app.config import (
    CachingConfig,
    FallbackRule,
    GatewayConfig,
    ModelsConfig,
    RoutingRule,
)
from app.core.router import get_token_count, route_prompt


@pytest.fixture
def sample_gateway_config():
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
            RoutingRule(
                name="complex_keyword",
                condition="contains_keyword",
                keywords=["synthesize", "analyze", "critique", "evaluate"],
                target_tier="high_tier",
            ),
            RoutingRule(
                name="long_context",
                condition="max_tokens_exceeded",
                threshold=1000,
                target_tier="mid_tier",
            ),
            RoutingRule(
                name="default",
                condition="catch_all",
                target_tier="low_tier",
            ),
        ],
        fallbacks=[
            FallbackRule(primary="gpt-4o", backup="claude-3-5-sonnet-20240620"),
            FallbackRule(primary="claude-3-5-sonnet-20240620", backup="groq/llama3-8b-8192"),
        ],
    )


def test_keyword_routing_to_high_tier(sample_gateway_config):
    prompt = "Please synthesize the key findings from this annual financial report."
    tier, model, tokens = route_prompt(prompt, config=sample_gateway_config)
    assert tier == "high_tier"
    assert model == "gpt-4o"
    assert tokens > 0


def test_long_context_routing_to_mid_tier(sample_gateway_config):
    # Construct a prompt with > 1000 tokens without triggering the keywords
    prompt = ("This is a repetitive text line designed to simulate a very long document. " * 150)
    tokens = get_token_count(prompt)
    assert tokens >= 1000

    tier, model, _ = route_prompt(prompt, config=sample_gateway_config)
    assert tier == "mid_tier"
    assert model == "claude-3-5-sonnet-20240620"


def test_simple_prompt_routing_to_low_tier(sample_gateway_config):
    prompt = "What is the capital of France?"
    tier, model, tokens = route_prompt(prompt, config=sample_gateway_config)
    assert tier == "low_tier"
    assert model == "groq/llama3-8b-8192"


def test_dynamic_config_modification_alters_routing(sample_gateway_config):
    # Changing low_tier model dynamically
    sample_gateway_config.models.low_tier = "mistral/mistral-small"
    prompt = "Say hello!"
    tier, model, _ = route_prompt(prompt, config=sample_gateway_config)
    assert tier == "low_tier"
    assert model == "mistral/mistral-small"

    # Changing condition rule priority
    sample_gateway_config.routing_rules.insert(
        0,
        RoutingRule(
            name="short_math",
            condition="contains_keyword",
            keywords=["calculate", "math"],
            target_tier="mid_tier",
        ),
    )
    math_prompt = "Can you calculate 25 * 40?"
    tier, model, _ = route_prompt(math_prompt, config=sample_gateway_config)
    assert tier == "mid_tier"
    assert model == "claude-3-5-sonnet-20240620"
