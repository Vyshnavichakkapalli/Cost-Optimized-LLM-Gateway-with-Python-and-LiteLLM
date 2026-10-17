import pytest
from app.config import (
    CachingConfig,
    FallbackRule,
    GatewayConfig,
    ModelsConfig,
    RoutingRule,
)
from app.core.cache import SemanticCacheManager


@pytest.fixture
def cache_config():
    return GatewayConfig(
        models=ModelsConfig(
            high_tier="gpt-4o",
            mid_tier="claude-3-5-sonnet-20240620",
            low_tier="groq/llama3-8b-8192",
            embedding="text-embedding-3-small",
        ),
        caching=CachingConfig(
            enabled=True,
            similarity_threshold=0.90,
            ttl_seconds=86400,
            embedding_dimension=1536,
        ),
        routing_rules=[
            RoutingRule(name="default", condition="catch_all", target_tier="low_tier")
        ],
        fallbacks=[],
    )


@pytest.mark.asyncio
async def test_cache_miss_then_hit(cache_config):
    cache_mgr = SemanticCacheManager(config=cache_config)

    prompt = "How do I reset my account password?"
    # 1. Initial search should miss
    miss_result = await cache_mgr.search(prompt)
    assert not miss_result.is_hit

    # 2. Save response to cache
    await cache_mgr.save(
        prompt=prompt,
        response="Click on 'Forgot Password' on the login screen.",
        model_used="groq/llama3-8b-8192",
        embedding_vector=miss_result.embedding_vector,
    )

    # 3. Search again with the exact same prompt -> Cache Hit!
    hit_result = await cache_mgr.search(prompt)
    assert hit_result.is_hit
    assert hit_result.response == "Click on 'Forgot Password' on the login screen."
    assert hit_result.model_used == "groq/llama3-8b-8192"
    assert hit_result.similarity >= 0.90


@pytest.mark.asyncio
async def test_dissimilar_prompt_cache_miss(cache_config):
    cache_mgr = SemanticCacheManager(config=cache_config)

    prompt1 = "Explain quantum entanglement in physics."
    prompt2 = "What is the best recipe for chocolate cake?"

    await cache_mgr.save(
        prompt=prompt1,
        response="Quantum entanglement is a phenomenon...",
        model_used="gpt-4o",
    )

    result = await cache_mgr.search(prompt2)
    assert not result.is_hit
