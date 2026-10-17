"""
Cost Optimization Simulation Script
Sends 50 diverse prompts to the LLM Gateway to evaluate:
1. Cache hit savings
2. Intelligent routing savings (low/mid tier vs high tier)
3. Total cost compared against a baseline where all 50 prompts use un-cached GPT-4o.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from httpx import ASGITransport, AsyncClient
from app.core.router import get_token_count
from app.db.database import get_engine
from app.db.models import Base
from app.main import app

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("simulation")

# 50 Diverse Prompts:
# - Category 1: Repetitive Prompts (Repeated queries to test semantic caching)
# - Category 2: Simple Extraction / Informational Prompts (routed to low_tier)
# - Category 3: Medium Context Tasks (routed to mid_tier)
# - Category 4: High Complexity Reasoning & Synthesis Tasks (routed to high_tier)
PROMPTS = [
    # Repetitive / Common queries (1-10)
    "How do I reset my account password?",
    "What is the process to reset my forgotten password?", # Semantic duplicate 1
    "How do I reset my password?", # Exact duplicate
    "What is your enterprise refund and cancellation policy?",
    "Can you explain the refund terms and conditions?", # Semantic duplicate 2
    "What is your enterprise refund and cancellation policy?", # Exact duplicate
    "Where can I find my API secret keys in the dashboard?",
    "How can I locate my API authentication keys?", # Semantic duplicate 3
    "What are your standard support operating hours?",
    "What are your standard support operating hours?", # Exact duplicate

    # Simple factual queries & short translations (11-25: routed to low_tier)
    "What is the capital city of Australia?",
    "Translate the word 'Welcome to our platform' into French and German.",
    "Extract the email addresses from: Contact support at help@example.com or sales@enterprise.io.",
    "Convert 100 kilometers into miles.",
    "Give me 5 synonyms for the word 'innovative'.",
    "List the primary colors in the RGB color model.",
    "What is the boiling point of water at sea level in Celsius?",
    "Extract the product names from: We offer CloudShield, DataVault, and StreamFlow services.",
    "Spell-check this sentence: The quik brown fox jump over the lazy dog.",
    "What does the HTTP 404 status code mean?",
    "Generate a short greeting email for a new SaaS user.",
    "What is the currency of Switzerland?",
    "Format this phone number into E.164 standard: (555) 123-4567.",
    "What is the JSON syntax for a list of three integers?",
    "Summarize this sentence in three words: Modern distributed microservices enable high availability.",

    # Complex reasoning & synthesis prompts (26-40: contain keywords like synthesize, analyze, critique, evaluate -> high_tier)
    "Please synthesize the strategic implications of adopting WebAssembly versus native microservices.",
    "Analyze the security trade-offs of using OAuth 2.0 PKCE versus client secret in mobile applications.",
    "Critique this software architecture proposal for a multi-tenant payment processing gateway.",
    "Evaluate the mathematical proof for time complexity in Dijkstra's shortest path algorithm.",
    "Synthesize the competitive landscape between Snowflake, Databricks, and Google BigQuery.",
    "Analyze the failure modes of eventual consistency in distributed Raft consensus clusters.",
    "Critique the database normalization schema proposed for a high-frequency trading matching engine.",
    "Evaluate the scalability limits of Redis Cluster when sharding across 100 nodes.",
    "Please analyze the memory overhead differences between asyncio coroutines and OS threads in Python.",
    "Synthesize an incident postmortem report outline for an S3 outage caused by DNS misconfiguration.",
    "Critique the cryptographic design of AES-GCM when reusing nonces.",
    "Evaluate whether a graph database or a relational database is better suited for social network fraud detection.",
    "Analyze the economic impact of GPU compute utilization bottlenecks on LLM inference costs.",
    "Synthesize the architectural trade-offs between zero-trust network access and traditional VPNs.",
    "Critique the following prompt engineering strategy for reducing hallucination in financial rag pipelines.",

    # Long context / Mid-tier prompts (41-50: long context > 1000 tokens)
    "Explain the history of computing from Charles Babbage to modern quantum computers with detailed timeline. " * 30,
    "Review this extensive corporate policy log detailing security compliance protocols across 50 international offices. " * 30,
    "Provide an exhaustive line-by-line code review of a distributed caching proxy implementation. " * 30,
    "Summarize all chapters of an operational handbook discussing incident triage, root cause analysis, and post-mortems. " * 30,
    "Draft a comprehensive compliance audit checklist for HIPAA, SOC2 Type II, and ISO 27001 certifications. " * 30,
    "Describe the evolution of relational database management systems from Edgar Codd's paper to Spanner. " * 30,
    "Explain the deep technical nuances of transformer self-attention mechanisms and FlashAttention optimization. " * 30,
    "Detail the operational lifecycle of a container from OCI specification to Linux cgroups and namespaces. " * 30,
    "Provide a comprehensive technical survey of columnar storage formats including Parquet, ORC, and Feather. " * 30,
    "Write an extensive tutorial on building resilient Kubernetes operators using Golang controller-runtime. " * 30,
]


def make_simulated_response(prompt: str, model: str):
    """Generates realistic responses and token usage for simulation."""
    resp = MagicMock()
    choice = MagicMock()

    if "password" in prompt.lower():
        text = "To reset your password, navigate to the Login page and click 'Forgot Password'. A reset link will be emailed."
    elif "refund" in prompt.lower():
        text = "Enterprise refunds are eligible within 30 days of purchase upon written request to support@example.com."
    elif "api" in prompt.lower():
        text = "Your API keys are located under Settings -> API Credentials in your user console."
    elif "support" in prompt.lower():
        text = "Our support team operates 24/7/365 with typical response times under 15 minutes."
    elif len(prompt) > 500:
        text = "Here is the comprehensive long-context analysis covering all requested specifications, architectural considerations, and implementation steps in detail."
    else:
        text = f"Processed request successfully using model {model}."

    choice.message.content = text
    resp.choices = [choice]
    resp.model = model

    p_tokens = get_token_count(prompt)
    c_tokens = get_token_count(text)
    usage = MagicMock()
    usage.prompt_tokens = p_tokens
    usage.completion_tokens = c_tokens
    usage.total_tokens = p_tokens + c_tokens
    resp.usage = usage
    return resp


def make_mock_embedding(text: str):
    """Generates realistic semantic embedding vectors with high similarity for related topics."""
    resp = MagicMock()
    # Base vector seeded on topic
    topic = "general"
    lower = text.lower()
    if "password" in lower or "reset" in lower:
        topic = "password_reset"
    elif "refund" in lower or "cancellation" in lower:
        topic = "refund_policy"
    elif "api" in lower or "key" in lower or "auth" in lower:
        topic = "api_keys"
    elif "support" in lower or "operating" in lower:
        topic = "support_hours"
    elif "synthesize" in lower or "analyze" in lower:
        topic = "deep_analysis"

    # Generate deterministic base vector for topic
    rng = np.random.RandomState(abs(hash(topic)) % (2**31))
    base_vec = rng.randn(1536).astype(np.float32)
    # Add tiny perturbation for specific wording
    noise_rng = np.random.RandomState(abs(hash(text)) % (2**31))
    noise = noise_rng.randn(1536).astype(np.float32) * 0.05
    vec = base_vec + noise
    vec = vec / np.linalg.norm(vec)

    item = {"embedding": vec.tolist()}
    resp.data = [item]
    resp.usage = {"total_tokens": len(text) // 4}
    return resp


async def run_simulation():
    print("=" * 80)
    print("STARTING 50-PROMPT COST OPTIMIZATION SIMULATION")
    print("=" * 80)

    # Initialize fresh DB
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Pricing constants (USD per 1,000 tokens)
    GPT4O_PROMPT_RATE = 0.005
    GPT4O_COMPLETION_RATE = 0.015

    results: List[Dict[str, Any]] = []
    total_gateway_cost = 0.0
    total_baseline_cost = 0.0
    cache_hits = 0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("app.core.cache.aembedding", new_callable=AsyncMock) as mock_aembedding:

            mock_aembedding.side_effect = lambda model, input, **kwargs: make_mock_embedding(input[0])
            for idx, prompt in enumerate(PROMPTS, 1):
                # Dynamically configure mock response based on routed model
                def side_effect(model, messages, **kwargs):
                    return make_simulated_response(messages[0]["content"], model)

                mock_acompletion.side_effect = side_effect

                res = await client.post(
                    "/v1/completions",
                    json={"prompt": prompt, "user_id": f"sim_user_{idx % 5}"},
                )
                assert res.status_code == 200, f"Request {idx} failed: {res.text}"
                data = res.json()

                # Calculate theoretical baseline cost (if sent to un-cached GPT-4o)
                p_tokens = get_token_count(prompt)
                c_tokens = get_token_count(data["response"])
                baseline_cost = (p_tokens / 1000.0 * GPT4O_PROMPT_RATE) + (c_tokens / 1000.0 * GPT4O_COMPLETION_RATE)

                gateway_cost = data["cost_usd"]
                is_hit = data["cache_hit"]
                if is_hit:
                    cache_hits += 1

                total_gateway_cost += gateway_cost
                total_baseline_cost += baseline_cost

                results.append({
                    "id": idx,
                    "prompt_sample": prompt[:45] + ("..." if len(prompt) > 45 else ""),
                    "model_used": data["model_used"],
                    "cache_hit": is_hit,
                    "gateway_cost": gateway_cost,
                    "baseline_cost": baseline_cost,
                })

        # Fetch analytics summary
        summary_res = await client.get("/v1/analytics/summary")
        summary_data = summary_res.json()

    # Calculate savings
    savings_usd = total_baseline_cost - total_gateway_cost
    savings_pct = (savings_usd / total_baseline_cost) * 100.0 if total_baseline_cost > 0 else 0.0

    print(f"\nCompleted {len(PROMPTS)} requests.")
    print(f"Cache Hits: {cache_hits} / {len(PROMPTS)} ({cache_hits / len(PROMPTS) * 100:.1f}%)")
    print(f"Total Theoretical Baseline Cost (100% GPT-4o): ${total_baseline_cost:.6f}")
    print(f"Total Actual Gateway Cost:                    ${total_gateway_cost:.6f}")
    print(f"Total Cost Saved:                             ${savings_usd:.6f}")
    print(f"Percentage Cost Reduction:                    {savings_pct:.2f}%\n")

    print("-" * 80)
    print(f"{'#':<3} | {'Prompt':<40} | {'Model':<20} | {'Cache':<5} | {'Cost ($)':<10}")
    print("-" * 80)
    for r in results[:15]:
        print(f"{r['id']:<3} | {r['prompt_sample']:<40} | {r['model_used']:<20} | {str(r['cache_hit']):<5} | ${r['gateway_cost']:.6f}")
    print(f"... [{len(results) - 15} more requests logged] ...")
    print("-" * 80)

    return {
        "total_requests": len(PROMPTS),
        "cache_hits": cache_hits,
        "cache_hit_rate": (cache_hits / len(PROMPTS)) * 100.0,
        "total_baseline_cost": round(total_baseline_cost, 6),
        "total_gateway_cost": round(total_gateway_cost, 6),
        "savings_usd": round(savings_usd, 6),
        "savings_percentage": round(savings_pct, 2),
        "summary_api": summary_data,
        "sample_rows": results,
    }


if __name__ == "__main__":
    asyncio.run(run_simulation())
