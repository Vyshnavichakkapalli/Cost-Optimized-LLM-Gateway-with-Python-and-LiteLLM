import logging
import time
from typing import Any, Dict, Optional, Tuple
import litellm
from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    InternalServerError,
)
from app.config import GatewayConfig, get_gateway_config

logger = logging.getLogger(__name__)

# Standard retryable LiteLLM exceptions
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
    InternalServerError,
)

# Standard pricing fallbacks per 1k tokens (prompt / completion) in USD if litellm.completion_cost is None
FALLBACK_PRICING = {
    "gpt-4o": (0.005, 0.015),
    "claude-3-5-sonnet-20240620": (0.003, 0.015),
    "groq/llama3-8b-8192": (0.00005, 0.00008),
    "text-embedding-3-small": (0.00002, 0.00002),
}


def calculate_cost(
    completion_response: Any,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """Calculates request cost in USD using LiteLLM cost calculator or fallback rates."""
    try:
        cost = litellm.completion_cost(completion_response=completion_response)
        if cost is not None and cost >= 0.0:
            return float(cost)
    except Exception as e:
        logger.debug(f"litellm.completion_cost calculation failed ({e}), using fallback estimate.")

    # Fallback heuristic calculation
    p_rate, c_rate = FALLBACK_PRICING.get(model_name, (0.0015, 0.002))
    estimated = (prompt_tokens / 1000.0 * p_rate) + (completion_tokens / 1000.0 * c_rate)
    return round(float(estimated), 6)


async def execute_llm_call(
    model_name: str,
    prompt: str,
    max_tokens: int = 500,
    config: Optional[GatewayConfig] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Executes a prompt completion call using LiteLLM acompletion with automated fallback handling.

    Returns dictionary with:
        - content: str
        - model_used: str
        - prompt_tokens: int
        - completion_tokens: int
        - total_tokens: int
        - latency_ms: float
        - cost_usd: float
    """
    cfg = config or get_gateway_config()
    messages = [{"role": "user", "content": prompt}]
    start_time = time.perf_counter()

    attempted_model = model_name
    response = None

    try:
        logger.info(f"Invoking primary model: '{attempted_model}' via litellm.acompletion")
        response = await acompletion(
            model=attempted_model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
    except RETRYABLE_EXCEPTIONS as e:
        logger.warning(
            f"Primary model '{attempted_model}' failed with retryable error {type(e).__name__}: {e}. "
            f"Checking fallback configuration..."
        )
        fallback_model = cfg.get_fallback_model(attempted_model)
        if fallback_model:
            logger.info(f"Retrying request with configured backup model: '{fallback_model}'")
            attempted_model = fallback_model
            response = await acompletion(
                model=attempted_model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs
            )
            logger.info(f"Backup model '{attempted_model}' succeeded!")
        else:
            logger.error(f"No fallback configured for '{attempted_model}'. Re-raising exception.")
            raise
    except Exception as e:
        # Non-retryable error or other litellm exception
        logger.error(f"Unexpected error executing model '{attempted_model}': {e}")
        raise

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Parse response content
    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        content = str(response)

    # Parse tokens
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else max(1, len(prompt) // 4)
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else max(1, len(content) // 4)
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

    # Compute cost
    cost_usd = calculate_cost(response, attempted_model, prompt_tokens, completion_tokens)

    return {
        "content": content,
        "model_used": attempted_model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "latency_ms": round(latency_ms, 2),
        "cost_usd": cost_usd,
    }
