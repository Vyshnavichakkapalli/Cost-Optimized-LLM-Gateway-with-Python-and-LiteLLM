import logging
import re
from typing import Optional, Tuple
from app.config import GatewayConfig, RoutingRule, get_gateway_config

logger = logging.getLogger(__name__)

# Lazy tiktoken encoder
_tokenizer = None


def get_token_count(text: str) -> int:
    """
    Counts estimated tokens in the prompt using tiktoken.
    Falls back to a fast character/word heuristic if tiktoken is unavailable.
    """
    global _tokenizer
    if not text:
        return 0

    try:
        if _tokenizer is None:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(_tokenizer.encode(text))
    except Exception as e:
        logger.debug(f"Tiktoken encoding failed or unavailable ({e}); using heuristic estimation.")
        # Fallback estimation: average 4 characters per token
        return max(1, len(text) // 4)


def matches_rule(prompt: str, token_count: int, rule: RoutingRule) -> bool:
    """Check if the given prompt and token count satisfy the routing rule condition."""
    condition = rule.condition.lower()

    if condition == "contains_keyword":
        if not rule.keywords:
            return False
        # Case-insensitive word boundary match or substring match
        prompt_lower = prompt.lower()
        for kw in rule.keywords:
            kw_clean = kw.strip().lower()
            if not kw_clean:
                continue
            # Regex word match preferred, fallback to substring
            pattern = rf"\b{re.escape(kw_clean)}\b"
            if re.search(pattern, prompt_lower):
                return True
        return False

    elif condition in ("max_tokens_exceeded", "token_count_exceeded", "tokens_greater_than"):
        threshold = rule.threshold or 0
        return token_count >= threshold

    elif condition in ("token_count_below", "token_count_under", "max_tokens_below", "tokens_less_than"):
        threshold = rule.threshold or 0
        return token_count < threshold

    elif condition in ("catch_all", "default"):
        return True

    logger.warning(f"Unknown routing condition: '{rule.condition}'. Skipping rule '{rule.name}'.")
    return False


def route_prompt(
    prompt: str,
    config: Optional[GatewayConfig] = None
) -> Tuple[str, str, int]:
    """
    Evaluates a preprocessed prompt against the routing rules in config.yaml.
    Returns:
        (target_tier, model_name, estimated_tokens)
    """
    cfg = config or get_gateway_config()
    token_count = get_token_count(prompt)

    for rule in cfg.routing_rules:
        if matches_rule(prompt, token_count, rule):
            target_tier = rule.target_tier
            model_name = cfg.get_model_for_tier(target_tier)
            logger.info(
                f"Prompt matched rule '{rule.name}' (condition='{rule.condition}'). "
                f"Routed to tier '{target_tier}' -> model '{model_name}' (tokens={token_count})"
            )
            return target_tier, model_name, token_count

    # If no rule matched, fallback to low_tier as safe default
    default_tier = "low_tier"
    default_model = cfg.models.low_tier
    logger.warning(
        f"No routing rule matched prompt. Defaulting to '{default_tier}' -> model '{default_model}'"
    )
    return default_tier, default_model, token_count
