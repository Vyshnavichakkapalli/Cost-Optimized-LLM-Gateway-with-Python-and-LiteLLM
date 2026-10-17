import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()


class ModelsConfig(BaseModel):
    high_tier: str = Field(..., description="High-tier model (e.g. gpt-4o)")
    mid_tier: str = Field(..., description="Mid-tier model (e.g. claude-3-5-sonnet-20240620)")
    low_tier: str = Field(..., description="Low-tier model (e.g. groq/llama3-8b-8192)")
    embedding: str = Field("text-embedding-3-small", description="Embedding model for semantic cache")


class CachingConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = Field(0.95, ge=0.0, le=1.0)
    ttl_seconds: int = Field(86400, ge=1)
    embedding_dimension: int = Field(1536, ge=1)
    index_name: str = "idx:prompt_embeddings"


class RoutingRule(BaseModel):
    name: str
    condition: str  # "contains_keyword", "max_tokens_exceeded", "token_count_below", "catch_all"
    keywords: Optional[List[str]] = None
    threshold: Optional[int] = None
    target_tier: str  # "high_tier", "mid_tier", "low_tier"


class FallbackRule(BaseModel):
    primary: str
    backup: str


class GatewayConfig(BaseModel):
    models: ModelsConfig
    caching: CachingConfig
    routing_rules: List[RoutingRule]
    fallbacks: List[FallbackRule] = []

    def get_model_for_tier(self, tier: str) -> str:
        """Resolve a tier name into its configured model string."""
        tier_mapping = {
            "high_tier": self.models.high_tier,
            "mid_tier": self.models.mid_tier,
            "low_tier": self.models.low_tier,
        }
        if tier in tier_mapping:
            return tier_mapping[tier]
        raise ValueError(f"Unknown tier: {tier}. Must be one of high_tier, mid_tier, low_tier.")

    def get_fallback_model(self, primary_model: str) -> Optional[str]:
        """Lookup designated backup model from fallbacks configuration."""
        for rule in self.fallbacks:
            if rule.primary == primary_model:
                return rule.backup
        return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys (Strictly environment variables)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite:///./gateway.db"
    ASYNC_DATABASE_URL: Optional[str] = None

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_INDEX_NAME: str = "idx:prompt_embeddings"

    # Gateway App Config
    CONFIG_PATH: str = "config.yaml"
    ENVIRONMENT: str = "development"
    PORT: int = 8000
    HOST: str = "0.0.0.0"


_gateway_config: Optional[GatewayConfig] = None
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_gateway_config(config_path: Optional[str] = None) -> GatewayConfig:
    global _gateway_config
    settings = get_settings()
    path = config_path or settings.CONFIG_PATH

    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = Path(__file__).resolve().parent.parent / path

    if not resolved_path.exists():
        # Fallback to current working directory
        resolved_path = Path.cwd() / path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _gateway_config = GatewayConfig(**data)
    return _gateway_config


def get_gateway_config() -> GatewayConfig:
    global _gateway_config
    if _gateway_config is None:
        _gateway_config = load_gateway_config()
    return _gateway_config
