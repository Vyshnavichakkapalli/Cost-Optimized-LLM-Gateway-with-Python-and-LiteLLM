from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class GatewayRequest(BaseModel):
    prompt: str = Field(..., description="The user's input prompt")
    user_id: str = Field(..., description="Identifier for the user making the request")
    max_tokens: Optional[int] = Field(500, description="Max tokens for completion")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional request metadata")


class GatewayResponse(BaseModel):
    response: str = Field(..., description="The LLM or cached output text")
    model_used: str = Field(..., description="Model identifier used to fulfill request")
    cache_hit: bool = Field(..., description="Whether response was served from cache")
    cost_usd: float = Field(..., description="Calculated total cost in USD")
    request_id: Optional[str] = Field(default=None, description="Unique telemetry request identifier")
    latency_ms: Optional[float] = Field(default=None, description="Total latency in milliseconds")


class RequestLogResponse(BaseModel):
    id: str = Field(..., description="UUID of the request log")
    timestamp: datetime = Field(..., description="Timestamp of the request")
    user_id: str = Field(..., description="User ID associated with request")
    model_used: str = Field(..., description="Model used to fulfill request")
    prompt_tokens: int = Field(..., description="Input prompt tokens")
    completion_tokens: int = Field(..., description="Output completion tokens")
    total_tokens: int = Field(..., description="Total tokens used")
    latency_ms: float = Field(..., description="Request latency in ms")
    cost_usd: float = Field(..., description="Total cost in USD")
    cache_hit: bool = Field(..., description="Whether served from cache")

    model_config = ConfigDict(from_attributes=True)


class AnalyticsSummaryResponse(BaseModel):
    total_cost_usd: float = Field(..., description="Total cost across all requests")
    total_requests: int = Field(..., description="Total number of logged requests")
    average_latency_ms: float = Field(..., description="Average latency in milliseconds")
    cache_hit_rate_percentage: float = Field(..., description="Percentage of requests served from cache")
