import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.api.schemas import (
    AnalyticsSummaryResponse,
    GatewayRequest,
    GatewayResponse,
    RequestLogResponse,
)
from app.core.cache import get_cache_manager
from app.core.orchestrator import execute_llm_call
from app.core.preprocessing import preprocess_prompt
from app.core.router import get_token_count, route_prompt
from app.db.database import get_db, get_session_local
from app.db.models import RequestLog

logger = logging.getLogger(__name__)

router = APIRouter()


def persist_request_log(
    request_id: str,
    user_id: str,
    model_used: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: float,
    cost_usd: float,
    cache_hit: bool,
):
    """Inserts a telemetry record into PostgreSQL/SQLite."""
    session_factory = get_session_local()
    db = session_factory()
    try:
        log_entry = RequestLog(
            id=request_id,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6),
            cache_hit=cache_hit,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist telemetry log ({request_id}): {e}")
    finally:
        db.close()


@router.post(
    "/v1/completions",
    response_model=GatewayResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a prompt to the cost-optimized LLM gateway",
)
async def create_completion(
    request: GatewayRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Main gateway entrypoint:
    1. Preprocesses and normalizes prompt.
    2. Performs semantic vector search on Redis cache.
    3. If cache hit, returns cached response immediately.
    4. If cache miss, routes prompt based on complexity heuristics.
    5. Executes LiteLLM call with automated fallbacks.
    6. Stores new embedding & response in cache asynchronously.
    7. Persists detailed usage and cost telemetry to database.
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Step 1: Preprocess prompt
    clean_prompt = preprocess_prompt(request.prompt)
    if not clean_prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty after preprocessing.",
        )

    # Step 2: Semantic Cache Lookup
    cache_manager = get_cache_manager()
    cache_result = await cache_manager.search(clean_prompt)

    if cache_result.is_hit:
        # Cache Hit flow
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = get_token_count(clean_prompt)
        completion_tokens = get_token_count(cache_result.response or "")
        total_tokens = prompt_tokens + completion_tokens
        # Caching saves completion LLM call; only embedding cost was incurred
        cost_usd = cache_result.embedding_cost_usd

        # Persist telemetry immediately
        persist_request_log(
            request_id=request_id,
            user_id=request.user_id,
            model_used=cache_result.model_used or "cached",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cache_hit=True,
        )

        return GatewayResponse(
            response=cache_result.response or "",
            model_used=cache_result.model_used or "cached",
            cache_hit=True,
            cost_usd=round(cost_usd, 6),
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
        )

    # Step 3: Cache Miss -> Intelligent Routing
    target_tier, selected_model, estimated_tokens = route_prompt(clean_prompt)

    # Step 4: Execute Downstream LLM call via LiteLLM Orchestrator
    try:
        llm_result = await execute_llm_call(
            model_name=selected_model,
            prompt=clean_prompt,
            max_tokens=request.max_tokens or 500,
        )
    except Exception as e:
        logger.error(f"Orchestrator failed to execute completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream LLM execution failed: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    total_cost = round(llm_result["cost_usd"] + cache_result.embedding_cost_usd, 6)

    # Step 5: Save prompt vector & response to cache in background
    background_tasks.add_task(
        cache_manager.save,
        prompt=clean_prompt,
        response=llm_result["content"],
        model_used=llm_result["model_used"],
        embedding_vector=cache_result.embedding_vector,
    )

    # Step 6: Persist telemetry record to database
    persist_request_log(
        request_id=request_id,
        user_id=request.user_id,
        model_used=llm_result["model_used"],
        prompt_tokens=llm_result["prompt_tokens"],
        completion_tokens=llm_result["completion_tokens"],
        total_tokens=llm_result["total_tokens"],
        latency_ms=latency_ms,
        cost_usd=total_cost,
        cache_hit=False,
    )

    return GatewayResponse(
        response=llm_result["content"],
        model_used=llm_result["model_used"],
        cache_hit=False,
        cost_usd=total_cost,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
    )


@router.get(
    "/v1/analytics/requests/{request_id}",
    response_model=RequestLogResponse,
    status_code=status.HTTP_200_OK,
    summary="Get telemetry record for a specific request ID",
)
def get_request_analytics(request_id: str, db: Session = Depends(get_db)):
    """Retrieves detailed telemetry record from PostgreSQL by UUID."""
    record = db.query(RequestLog).filter(RequestLog.id == request_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request telemetry log with ID '{request_id}' not found.",
        )
    return record


@router.get(
    "/v1/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get aggregated gateway usage statistics",
)
def get_analytics_summary(
    start_date: Optional[str] = Query(None, description="Optional ISO start date filter"),
    end_date: Optional[str] = Query(None, description="Optional ISO end date filter"),
    db: Session = Depends(get_db),
):
    """
    Computes real-time aggregated metrics from the database:
    - total_cost_usd
    - total_requests
    - average_latency_ms
    - cache_hit_rate_percentage
    """
    query = db.query(RequestLog)

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            query = query.filter(RequestLog.timestamp >= start_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format. Expected ISO format (YYYY-MM-DDTHH:MM:SS).",
            )

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            query = query.filter(RequestLog.timestamp <= end_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format. Expected ISO format (YYYY-MM-DDTHH:MM:SS).",
            )

    total_requests = query.count()
    if total_requests == 0:
        return AnalyticsSummaryResponse(
            total_cost_usd=0.0,
            total_requests=0,
            average_latency_ms=0.0,
            cache_hit_rate_percentage=0.0,
        )

    # Compute aggregate sums & averages
    total_cost = query.with_entities(func.sum(RequestLog.cost_usd)).scalar() or 0.0
    avg_latency = query.with_entities(func.avg(RequestLog.latency_ms)).scalar() or 0.0
    cache_hits = query.filter(RequestLog.cache_hit == True).count()

    hit_rate = (cache_hits / total_requests) * 100.0

    return AnalyticsSummaryResponse(
        total_cost_usd=round(float(total_cost), 4),
        total_requests=int(total_requests),
        average_latency_ms=round(float(avg_latency), 2),
        cache_hit_rate_percentage=round(float(hit_rate), 2),
    )
