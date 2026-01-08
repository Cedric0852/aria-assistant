"""Stats endpoints for API performance monitoring."""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.session.redis_store import RedisSessionStore
from agent.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["stats"])

STATS_KEY = "api:stats"
STATS_HISTORY_KEY = "api:stats:history"
STATS_TTL = 86400 * 7  # 7 days

# Singleton Redis store
_redis_store: Optional[RedisSessionStore] = None


async def get_redis_store() -> RedisSessionStore:
    """Get or create the Redis session store."""
    global _redis_store
    if _redis_store is None:
        _redis_store = RedisSessionStore(redis_url=settings.REDIS_URL)
        await _redis_store.connect()
    return _redis_store


class EndpointStats(BaseModel):
    """Stats for a single endpoint."""
    endpoint: str
    first_response_ms: float = Field(..., description="First response time in milliseconds")
    cached_response_ms: Optional[float] = Field(None, description="Cached response time in milliseconds")
    improvement: Optional[str] = Field(None, description="Improvement factor (e.g., '5.8x faster')")
    cache_hits: int = Field(0, description="Number of cache hits")
    cache_misses: int = Field(0, description="Number of cache misses")
    total_requests: int = Field(0, description="Total requests")
    avg_response_ms: float = Field(0, description="Average response time")
    ttft_ms: Optional[float] = Field(None, description="Time to first token (streaming endpoints)")
    cached_ttft_ms: Optional[float] = Field(None, description="Cached time to first token")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class StatsRecord(BaseModel):
    """Request model for recording stats."""
    endpoint: str = Field(..., description="Endpoint path (e.g., '/api/query/text')")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    cache_hit: bool = Field(False, description="Whether this was a cache hit")
    ttft_ms: Optional[float] = Field(None, description="Time to first token (for streaming)")


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    endpoints: Dict[str, EndpointStats]
    summary: Dict[str, Any]
    generated_at: str


async def record_stat(endpoint: str, response_time_ms: float, cache_hit: bool = False, ttft_ms: Optional[float] = None):
    """Record a stat to Redis.

    Args:
        endpoint: The endpoint path
        response_time_ms: Response time in milliseconds
        cache_hit: Whether this was a cache hit
        ttft_ms: Time to first token (for streaming endpoints)
    """
    try:
        store = await get_redis_store()

        stats_json = await store.client.hget(STATS_KEY, endpoint)
        if stats_json:
            stats = json.loads(stats_json)
        else:
            stats = {
                "endpoint": endpoint,
                "first_response_ms": response_time_ms,
                "cached_response_ms": None,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_requests": 0,
                "total_response_ms": 0,
                "ttft_ms": None,
                "cached_ttft_ms": None,
            }

        stats["total_requests"] += 1
        stats["total_response_ms"] = stats.get("total_response_ms", 0) + response_time_ms
        stats["avg_response_ms"] = stats["total_response_ms"] / stats["total_requests"]
        stats["last_updated"] = datetime.utcnow().isoformat()

        if cache_hit:
            stats["cache_hits"] += 1
            # Update cached response time (use min for best case)
            if stats["cached_response_ms"] is None or response_time_ms < stats["cached_response_ms"]:
                stats["cached_response_ms"] = response_time_ms
            # Update cached TTFT (for streaming)
            if ttft_ms is not None:
                if stats.get("cached_ttft_ms") is None or ttft_ms < stats["cached_ttft_ms"]:
                    stats["cached_ttft_ms"] = ttft_ms
        else:
            stats["cache_misses"] += 1
            # Update first response time (use recent value)
            stats["first_response_ms"] = response_time_ms
            # Update TTFT (for streaming)
            if ttft_ms is not None:
                stats["ttft_ms"] = ttft_ms

        if stats["cached_response_ms"] and stats["first_response_ms"]:
            improvement = stats["first_response_ms"] / stats["cached_response_ms"]
            stats["improvement"] = f"{improvement:.1f}x faster"

        await store.client.hset(STATS_KEY, endpoint, json.dumps(stats))

        # Also record to history (for time-series analysis)
        history_entry = {
            "endpoint": endpoint,
            "response_time_ms": response_time_ms,
            "cache_hit": cache_hit,
            "ttft_ms": ttft_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await store.client.lpush(f"{STATS_HISTORY_KEY}:{endpoint}", json.dumps(history_entry))
        await store.client.ltrim(f"{STATS_HISTORY_KEY}:{endpoint}", 0, 999)  # Keep last 1000
        await store.client.expire(f"{STATS_HISTORY_KEY}:{endpoint}", STATS_TTL)

        logger.debug(f"Recorded stat: {endpoint} - {response_time_ms}ms (cache_hit={cache_hit})")

    except Exception as e:
        logger.warning(f"Failed to record stat: {e}")


async def get_all_stats() -> Dict[str, EndpointStats]:
    """Get all endpoint stats from Redis."""
    try:
        store = await get_redis_store()
        all_stats = await store.client.hgetall(STATS_KEY)

        result = {}
        for endpoint, stats_json in all_stats.items():
            stats = json.loads(stats_json)
            result[endpoint] = EndpointStats(**stats)

        return result
    except Exception as e:
        logger.warning(f"Failed to get stats: {e}")
        return {}


@router.post("/stats/record")
async def record_stats(record: StatsRecord):
    """
    Record API performance stats.

    Use this endpoint to record response times for monitoring.

    Args:
        record: Stats record with endpoint, response time, and cache hit status

    Returns:
        Success confirmation
    """
    await record_stat(record.endpoint, record.response_time_ms, record.cache_hit)
    return {"status": "recorded", "endpoint": record.endpoint}


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get API performance stats.

    Returns stats for all endpoints including:
    - First response time (cache miss)
    - Cached response time (cache hit)
    - Improvement factor
    - Cache hit/miss counts
    - Average response time

    Returns:
        StatsResponse with all endpoint stats and summary
    """
    stats = await get_all_stats()

    total_requests = sum(s.total_requests for s in stats.values())
    total_cache_hits = sum(s.cache_hits for s in stats.values())
    total_cache_misses = sum(s.cache_misses for s in stats.values())
    cache_hit_rate = (total_cache_hits / total_requests * 100) if total_requests > 0 else 0

    summary = {
        "total_requests": total_requests,
        "total_cache_hits": total_cache_hits,
        "total_cache_misses": total_cache_misses,
        "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        "endpoints_tracked": len(stats),
    }

    return StatsResponse(
        endpoints=stats,
        summary=summary,
        generated_at=datetime.utcnow().isoformat(),
    )


@router.get("/stats/{endpoint:path}")
async def get_endpoint_stats(endpoint: str):
    """
    Get stats for a specific endpoint.

    Args:
        endpoint: The endpoint path (e.g., 'query/text')

    Returns:
        Stats for the specified endpoint
    """
    # Normalize endpoint path
    if not endpoint.startswith("/api/"):
        endpoint = f"/api/{endpoint}"

    stats = await get_all_stats()

    if endpoint not in stats:
        raise HTTPException(status_code=404, detail=f"No stats found for endpoint: {endpoint}")

    return stats[endpoint]


@router.get("/stats/history/{endpoint:path}")
async def get_endpoint_history(endpoint: str, limit: int = 100):
    """
    Get recent request history for an endpoint.

    Args:
        endpoint: The endpoint path
        limit: Maximum number of records to return (default: 100, max: 1000)

    Returns:
        List of recent requests with timestamps and response times
    """
    if not endpoint.startswith("/api/"):
        endpoint = f"/api/{endpoint}"

    limit = min(limit, 1000)

    try:
        store = await get_redis_store()
        history_key = f"{STATS_HISTORY_KEY}:{endpoint}"
        history = await store.client.lrange(history_key, 0, limit - 1)

        return {
            "endpoint": endpoint,
            "history": [json.loads(h) for h in history],
            "count": len(history),
        }
    except Exception as e:
        logger.warning(f"Failed to get history: {e}")
        return {"endpoint": endpoint, "history": [], "count": 0}


@router.delete("/stats/reset")
async def reset_stats():
    """
    Reset all stats.

    WARNING: This will delete all recorded stats.

    Returns:
        Confirmation of reset
    """
    try:
        store = await get_redis_store()

        await store.client.delete(STATS_KEY)

        # Delete all history keys
        cursor = 0
        while True:
            cursor, keys = await store.client.scan(cursor, match=f"{STATS_HISTORY_KEY}:*")
            if keys:
                await store.client.delete(*keys)
            if cursor == 0:
                break

        logger.info("Stats reset successfully")
        return {"status": "reset", "message": "All stats have been cleared"}
    except Exception as e:
        logger.error(f"Failed to reset stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset stats: {e}")
