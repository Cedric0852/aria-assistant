"""Health check endpoint for the Citizen Support API."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
import redis.asyncio as redis
from functools import lru_cache
import os

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    redis_connected: bool
    index_loaded: bool
    message: Optional[str] = None


@lru_cache
def get_redis_url() -> str:
    """Get Redis URL from environment."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


async def check_redis_connection() -> bool:
    """Check if Redis is reachable."""
    try:
        client = redis.from_url(get_redis_url())
        await client.ping()
        await client.close()
        return True
    except Exception:
        return False


async def check_index_status() -> bool:
    """Check if the vector index is loaded and ready."""
    try:
        # Import here to avoid circular imports
        from agent.rag.indexer import get_index_status
        return await get_index_status()
    except ImportError:
        # Indexer not yet implemented, return False
        return False
    except Exception:
        return False


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check endpoint.

    Returns the status of:
    - API service
    - Redis connection
    - Vector index status
    """
    redis_ok = await check_redis_connection()
    index_ok = await check_index_status()

    # Determine overall status
    if redis_ok and index_ok:
        status = "healthy"
        message = "All systems operational"
    elif redis_ok:
        status = "degraded"
        message = "Index not loaded - documents may not be searchable"
    elif index_ok:
        status = "degraded"
        message = "Redis not connected - sessions may not persist"
    else:
        status = "unhealthy"
        message = "Redis and index both unavailable"

    return HealthStatus(
        status=status,
        redis_connected=redis_ok,
        index_loaded=index_ok,
        message=message,
    )
