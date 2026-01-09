"""
Redis Session Store for AI Citizen Support Assistant

Low-level Redis operations for session persistence with support for:
- Session CRUD operations
- User-session mappings
- Room-session mappings
- Session expiration and cleanup
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)

# Redis key prefixes
SESSION_PREFIX = "session:"
USER_SESSIONS_PREFIX = "user_sessions:"
ROOM_SESSION_PREFIX = "room_session:"
SESSION_INDEX = "session_index"


@dataclass
class SessionData:
    """
    Data class representing a user session.

    Attributes:
        session_id: Unique session identifier
        user_id: Optional user identifier
        room_id: Optional LiveKit room ID
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        expires_at: Session expiration timestamp
        conversation_history: List of conversation messages
        context: Session context data
        metadata: Additional session metadata
    """

    session_id: str
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=datetime.utcnow)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["expires_at"] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create session from dictionary."""
        # Parse datetime strings
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)


class RedisSessionStore:
    """
    Redis-backed session storage implementation.

    Provides atomic operations for session management with support
    for clustering and high availability configurations.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 20,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ):
        """
        Initialize the Redis session store.

        Args:
            redis_url: Redis connection URL
            max_connections: Maximum pool connections
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
        """
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        self._pool = ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.max_connections,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)

        # Test connection
        try:
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Disconnected from Redis")

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client instance."""
        if not self._client:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    async def save_session(self, session: SessionData, ttl: int = 1800) -> None:
        """
        Save a session to Redis.

        Args:
            session: Session data to save
            ttl: Time-to-live in seconds
        """
        key = f"{SESSION_PREFIX}{session.session_id}"
        data = json.dumps(session.to_dict())

        async with self.client.pipeline(transaction=True) as pipe:
            # Save session data
            pipe.setex(key, ttl, data)

            # Add to session index
            pipe.sadd(SESSION_INDEX, session.session_id)

            # Map room to session if room_id exists
            if session.room_id:
                room_key = f"{ROOM_SESSION_PREFIX}{session.room_id}"
                pipe.setex(room_key, ttl, session.session_id)

            await pipe.execute()

        logger.debug(f"Saved session {session.session_id}")

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieve a session from Redis.

        Args:
            session_id: The session identifier

        Returns:
            Session data if found, None otherwise
        """
        key = f"{SESSION_PREFIX}{session_id}"
        data = await self.client.get(key)

        if data:
            try:
                return SessionData.from_dict(json.loads(data))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"Failed to deserialize session {session_id}: {e}")
                return None
        return None

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from Redis.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found
        """
        key = f"{SESSION_PREFIX}{session_id}"

        # Get session first to clean up room mapping
        session = await self.get_session(session_id)

        async with self.client.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.srem(SESSION_INDEX, session_id)

            if session and session.room_id:
                room_key = f"{ROOM_SESSION_PREFIX}{session.room_id}"
                pipe.delete(room_key)

            results = await pipe.execute()

        deleted = results[0] > 0
        if deleted:
            logger.debug(f"Deleted session {session_id}")
        return deleted

    async def add_user_session(self, user_id: str, session_id: str) -> None:
        """
        Associate a session with a user.

        Args:
            user_id: The user identifier
            session_id: The session identifier
        """
        key = f"{USER_SESSIONS_PREFIX}{user_id}"
        await self.client.sadd(key, session_id)

    async def remove_user_session(self, user_id: str, session_id: str) -> None:
        """
        Remove a session association from a user.

        Args:
            user_id: The user identifier
            session_id: The session identifier
        """
        key = f"{USER_SESSIONS_PREFIX}{user_id}"
        await self.client.srem(key, session_id)

    async def get_user_sessions(self, user_id: str) -> Set[str]:
        """
        Get all session IDs for a user.

        Args:
            user_id: The user identifier

        Returns:
            Set of session IDs
        """
        key = f"{USER_SESSIONS_PREFIX}{user_id}"
        return await self.client.smembers(key)

    async def get_session_by_room(self, room_id: str) -> Optional[SessionData]:
        """
        Find a session by its associated room ID.

        Args:
            room_id: The LiveKit room ID

        Returns:
            Session data if found, None otherwise
        """
        room_key = f"{ROOM_SESSION_PREFIX}{room_id}"
        session_id = await self.client.get(room_key)

        if session_id:
            return await self.get_session(session_id)
        return None

    async def get_session_count(self) -> int:
        """
        Get the count of active sessions.

        Returns:
            Number of sessions in the index
        """
        return await self.client.scard(SESSION_INDEX)

    async def get_all_session_ids(self) -> Set[str]:
        """
        Get all session IDs.

        Returns:
            Set of all session IDs
        """
        return await self.client.smembers(SESSION_INDEX)

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions from the index.

        Returns:
            Number of sessions cleaned up
        """
        session_ids = await self.get_all_session_ids()
        cleaned = 0

        for session_id in session_ids:
            key = f"{SESSION_PREFIX}{session_id}"
            exists = await self.client.exists(key)

            if not exists:
                # Session expired, remove from index
                await self.client.srem(SESSION_INDEX, session_id)
                cleaned += 1

        return cleaned

    async def extend_session_ttl(self, session_id: str, ttl: int) -> bool:
        """
        Extend the TTL of a session.

        Args:
            session_id: The session identifier
            ttl: New TTL in seconds

        Returns:
            True if TTL was extended, False if session not found
        """
        key = f"{SESSION_PREFIX}{session_id}"
        return await self.client.expire(key, ttl)

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.client.ping()
            return True
        except redis.ConnectionError:
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get session store statistics.

        Returns:
            Dictionary with store statistics
        """
        info = await self.client.info(section="memory")
        session_count = await self.get_session_count()

        return {
            "session_count": session_count,
            "used_memory": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "redis_version": info.get("redis_version", "unknown"),
        }

    async def __aenter__(self) -> "RedisSessionStore":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
