"""
Session Manager for AI Citizen Support Assistant

Provides high-level session management with support for:
- Session lifecycle management (create, get, update, delete)
- Conversation history tracking
- Context preservation across interactions
- Session timeout and cleanup
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .redis_store import RedisSessionStore, SessionData

logger = logging.getLogger(__name__)


class SessionManager:
    """
    High-level session manager that coordinates session operations.

    This class provides an abstraction layer over the Redis store,
    handling session lifecycle, conversation context, and cleanup.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        session_timeout: int = 1800,  # 30 minutes default
        max_sessions_per_user: int = 5,
        max_conversation_history: int = 100,
    ):
        """
        Initialize the session manager.

        Args:
            redis_url: Redis connection URL
            session_timeout: Session timeout in seconds
            max_sessions_per_user: Maximum concurrent sessions per user
            max_conversation_history: Maximum messages to keep in history
        """
        self.store = RedisSessionStore(redis_url=redis_url)
        self.session_timeout = session_timeout
        self.max_sessions_per_user = max_sessions_per_user
        self.max_conversation_history = max_conversation_history
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the session manager and connect to Redis."""
        await self.store.connect()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager and disconnect from Redis."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.store.disconnect()
        logger.info("Session manager stopped")

    async def create_session(
        self,
        user_id: Optional[str] = None,
        room_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionData:
        """
        Create a new session.

        Args:
            user_id: Optional user identifier
            room_id: Optional LiveKit room ID
            metadata: Optional additional metadata

        Returns:
            The created session data

        Raises:
            ValueError: If max sessions per user exceeded
        """
        # Check session limit for user
        if user_id:
            user_sessions = await self.get_user_sessions(user_id)
            if len(user_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest = min(user_sessions, key=lambda s: s.created_at)
                await self.delete_session(oldest.session_id)
                logger.info(f"Removed oldest session {oldest.session_id} for user {user_id}")

        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            room_id=room_id,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=self.session_timeout),
            conversation_history=[],
            context={},
            metadata=metadata or {},
        )

        await self.store.save_session(session, ttl=self.session_timeout)

        # Track user's sessions
        if user_id:
            await self.store.add_user_session(user_id, session_id)

        logger.info(f"Created session {session_id} for user {user_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieve a session by ID.

        Args:
            session_id: The session identifier

        Returns:
            The session data if found, None otherwise
        """
        session = await self.store.get_session(session_id)
        if session and session.expires_at < datetime.utcnow():
            # Session expired, clean it up
            await self.delete_session(session_id)
            return None
        return session

    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all sessions for a user.

        Args:
            user_id: The user identifier

        Returns:
            List of session data objects
        """
        session_ids = await self.store.get_user_sessions(user_id)
        sessions = []
        for sid in session_ids:
            session = await self.get_session(sid)
            if session:
                sessions.append(session)
        return sessions

    async def update_session(
        self,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionData]:
        """
        Update session context and metadata.

        Args:
            session_id: The session identifier
            context: Context data to merge
            metadata: Metadata to merge

        Returns:
            Updated session data if found, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        now = datetime.utcnow()
        session.updated_at = now
        session.expires_at = now + timedelta(seconds=self.session_timeout)

        if context:
            session.context.update(context)
        if metadata:
            session.metadata.update(metadata)

        await self.store.save_session(session, ttl=self.session_timeout)
        return session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionData]:
        """
        Add a message to the conversation history.

        Args:
            session_id: The session identifier
            role: Message role (user, assistant, system)
            content: Message content
            message_metadata: Optional message metadata

        Returns:
            Updated session data if found, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": message_metadata or {},
        }

        session.conversation_history.append(message)

        # Trim history if exceeds max
        if len(session.conversation_history) > self.max_conversation_history:
            # Keep system messages and trim oldest regular messages
            system_msgs = [m for m in session.conversation_history if m["role"] == "system"]
            other_msgs = [m for m in session.conversation_history if m["role"] != "system"]

            # Keep the most recent messages
            trim_count = len(session.conversation_history) - self.max_conversation_history
            other_msgs = other_msgs[trim_count:]

            session.conversation_history = system_msgs + other_msgs

        now = datetime.utcnow()
        session.updated_at = now
        session.expires_at = now + timedelta(seconds=self.session_timeout)

        await self.store.save_session(session, ttl=self.session_timeout)
        return session

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: The session identifier
            limit: Optional limit on number of messages

        Returns:
            List of message dictionaries
        """
        session = await self.get_session(session_id)
        if not session:
            return []

        history = session.conversation_history
        if limit:
            history = history[-limit:]
        return history

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found
        """
        session = await self.store.get_session(session_id)
        if session and session.user_id:
            await self.store.remove_user_session(session.user_id, session_id)

        result = await self.store.delete_session(session_id)
        if result:
            logger.info(f"Deleted session {session_id}")
        return result

    async def refresh_session(self, session_id: str) -> Optional[SessionData]:
        """
        Refresh session expiration time.

        Args:
            session_id: The session identifier

        Returns:
            Updated session data if found, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        now = datetime.utcnow()
        session.updated_at = now
        session.expires_at = now + timedelta(seconds=self.session_timeout)

        await self.store.save_session(session, ttl=self.session_timeout)
        return session

    async def set_room_id(self, session_id: str, room_id: str) -> Optional[SessionData]:
        """
        Associate a LiveKit room with the session.

        Args:
            session_id: The session identifier
            room_id: The LiveKit room ID

        Returns:
            Updated session data if found, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        session.room_id = room_id
        session.updated_at = datetime.utcnow()

        await self.store.save_session(session, ttl=self.session_timeout)
        return session

    async def get_session_by_room(self, room_id: str) -> Optional[SessionData]:
        """
        Find a session by its associated room ID.

        Args:
            room_id: The LiveKit room ID

        Returns:
            Session data if found, None otherwise
        """
        return await self.store.get_session_by_room(room_id)

    async def get_active_session_count(self) -> int:
        """
        Get the count of active sessions.

        Returns:
            Number of active sessions
        """
        return await self.store.get_session_count()

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                cleaned = await self.store.cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")

    async def __aenter__(self) -> "SessionManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
