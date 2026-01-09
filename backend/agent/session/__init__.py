"""
Session Management Module for AI Citizen Support Assistant

This module provides session management capabilities using Redis as the backend store.
It supports session creation, retrieval, updates, and expiration handling.
"""

from .manager import SessionManager
from .redis_store import RedisSessionStore, SessionData

__all__ = [
    "SessionManager",
    "RedisSessionStore",
    "SessionData",
]
