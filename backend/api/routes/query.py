"""Query endpoints for text and audio queries."""

import asyncio
import hashlib
import json
import re
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator, Dict, List
from datetime import datetime, timedelta
import base64
import io
import os
import uuid
from pathlib import Path
from groq import Groq
from openai import OpenAI, AsyncOpenAI
import logging

from agent.session.redis_store import RedisSessionStore, SessionData
from agent.utils.config import settings
from api.routes.stats import record_stat

logger = logging.getLogger(__name__)

# Redis session store - singleton instance
_redis_store: Optional[RedisSessionStore] = None
SESSION_TTL_SECONDS = 1800  # 30 minutes


async def get_redis_store() -> RedisSessionStore:
    """Get or create the Redis session store."""
    global _redis_store
    if _redis_store is None:
        _redis_store = RedisSessionStore(redis_url=settings.REDIS_URL)
        await _redis_store.connect()
        logger.info(f"Connected to Redis at {settings.REDIS_URL}")
    return _redis_store


QUERY_CACHE_TTL = 3600  # 1 hour


def get_query_cache_key(query: str, session_id: Optional[str] = None) -> str:
    """Generate cache key for query (session-aware).

    Args:
        query: The query text
        session_id: Optional session ID for session-aware caching

    Returns:
        Cache key string. If session_id provided, cache is per-session.
    """
    normalized_query = query.lower().strip()
    if session_id:
        # Session-aware: same query in different sessions = different cache entries
        key_content = f"{session_id}:{normalized_query}"
    else:
        # Global: same query from anyone = same cache entry
        key_content = normalized_query
    query_hash = hashlib.md5(key_content.encode()).hexdigest()
    return f"query_cache:{query_hash}"


async def get_cached_response(query: str, session_id: Optional[str] = None) -> Optional[dict]:
    """Get cached response for query (session-aware).

    Args:
        query: The query text
        session_id: Optional session ID for session-aware lookup
    """
    try:
        store = await get_redis_store()
        cache_key = get_query_cache_key(query, session_id)
        cached = await store.client.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for query: {query[:50]}... (session={session_id})")
            return json.loads(cached)
        logger.debug(f"Cache MISS for query: {query[:50]}... (session={session_id})")
        return None
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None


async def cache_response(query: str, response: dict, session_id: Optional[str] = None):
    """Cache query response (session-aware).

    Args:
        query: The query text
        response: Response data to cache
        session_id: Optional session ID for session-aware caching
    """
    try:
        store = await get_redis_store()
        cache_key = get_query_cache_key(query, session_id)
        await store.client.setex(cache_key, QUERY_CACHE_TTL, json.dumps(response))
        logger.info(f"Cached response for: {query[:50]}... (session={session_id})")
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


# Audio caching (TTS responses)
AUDIO_CACHE_TTL = 3600  # 1 hour - same as query cache


async def get_cached_audio(query: str, session_id: Optional[str] = None) -> Optional[str]:
    """Get cached audio (base64) for a query (session-aware).

    Args:
        query: The query text
        session_id: Optional session ID for session-aware lookup

    Returns:
        Base64-encoded audio string if cached, None otherwise
    """
    try:
        store = await get_redis_store()
        # Use same key pattern as query cache but with audio prefix
        query_key = get_query_cache_key(query, session_id)
        cache_key = f"audio:{query_key}"
        cached = await store.client.get(cache_key)
        if cached:
            logger.info(f"Audio cache HIT for query: {query[:50]}... (session={session_id})")
            return cached
        logger.debug(f"Audio cache MISS for query: {query[:50]}... (session={session_id})")
        return None
    except Exception as e:
        logger.warning(f"Audio cache read failed: {e}")
        return None


async def cache_audio(query: str, audio_b64: str, session_id: Optional[str] = None):
    """Cache audio (base64) for a query (session-aware).

    Args:
        query: The query text
        audio_b64: Base64-encoded audio data
        session_id: Optional session ID for session-aware caching
    """
    try:
        store = await get_redis_store()
        query_key = get_query_cache_key(query, session_id)
        cache_key = f"audio:{query_key}"
        await store.client.setex(cache_key, AUDIO_CACHE_TTL, audio_b64)
        logger.info(f"Cached audio for: {query[:50]}... (session={session_id})")
    except Exception as e:
        logger.warning(f"Audio cache write failed: {e}")


async def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session from Redis."""
    try:
        store = await get_redis_store()
        session = await store.get_session(session_id)
        if session:
            # Extend TTL on access
            await store.extend_session_ttl(session_id, SESSION_TTL_SECONDS)
            return session.conversation_history
        return []
    except Exception as e:
        logger.error(f"Failed to get session history from Redis: {e}")
        return []


async def add_to_session_history(session_id: str, role: str, content: str):
    """Add a message to session history in Redis."""
    try:
        store = await get_redis_store()
        session = await store.get_session(session_id)

        if session is None:
            # Create new session
            session = SessionData(
                session_id=session_id,
                conversation_history=[],
                expires_at=datetime.utcnow() + timedelta(seconds=SESSION_TTL_SECONDS),
            )

        # Add message
        session.conversation_history.append({
            "role": role,
            "content": content
        })

        # Keep only last 20 messages
        if len(session.conversation_history) > 20:
            session.conversation_history = session.conversation_history[-20:]

        # Update timestamps
        session.updated_at = datetime.utcnow()
        session.expires_at = datetime.utcnow() + timedelta(seconds=SESSION_TTL_SECONDS)

        # Save to Redis
        await store.save_session(session, ttl=SESSION_TTL_SECONDS)
        logger.debug(f"Saved session {session_id} with {len(session.conversation_history)} messages")

    except Exception as e:
        logger.error(f"Failed to save session history to Redis: {e}")


async def close_redis_store():
    """Close the Redis connection gracefully."""
    global _redis_store
    if _redis_store is not None:
        await _redis_store.disconnect()
        _redis_store = None
        logger.info("Disconnected from Redis")

router = APIRouter(prefix="/api", tags=["query"])

# Chat UI history storage (separate from LLM conversation history)
CHAT_UI_KEY_PREFIX = "chat_ui:"
CHAT_UI_TTL = 86400 * 7  # 7 days


async def get_chat_ui_history(session_id: str) -> List[Dict]:
    """Get chat UI history for a session from Redis."""
    try:
        store = await get_redis_store()
        key = f"{CHAT_UI_KEY_PREFIX}{session_id}"
        data = await store.client.get(key)
        if data:
            return json.loads(data)
        return []
    except Exception as e:
        logger.error(f"Failed to get chat UI history: {e}")
        return []


async def save_chat_ui_history(session_id: str, messages: List[Dict]):
    """Save chat UI history to Redis."""
    try:
        store = await get_redis_store()
        key = f"{CHAT_UI_KEY_PREFIX}{session_id}"
        await store.client.setex(key, CHAT_UI_TTL, json.dumps(messages))
        logger.debug(f"Saved chat UI history for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save chat UI history: {e}")


class ChatMessage(BaseModel):
    """Chat UI message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    sources: Optional[List[Dict]] = Field(None, description="Source documents")
    confidence: Optional[float] = Field(None, description="Response confidence")
    audio_base64: Optional[str] = Field(None, description="Base64 audio (excluded from storage)")


class ChatHistoryRequest(BaseModel):
    """Request to save chat history."""
    messages: List[ChatMessage] = Field(..., description="Chat messages to save")


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat UI history for a session.

    Args:
        session_id: The session ID

    Returns:
        List of chat messages
    """
    messages = await get_chat_ui_history(session_id)
    return {"session_id": session_id, "messages": messages}


@router.post("/chat/history/{session_id}")
async def save_chat_history(session_id: str, request: ChatHistoryRequest):
    """Save chat UI history for a session.

    Args:
        session_id: The session ID
        request: Chat messages to save

    Returns:
        Success confirmation
    """
    # Convert to dict and exclude audio (too large to store)
    messages_to_save = []
    for msg in request.messages:
        msg_dict = msg.model_dump()
        # Remove audio_base64 to save space (can be regenerated)
        msg_dict.pop("audio_base64", None)
        messages_to_save.append(msg_dict)

    await save_chat_ui_history(session_id, messages_to_save)
    return {"status": "saved", "session_id": session_id, "message_count": len(messages_to_save)}


@router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat UI history for a session.

    Args:
        session_id: The session ID

    Returns:
        Success confirmation
    """
    try:
        store = await get_redis_store()
        key = f"{CHAT_UI_KEY_PREFIX}{session_id}"
        await store.client.delete(key)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


class TextQueryRequest(BaseModel):
    """Request model for text queries."""
    query: str = Field(..., min_length=1, description="The question to ask")
    session_id: Optional[str] = Field(None, description="Session ID for follow-up questions")
    include_audio: bool = Field(False, description="Whether to include TTS audio in response")


class QuerySource(BaseModel):
    """Source document reference."""
    title: str
    url: Optional[str] = None
    article_id: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str = Field(..., description="Original query (or transcript for audio)")
    answer: str = Field(..., description="Generated answer")
    sources: list[QuerySource] = Field(default_factory=list, description="Source documents used")
    session_id: str = Field(..., description="Session ID for follow-ups")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio response")


class AudioQueryResponse(QueryResponse):
    """Response model for audio queries (includes transcript)."""
    transcript: str = Field(..., description="Transcribed audio input")


def get_groq_client() -> Groq:
    """Get Groq client instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured",
        )
    return Groq(api_key=api_key)


def get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured",
        )
    return OpenAI(api_key=api_key)


_async_openai_client: Optional[AsyncOpenAI] = None


async def get_async_openai_client() -> AsyncOpenAI:
    """Get async OpenAI client with connection reuse."""
    global _async_openai_client
    if _async_openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        _async_openai_client = AsyncOpenAI(api_key=api_key)
    return _async_openai_client


async def transcribe_audio(audio_bytes: bytes, file_ext: str) -> str:
    """
    Transcribe audio using Groq Whisper.

    Args:
        audio_bytes: Raw audio file bytes
        file_ext: File extension (e.g., '.wav')

    Returns:
        Transcribed text
    """
    client = get_groq_client()

    # Create a file-like object with a name
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = f"audio{file_ext}"

    try:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        return transcription
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}",
        )


async def generate_tts_openai(text: str) -> Optional[bytes]:
    """
    Generate speech from text using OpenAI TTS.

    Args:
        text: Text to convert to speech

    Returns:
        Audio bytes (WAV format) or None if TTS fails
    """
    try:
        client = get_openai_client()
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="wav",
        )
        return response.content
    except Exception as e:
        logger.error(f"OpenAI TTS failed: {str(e)}")
        return None


async def generate_tts_openai_streaming(text: str) -> AsyncGenerator[bytes, None]:
    """
    Generate speech from text using OpenAI TTS with streaming.
    Uses gpt-4o-mini-tts for streaming support.

    Args:
        text: Text to convert to speech

    Yields:
        Audio bytes chunks (PCM format, 24kHz, 16-bit)
    """
    try:
        client = get_openai_client()
        # Use streaming response with PCM for low-latency
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="nova",
            input=text,
            response_format="pcm",  # Raw PCM for streaming (24kHz, 16-bit)
        ) as response:
            # Stream chunks as they arrive
            for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk
    except Exception as e:
        logger.error(f"OpenAI streaming TTS failed: {str(e)}")


async def collect_streaming_tts(text: str) -> Optional[bytes]:
    """
    Collect all chunks from streaming TTS into a single audio buffer.
    Returns complete audio with WAV header for caching.

    Args:
        text: Text to convert to speech

    Returns:
        Complete audio bytes with WAV header, or None if failed
    """
    try:
        chunks = []
        async for chunk in generate_tts_openai_streaming(text):
            chunks.append(chunk)

        if not chunks:
            return None

        # Combine all PCM chunks
        pcm_data = b''.join(chunks)

        # Add WAV header (PCM is 24kHz, 16-bit, mono)
        wav_audio = add_wav_header(pcm_data, sample_rate=24000, channels=1, bits_per_sample=16)
        return wav_audio
    except Exception as e:
        logger.error(f"Failed to collect streaming TTS: {e}")
        return None


def add_wav_header(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Add WAV header to raw PCM data.

    Args:
        pcm_data: Raw PCM audio data
        sample_rate: Sample rate in Hz (default 24000 for OpenAI)
        channels: Number of channels (1 for mono)
        bits_per_sample: Bits per sample (16 for OpenAI)

    Returns:
        Complete WAV file bytes
    """
    import struct

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    file_size = 36 + data_size

    # WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size for PCM
        1,   # AudioFormat (1 = PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )

    return header + pcm_data


async def generate_tts_groq(text: str) -> Optional[bytes]:
    """
    Generate speech from text using Groq TTS (fallback).

    Args:
        text: Text to convert to speech

    Returns:
        Audio bytes (WAV format) or None if TTS fails
    """
    try:
        client = get_groq_client()
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="hannah",
            input=text,
            response_format="wav",
        )
        # Groq returns BinaryAPIResponse - use read() to get bytes
        return response.read()
    except Exception as e:
        logger.error(f"Groq TTS failed: {str(e)}")
        return None


async def generate_tts(text: str) -> Optional[bytes]:
    """
    Generate speech from text. Uses OpenAI as primary (Groq is heavily rate-limited).

    Args:
        text: Text to convert to speech

    Returns:
        Audio bytes or None if all providers fail
    """
    # Use OpenAI as primary - Groq has aggressive rate limits (3600 TPD)
    audio = await generate_tts_openai(text)
    if audio:
        return audio

    # Fall back to Groq only if OpenAI fails
    logger.info("OpenAI TTS failed, falling back to Groq")
    return await generate_tts_groq(text)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked TTS."""
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


async def generate_tts_chunked(text: str) -> AsyncGenerator[bytes, None]:
    """Generate TTS audio in sentence chunks for faster first audio."""
    sentences = split_into_sentences(text)

    # Process sentences in parallel batches
    for sentence in sentences:
        if sentence.strip():
            audio = await generate_tts(sentence)
            if audio:
                yield audio


async def query_knowledge_base(query: str, session_id: Optional[str] = None) -> dict:
    """
    Query using LLM with RAG-grounded responses.

    The LLM decides whether to use knowledge base based on the query.
    For conversational queries (greetings, etc.), it responds directly.
    For information queries, it searches the knowledge base first.
    Maintains conversation history per session for context.

    Args:
        query: User question
        session_id: Optional session ID for context

    Returns:
        Dict with answer and sources
    """
    actual_session_id = session_id or str(uuid.uuid4())

    # Check cache first (session-aware)
    cached = await get_cached_response(query, actual_session_id)
    if cached:
        # Update session_id in cached response to current session
        cached["session_id"] = actual_session_id
        return cached

    try:
        # First, try RAG query which includes domain classification
        rag_context = ""
        sources = []
        try:
            from agent.rag.query_engine import query_knowledge_base as rag_query
            rag_result = await rag_query(query)

            # If query was rejected (off-topic, greeting, small_talk) or answered by classifier,
            # use the answer directly without making another OpenAI call
            if not rag_result.has_relevant_context and rag_result.confidence == 0.0:
                # Off-topic query - use the rejection message
                logger.info(f"Query rejected by classifier, returning rejection: {query[:50]}...")
                result = {
                    "answer": rag_result.answer,
                    "sources": [],
                    "session_id": actual_session_id,
                }
                await cache_response(query, result, actual_session_id)
                return result

            if rag_result.has_relevant_context and rag_result.confidence == 1.0 and not rag_result.sources:
                # Greeting or small talk - use the direct response
                logger.info(f"Greeting/small talk handled by classifier: {query[:50]}...")
                result = {
                    "answer": rag_result.answer,
                    "sources": [],
                    "session_id": actual_session_id,
                }
                await cache_response(query, result, actual_session_id)
                return result

            # Only use RAG context if we found relevant information
            if rag_result.has_relevant_context and rag_result.sources:
                sources = rag_result.sources
                # Build context from source excerpts
                context_parts = []
                for s in sources[:3]:  # Top 3 sources
                    excerpt = s.get("excerpt", "")
                    title = s.get("title", "")
                    if excerpt:
                        context_parts.append(f"[{title}]: {excerpt}")
                rag_context = "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG query failed, proceeding without context: {e}")

        # Use OpenAI for Irembo service queries that need RAG context
        client = get_openai_client()

        # Build system prompt (agent-style)
        system_prompt = """You are ARIA, an AI Citizen Support Assistant for Irembo, Rwanda's e-government platform.

Your role is to:
- Help citizens with questions about government services
- Provide accurate information from the knowledge base when available
- Be friendly, helpful, and conversational
- For greetings and casual conversation, respond naturally without requiring knowledge base info
- If asked about specific services and no relevant context is provided, acknowledge you don't have that specific information and suggest visiting irembo.gov.rw
- Remember information shared by the user in this conversation (like their name, what they're looking for, etc.)

Guidelines:
- Be concise but thorough
- Use numbered steps for procedures
- Cite sources when providing specific information from context
- Never make up fees, processing times, or requirements
- Refer back to earlier parts of the conversation when relevant"""

        # Build messages - start with system prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add RAG context if available
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Relevant information from knowledge base:\n\n{rag_context}"
            })

        # Get conversation history for this session and add to messages
        history = await get_session_history(actual_session_id)
        if history:
            logger.debug(f"Found {len(history)} messages in session {actual_session_id}")
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current user query
        messages.append({"role": "user", "content": query})

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # Store this exchange in session history
        await add_to_session_history(actual_session_id, "user", query)
        await add_to_session_history(actual_session_id, "assistant", answer)

        # Format sources for response
        formatted_sources = [
            {
                "title": s.get("title", "Unknown"),
                "url": s.get("url", ""),
                "article_id": s.get("doc_id", ""),
                "score": s.get("score", 0),
            }
            for s in sources
        ]

        result = {
            "answer": answer,
            "sources": formatted_sources,
            "session_id": actual_session_id,
        }

        # Cache successful response (session-aware)
        await cache_response(query, result, actual_session_id)

        return result

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {
            "answer": f"I encountered an error processing your question. Please try again.",
            "sources": [],
            "session_id": actual_session_id,
        }


@router.post("/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest) -> QueryResponse:
    """
    Submit a text query and get a response.

    Supports:
    - Text-only response (default)
    - Text + audio response (when include_audio=True)

    Args:
        request: Query request with question and options

    Returns:
        QueryResponse with answer, sources, and optional audio
    """
    start_time = time.perf_counter()
    cache_hit = False

    # Check if response will come from cache
    cached = await get_cached_response(request.query, request.session_id)
    if cached:
        cache_hit = True

    # Query the knowledge base
    result = await query_knowledge_base(request.query, request.session_id)

    # Convert sources to QuerySource models
    sources = [
        QuerySource(
            title=s.get("title", "Unknown"),
            url=s.get("url"),
            article_id=s.get("article_id"),
            score=s.get("score"),
        )
        for s in result.get("sources", [])
    ]

    response = QueryResponse(
        query=request.query,
        answer=result["answer"],
        sources=sources,
        session_id=result["session_id"],
    )

    # Generate TTS audio if requested (with caching)
    if request.include_audio:
        # Check audio cache first (session-aware)
        cached_audio = await get_cached_audio(request.query, result["session_id"])
        if cached_audio:
            response.audio_base64 = cached_audio
            cache_hit = True  # Audio was cached
        else:
            # Generate TTS and cache it
            audio_bytes = await generate_tts(result["answer"])
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                response.audio_base64 = audio_b64
                # Cache audio for future requests
                await cache_audio(request.query, audio_b64, result["session_id"])

    # Record stats (separate tracking for with/without audio)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    endpoint = "/api/query/text?include_audio=true" if request.include_audio else "/api/query/text?include_audio=false"
    asyncio.create_task(record_stat(endpoint, elapsed_ms, cache_hit))

    return response


class StreamingQueryRequest(BaseModel):
    """Request model for streaming text queries."""
    query: str = Field(..., min_length=1, description="The question to ask")
    session_id: Optional[str] = Field(None, description="Session ID for follow-up questions")


async def stream_query_response(query: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Stream a query response using Server-Sent Events format.

    Yields SSE-formatted chunks with the response.
    Includes conversation history for follow-up questions.
    Uses response cache to avoid redundant API calls for identical queries.
    """
    actual_session_id = session_id or str(uuid.uuid4())
    max_retries = 2
    heartbeat_interval = 15  # seconds

    # Validate input
    if not query or not query.strip():
        yield f"data: {json.dumps({'type': 'error', 'error': 'Empty query provided'})}\n\n"
        return

    # ==========================================================================
    # LAYER 0: Domain Classification (before cache check for proper filtering)
    # ==========================================================================
    try:
        from agent.rag.domain_classifier import classify_query, QueryCategory, OFF_TOPIC_RESPONSE

        logger.info(f"[STREAM] Classifying query: '{query[:50]}'")
        classification = await classify_query(query)
        logger.info(f"[STREAM] Classification: {classification.category.value} (reasoning: {classification.reasoning[:50]}...)")

        # Check if classifier failed
        classifier_failed = classification.reasoning.startswith("Classification failed")

        if not classifier_failed:
            # Handle greetings - direct response, no RAG needed
            if classification.category == QueryCategory.GREETING:
                logger.info(f"[STREAM] Greeting detected, returning direct response")
                yield f"data: {json.dumps({'type': 'session', 'session_id': actual_session_id})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'confidence': 1.0})}\n\n"
                response = classification.direct_response or "Hello! How can I help you with Irembo services today?"
                for i in range(0, len(response), 20):
                    yield f"data: {json.dumps({'type': 'token', 'token': response[i:i+20]})}\n\n"
                    await asyncio.sleep(0.01)
                yield f"data: {json.dumps({'type': 'done', 'full_response': response})}\n\n"
                return

            # Handle small talk - direct response, no RAG needed
            if classification.category == QueryCategory.SMALL_TALK:
                logger.info(f"[STREAM] Small talk detected, returning direct response")
                yield f"data: {json.dumps({'type': 'session', 'session_id': actual_session_id})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'confidence': 1.0})}\n\n"
                response = classification.direct_response or "I'm here to help with Irembo government services!"
                for i in range(0, len(response), 20):
                    yield f"data: {json.dumps({'type': 'token', 'token': response[i:i+20]})}\n\n"
                    await asyncio.sleep(0.01)
                yield f"data: {json.dumps({'type': 'done', 'full_response': response})}\n\n"
                return

            # Handle off-topic queries - polite decline
            if classification.category == QueryCategory.OFF_TOPIC:
                logger.info(f"[STREAM] Off-topic query rejected: '{query[:50]}'")
                yield f"data: {json.dumps({'type': 'session', 'session_id': actual_session_id})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'confidence': 0.0})}\n\n"
                for i in range(0, len(OFF_TOPIC_RESPONSE), 20):
                    yield f"data: {json.dumps({'type': 'token', 'token': OFF_TOPIC_RESPONSE[i:i+20]})}\n\n"
                    await asyncio.sleep(0.01)
                yield f"data: {json.dumps({'type': 'done', 'full_response': OFF_TOPIC_RESPONSE})}\n\n"
                return
        else:
            # Classifier failed - use fallback pattern matching
            logger.warning(f"[STREAM] Classifier failed, using fallback pattern matching")
            from agent.rag.query_engine import is_off_topic_query
            from agent.rag.prompts import NO_INFORMATION_RESPONSE
            if is_off_topic_query(query):
                logger.info(f"[STREAM] Query rejected by fallback pattern: '{query[:50]}'")
                yield f"data: {json.dumps({'type': 'session', 'session_id': actual_session_id})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'confidence': 0.0})}\n\n"
                for i in range(0, len(NO_INFORMATION_RESPONSE), 20):
                    yield f"data: {json.dumps({'type': 'token', 'token': NO_INFORMATION_RESPONSE[i:i+20]})}\n\n"
                    await asyncio.sleep(0.01)
                yield f"data: {json.dumps({'type': 'done', 'full_response': NO_INFORMATION_RESPONSE})}\n\n"
                return

    except Exception as e:
        logger.error(f"[STREAM] Classification error: {e}, continuing to RAG")

    # Check cache FIRST before any API calls (session-aware)
    cached = await get_cached_response(query, actual_session_id)
    if cached:
        logger.info(f"Streaming from cache for: {query[:50]}... (session={actual_session_id})")
        yield f"data: {json.dumps({'type': 'session', 'session_id': actual_session_id})}\n\n"
        answer = cached.get("answer", "")
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
            await asyncio.sleep(0.01)
        # Send done first, then sources (order: text -> done -> sources -> audio)
        yield f"data: {json.dumps({'type': 'done', 'full_response': answer})}\n\n"
        sources = cached.get("sources", [])
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'confidence': 0.9})}\n\n"
        return

    try:
        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': actual_session_id})}\n\n"

        # Get RAG context (non-blocking, optional)
        rag_context = ""
        sources = []
        try:
            from agent.rag.query_engine import get_or_create_index
            from agent.rag.indexer import DEFAULT_INDEX_DIR
            from llama_index.core.retrievers import VectorIndexRetriever

            index = await get_or_create_index(index_dir=DEFAULT_INDEX_DIR)
            if index:
                # Just retrieve documents, don't use LlamaIndex for response generation
                retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
                nodes = retriever.retrieve(query)

                if nodes:
                    context_parts = []
                    for node in nodes[:3]:  # Top 3 sources
                        metadata = node.node.metadata or {}
                        title = metadata.get("title", "Unknown")
                        excerpt = node.node.text[:500] if node.node.text else ""
                        if excerpt:
                            context_parts.append(f"[{title}]: {excerpt}")
                        # Use node_id or id_ instead of doc_id (TextNode attribute)
                        node_id = metadata.get("article_id") or getattr(node.node, 'node_id', None) or getattr(node.node, 'id_', "") or ""
                        sources.append({
                            "doc_id": node_id,
                            "title": title,
                            "url": metadata.get("url", ""),
                            "score": round(node.score, 4) if node.score else 0.0,
                            "excerpt": excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
                        })
                    rag_context = "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG retrieval failed, proceeding without context: {e}")

        # Build system prompt with RAG context
        system_prompt = """You are ARIA, an AI Citizen Support Assistant for Irembo, Rwanda's e-government platform.

Your role is to:
- Help citizens with questions about government services
- Provide accurate information from the knowledge base when available
- Be friendly, helpful, and conversational
- For greetings and casual conversation, respond naturally
- Remember information shared by the user in this conversation (like their name, what they're looking for, etc.)
- If asked about specific services and no relevant context is provided, acknowledge you don't have that specific information

Guidelines:
- Be concise but thorough
- Use numbered steps for procedures
- Cite sources when providing specific information
- Never make up fees, processing times, or requirements
- Refer back to earlier parts of the conversation when relevant"""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add RAG context if available
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Relevant information from knowledge base:\n\n{rag_context}"
            })

        # Add conversation history for follow-ups
        history = await get_session_history(actual_session_id)
        if history:
            logger.info(f"Including {len(history)} messages from session history")
            for msg in history[-10:]:  # Last 10 messages
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current query
        messages.append({"role": "user", "content": query})

        # Calculate confidence for later use (sources sent after text streaming)
        confidence = 0.0
        if sources:
            confidence = max(s.get("score", 0) for s in sources)

        # Stream response from OpenAI directly (with history support)
        client = get_openai_client()
        stream = None

        for attempt in range(max_retries + 1):
            try:
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.0,
                    stream=True,
                )
                break
            except Exception as stream_error:
                logger.warning(f"OpenAI stream attempt {attempt + 1} failed: {stream_error}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat(), 'retry': attempt + 1})}\n\n"
                else:
                    raise stream_error

        if stream is None:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to create stream after retries'})}\n\n"
            return

        full_response = ""
        last_heartbeat = asyncio.get_event_loop().time()

        for chunk in stream:
            if chunk is None or not hasattr(chunk, 'choices') or not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice is None or not hasattr(choice, 'delta') or choice.delta is None:
                continue
            content = getattr(choice.delta, 'content', None)
            if content is None or content == '':
                continue

            token = str(content)
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Heartbeat
            current_time = asyncio.get_event_loop().time()
            if current_time - last_heartbeat >= heartbeat_interval:
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                last_heartbeat = current_time

            await asyncio.sleep(0)

        if not full_response or not full_response.strip():
            yield f"data: {json.dumps({'type': 'error', 'error': 'Empty response generated'})}\n\n"
            return

        # Save to session history
        await add_to_session_history(actual_session_id, "user", query)
        await add_to_session_history(actual_session_id, "assistant", full_response)

        # Cache response
        cache_data = {
            "answer": full_response,
            "sources": sources,
            "session_id": actual_session_id,
        }
        await cache_response(query, cache_data, actual_session_id)
        logger.info(f"Cached streaming response for: {query[:50]}... (session={actual_session_id})")

        # Send done event with full response (sources will be sent separately after)
        yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

        # Send sources AFTER text streaming is complete
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'confidence': confidence})}\n\n"

    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


async def _stream_openai_fallback(
    query: str,
    session_id: str,
    max_retries: int = 2,
    heartbeat_interval: int = 15
) -> AsyncGenerator[str, None]:
    """
    Fallback streaming using OpenAI directly when RAG is unavailable.

    Args:
        query: The user query
        session_id: Session ID for conversation history
        max_retries: Number of retry attempts
        heartbeat_interval: Seconds between heartbeat events
    """
    try:
        client = get_openai_client()

        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'confidence': 0})}\n\n"

        # Build messages with session history
        messages = [
            {"role": "system", "content": "You are ARIA, a helpful AI citizen support assistant. Answer questions clearly and concisely. Remember information shared by the user in this conversation."}
        ]

        # Add session history
        try:
            history = await get_session_history(session_id)
            for msg in history:
                if msg and isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        except Exception as hist_error:
            logger.warning(f"Failed to get session history: {hist_error}")

        messages.append({"role": "user", "content": query})

        # Retry logic for OpenAI streaming
        stream = None
        for attempt in range(max_retries + 1):
            try:
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    stream=True,
                )
                break
            except Exception as stream_error:
                logger.warning(f"OpenAI stream attempt {attempt + 1} failed: {stream_error}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat(), 'retry': attempt + 1})}\n\n"
                else:
                    raise stream_error

        if stream is None:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to create OpenAI stream after retries'})}\n\n"
            return

        full_response = ""
        last_heartbeat = asyncio.get_event_loop().time()

        for chunk in stream:
            # Explicit null checks on chunk and its properties
            if chunk is None:
                continue
            if not hasattr(chunk, 'choices') or not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice is None:
                continue
            if not hasattr(choice, 'delta') or choice.delta is None:
                continue

            # Explicit null check on delta.content
            content = getattr(choice.delta, 'content', None)
            if content is None or content == '':
                continue

            token = str(content)
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Send heartbeat periodically
            current_time = asyncio.get_event_loop().time()
            if current_time - last_heartbeat >= heartbeat_interval:
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                last_heartbeat = current_time

            await asyncio.sleep(0)  # Allow other tasks to run

        # Check for empty response
        if not full_response or not full_response.strip():
            yield f"data: {json.dumps({'type': 'error', 'error': 'OpenAI returned empty response'})}\n\n"
            return

        # Save session history
        await add_to_session_history(session_id, "user", query)
        await add_to_session_history(session_id, "assistant", full_response)

        yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

    except Exception as e:
        logger.error(f"OpenAI fallback streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@router.post("/query/text/stream")
async def query_text_stream(request: StreamingQueryRequest):
    """
    Submit a text query and get a streaming response.

    Uses Server-Sent Events (SSE) to stream the response tokens.

    Event types:
    - session: Contains session_id
    - sources: Contains sources array and confidence
    - token: Contains individual response tokens
    - done: Indicates completion with full_response
    - error: Contains error message

    Args:
        request: Query request with question

    Returns:
        StreamingResponse with text/event-stream content
    """
    async def stream_with_stats():
        start_time = time.perf_counter()
        ttft = None
        cache_hit = False
        first_token_sent = False

        async for event in stream_query_response(request.query, request.session_id):
            yield event

            # Track TTFT on first token
            if not first_token_sent and '"type": "token"' in event:
                ttft = (time.perf_counter() - start_time) * 1000
                first_token_sent = True

            # Check if this was a cache hit (fast session event followed by tokens)
            if '"type": "session"' in event and ttft is None:
                # Will determine cache hit based on speed
                pass

            # Record stats when done
            if '"type": "done"' in event:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                # Consider it a cache hit if TTFT < 100ms (cached responses stream very fast)
                cache_hit = ttft is not None and ttft < 100
                asyncio.create_task(record_stat("/api/query/text/stream", elapsed_ms, cache_hit, ttft))

    return StreamingResponse(
        stream_with_stats(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/query/text/stream-with-audio")
async def query_text_stream_with_audio(request: StreamingQueryRequest):
    """Stream text response with streaming TTS audio generation.

    Uses Server-Sent Events (SSE) to stream text tokens, then streams
    TTS audio chunks for faster audio playback.

    Event types:
    - session: Contains session_id
    - token: Contains individual response tokens
    - done: Indicates text completion with full_response
    - sources: Contains sources array and confidence
    - audio_start: Indicates audio streaming is starting (includes sample_rate, channels, bits_per_sample)
    - audio_chunk: Contains base64-encoded audio chunk (PCM data)
    - audio_end: Indicates audio streaming is complete
    - error: Contains error message

    Args:
        request: Query request with question and optional session_id

    Returns:
        StreamingResponse with text/event-stream content including streaming audio
    """

    async def stream_with_audio():
        start_time = time.perf_counter()
        ttft = None
        cache_hit = False
        first_token_sent = False
        actual_session_id = request.session_id or str(uuid.uuid4())
        full_text = ""

        # First, yield all events from stream_query_response (text, done, sources)
        async for event in stream_query_response(request.query, request.session_id):
            yield event

            # Track TTFT on first token
            if not first_token_sent and '"type": "token"' in event:
                ttft = (time.perf_counter() - start_time) * 1000
                first_token_sent = True

            # Capture full_response from done event for TTS later
            if '"type": "done"' in event:
                try:
                    data = json.loads(event.replace("data: ", "").strip())
                    full_text = data.get("full_response", "")
                except Exception:
                    full_text = ""

        # Now all events have been yielded (including sources)
        # Start audio streaming AFTER sources have been sent
        if full_text:
            logger.info(f"Starting streaming TTS for response ({len(full_text)} chars)")

            # Send audio_start event with format info
            yield f"data: {json.dumps({'type': 'audio_start', 'sample_rate': 24000, 'channels': 1, 'bits_per_sample': 16})}\n\n"

            # Stream audio chunks
            all_chunks = []
            chunk_count = 0
            try:
                async for audio_chunk in generate_tts_openai_streaming(full_text):
                    if audio_chunk:
                        all_chunks.append(audio_chunk)
                        chunk_b64 = base64.b64encode(audio_chunk).decode()
                        yield f"data: {json.dumps({'type': 'audio_chunk', 'chunk': chunk_b64, 'index': chunk_count})}\n\n"
                        chunk_count += 1

                # Send audio_end event
                yield f"data: {json.dumps({'type': 'audio_end', 'total_chunks': chunk_count})}\n\n"
                logger.info(f"Streamed {chunk_count} audio chunks")

                # Cache complete audio for future requests
                if all_chunks:
                    pcm_data = b''.join(all_chunks)
                    wav_audio = add_wav_header(pcm_data, sample_rate=24000, channels=1, bits_per_sample=16)
                    audio_b64 = base64.b64encode(wav_audio).decode()
                    await cache_audio(request.query, audio_b64, actual_session_id)
                    logger.info(f"Cached complete audio ({len(audio_b64)} bytes base64)")

            except Exception as e:
                logger.error(f"Streaming TTS error: {e}")
                # Fallback to non-streaming TTS
                logger.info("Falling back to non-streaming TTS")
                audio_bytes = await generate_tts(full_text)
                if audio_bytes:
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    await cache_audio(request.query, audio_b64, actual_session_id)
                    yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_b64})}\n\n"

        # Record stats (total includes audio generation time)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if ttft is not None and ttft < 100:
            cache_hit = True
        asyncio.create_task(record_stat("/api/query/text/stream-with-audio", elapsed_ms, cache_hit, ttft))

    return StreamingResponse(
        stream_with_audio(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query/audio")
async def query_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    session_id: Optional[str] = Form(None, description="Session ID for follow-ups"),
    include_audio: bool = Form(True, description="Include TTS audio in response"),
    include_text: bool = Form(True, description="Include text answer in response"),
    response_format: str = Form("json", description="Response format: 'binary' for WAV audio, 'json' for JSON with base64"),
):
    """
    Upload an audio file, transcribe it, and get a response.

    Supports: WAV, MP3, M4A, OGG, FLAC, WEBM

    Args:
        file: Audio file upload
        session_id: Optional session ID for context
        include_audio: Whether to include TTS audio (default: True)
        include_text: Whether to include text answer (default: True)
        response_format: 'binary' returns WAV directly, 'json' returns JSON with base64

    Returns:
        If response_format='binary': WAV audio stream with metadata in headers
        If response_format='json': AudioQueryResponse with transcript, answer, sources, and optional audio
    """
    start_time = time.perf_counter()
    cache_hit = False

    # Validate file extension
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: {SUPPORTED_AUDIO_FORMATS}",
        )

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Transcribe audio
    transcript = await transcribe_audio(content, suffix)

    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not transcribe audio. Please speak clearly and try again.",
        )

    # Query knowledge base with transcript
    result = await query_knowledge_base(transcript, session_id)
    actual_session_id = result["session_id"]

    # Convert sources
    sources = [
        QuerySource(
            title=s.get("title", "Unknown"),
            url=s.get("url"),
            article_id=s.get("article_id"),
            score=s.get("score"),
        )
        for s in result.get("sources", [])
    ]

    # Generate TTS audio if requested (with caching)
    audio_bytes = None
    audio_b64 = None
    if include_audio:
        # Check audio cache first (session-aware, keyed by transcript)
        cached_audio = await get_cached_audio(transcript, actual_session_id)
        if cached_audio:
            audio_b64 = cached_audio
            audio_bytes = base64.b64decode(cached_audio)
            cache_hit = True
        else:
            # Generate TTS and cache it
            audio_bytes = await generate_tts(result["answer"])
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await cache_audio(transcript, audio_b64, actual_session_id)

    # Return binary WAV directly
    if response_format == "binary" and audio_bytes:
        import urllib.parse
        # Record stats before returning (track include_audio separately)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        endpoint = f"/api/query/audio?include_audio={include_audio}"
        asyncio.create_task(record_stat(endpoint, elapsed_ms, cache_hit))
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "X-Transcript": urllib.parse.quote(transcript[:500]),
                "X-Answer": urllib.parse.quote(result["answer"][:500]) if include_text else "",
                "X-Session-ID": actual_session_id,
                "Content-Disposition": "attachment; filename=response.wav",
            },
        )

    # Return JSON response
    response = AudioQueryResponse(
        query=transcript,
        transcript=transcript,
        answer=result["answer"] if include_text else "",
        sources=sources if include_text else [],
        session_id=actual_session_id,
    )

    if audio_b64:
        response.audio_base64 = audio_b64

    # Record stats (track include_audio separately)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    endpoint = f"/api/query/audio?include_audio={include_audio}"
    asyncio.create_task(record_stat(endpoint, elapsed_ms, cache_hit))

    return response


@router.post("/query/audio/stream")
async def query_audio_stream(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    session_id: Optional[str] = Form(None, description="Session ID for follow-ups"),
    include_text: bool = Form(True, description="Include text answer in headers"),
) -> StreamingResponse:
    """
    Upload audio and stream back the audio response.

    Returns audio/wav stream directly with transcript and answer in headers.
    Uses audio caching to avoid redundant TTS calls.

    Args:
        file: Audio file upload
        session_id: Optional session ID for context
        include_text: Whether to include text answer in headers (default: True)

    Returns:
        StreamingResponse with audio/wav content
    """
    import urllib.parse

    start_time = time.perf_counter()
    cache_hit = False

    # Validate file extension
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: {SUPPORTED_AUDIO_FORMATS}",
        )

    # Read and transcribe
    content = await file.read()
    transcript = await transcribe_audio(content, suffix)

    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not transcribe audio",
        )

    # Query knowledge base
    result = await query_knowledge_base(transcript, session_id)
    actual_session_id = result["session_id"]

    # Check audio cache first (session-aware, keyed by transcript)
    audio_bytes = None
    cached_audio = await get_cached_audio(transcript, actual_session_id)
    if cached_audio:
        audio_bytes = base64.b64decode(cached_audio)
        cache_hit = True
    else:
        # Generate TTS and cache it
        audio_bytes = await generate_tts(result["answer"])
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await cache_audio(transcript, audio_b64, actual_session_id)

    if not audio_bytes:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable",
        )

    # Record stats
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    asyncio.create_task(record_stat("/api/query/audio/stream", elapsed_ms, cache_hit))

    # Stream response with metadata in headers
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "X-Transcript": urllib.parse.quote(transcript[:500]),
            "X-Answer": urllib.parse.quote(result["answer"][:500]) if include_text else "",
            "X-Session-ID": actual_session_id,
            "Content-Disposition": "attachment; filename=response.wav",
        },
    )
