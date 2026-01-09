"""
RAG query engine for the Citizen Support Assistant.

This module provides the query functionality for the RAG pipeline,
including retrieval, response generation, and source tracking.

Now integrated with domain classifier to filter out off-topic queries
before RAG processing (Layer 0 of hallucination prevention).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI

from .indexer import get_or_create_index, load_index, DEFAULT_INDEX_DIR
from .prompts import (
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
    NO_INFORMATION_RESPONSE,
    LOW_CONFIDENCE_PREFIX,
    format_sources,
)
from .domain_classifier import (
    classify_query,
    ClassificationResult,
    QueryCategory,
    OFF_TOPIC_RESPONSE,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_CUTOFF = 0.5
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1


@dataclass
class QueryResult:
    """Result from a knowledge base query."""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str
    has_relevant_context: bool
    raw_response: Optional[Any] = None
    classification: Optional[ClassificationResult] = None
    was_filtered: bool = False  # True if query was filtered by domain classifier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "query": self.query,
            "has_relevant_context": self.has_relevant_context,
            "was_filtered": self.was_filtered,
        }
        if self.classification:
            result["classification"] = {
                "category": self.classification.category.value,
                "reasoning": self.classification.reasoning,
            }
        return result


def get_llm(
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
) -> OpenAI:
    """
    Get the OpenAI LLM for response generation.

    Args:
        model: The model name to use
        temperature: Temperature for generation (lower = more deterministic)
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Configured OpenAI LLM instance
    """
    kwargs = {
        "model": model,
        "temperature": temperature,
        "system_prompt": SYSTEM_PROMPT,
    }
    if api_key:
        kwargs["api_key"] = api_key

    return OpenAI(**kwargs)


def create_query_engine(
    index: VectorStoreIndex,
    top_k: int = DEFAULT_TOP_K,
    similarity_cutoff: Optional[float] = None,  # Disabled by default - let LLM decide relevance
    llm: Optional[OpenAI] = None,
    streaming: bool = False,
) -> RetrieverQueryEngine:
    """
    Create a query engine from a vector store index.

    Args:
        index: The VectorStoreIndex to query
        top_k: Number of top similar documents to retrieve
        similarity_cutoff: Minimum similarity score threshold (None = no filtering)
        llm: The LLM to use for response generation
        streaming: Whether to enable streaming responses

    Returns:
        Configured RetrieverQueryEngine
    """
    # Configure LLM
    if llm is None:
        llm = get_llm()
    Settings.llm = llm

    # Create retriever with top-k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # Only add similarity filter if explicitly requested
    # Without filter, LLM gets all retrieved docs and decides relevance itself
    node_postprocessors = []
    if similarity_cutoff is not None and similarity_cutoff > 0:
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=similarity_cutoff,
        )
        node_postprocessors.append(postprocessor)

    # Create response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",  # Compact mode synthesizes responses efficiently
        streaming=streaming,
    )

    # Build query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        response_synthesizer=response_synthesizer,
    )

    return query_engine


def extract_sources(response) -> List[Dict[str, Any]]:
    """
    Extract source information from a query response.

    Args:
        response: The response from the query engine

    Returns:
        List of source dictionaries with metadata
    """
    sources = []
    seen_ids = set()

    if not hasattr(response, "source_nodes"):
        return sources

    for node in response.source_nodes:
        # Get node metadata
        metadata = node.node.metadata or {}
        doc_id = metadata.get("article_id") or node.node.doc_id or ""

        # Skip duplicates
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        source = {
            "doc_id": doc_id,
            "title": metadata.get("title", "Unknown"),
            "url": metadata.get("url", ""),
            "score": round(node.score, 4) if node.score else 0.0,
            "source_file": metadata.get("source_file", ""),
            "file_type": metadata.get("file_type", ""),
            "excerpt": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
        }
        sources.append(source)

    return sources


def calculate_confidence(sources: List[Dict[str, Any]]) -> float:
    """
    Calculate overall confidence based on source scores.

    Args:
        sources: List of sources with scores

    Returns:
        Confidence score between 0 and 1
    """
    if not sources:
        return 0.0

    # Use the highest score as the primary indicator
    max_score = max(s.get("score", 0) for s in sources)

    # Average of top scores as secondary
    top_scores = sorted([s.get("score", 0) for s in sources], reverse=True)[:3]
    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

    # Weighted combination
    confidence = 0.7 * max_score + 0.3 * avg_score

    return round(min(1.0, max(0.0, confidence)), 4)


async def query_knowledge_base(
    query: str,
    session: Optional[Any] = None,
    index_dir: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
    include_sources: bool = True,
    skip_classification: bool = False,
) -> QueryResult:
    """
    Query the knowledge base and return a grounded response.

    This is the main query function that:
    1. Classifies the query using domain classifier (Layer 0)
    2. Returns direct response for greetings/small_talk/off_topic
    3. Retrieves relevant documents from the vector store for Irembo queries
    4. Generates a response using the LLM
    5. Tracks sources and confidence

    Args:
        query: The user's question
        session: Optional session for conversation history
        index_dir: Optional path to the index directory
        top_k: Number of documents to retrieve
        similarity_cutoff: Minimum similarity threshold
        include_sources: Whether to include source citations in response
        skip_classification: Skip domain classification (for internal calls)

    Returns:
        QueryResult with answer, sources, and confidence
    """
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    # ============================================
    # LAYER 0: Domain Classification
    # ============================================
    classification = None
    if not skip_classification:
        logger.info(f"[QUERY_ENGINE] Classifying query: '{query[:50]}...'")
        classification = await classify_query(query)
        logger.info(
            f"[QUERY_ENGINE] Classification result: {classification.category.value} "
            f"(should_use_rag={classification.should_use_rag})"
        )

        # Handle non-RAG categories immediately
        if not classification.should_use_rag:
            # For greetings and small_talk, return the direct response
            if classification.category in (QueryCategory.GREETING, QueryCategory.SMALL_TALK):
                return QueryResult(
                    answer=classification.direct_response or "Hello! How can I help you with Irembo services?",
                    sources=[],
                    confidence=1.0,
                    query=query,
                    has_relevant_context=False,
                    classification=classification,
                    was_filtered=True,
                )

            # For off-topic queries, return the polite redirect
            if classification.category == QueryCategory.OFF_TOPIC:
                logger.info(f"[QUERY_ENGINE] Filtering off-topic query: '{query[:50]}...'")
                return QueryResult(
                    answer=OFF_TOPIC_RESPONSE,
                    sources=[],
                    confidence=1.0,
                    query=query,
                    has_relevant_context=False,
                    classification=classification,
                    was_filtered=True,
                )

    # ============================================
    # LAYER 1+: RAG Pipeline (Irembo queries only)
    # ============================================
    logger.info(f"[QUERY_ENGINE] Processing Irembo query through RAG: '{query[:50]}...'")

    # Load or create index
    try:
        index = await get_or_create_index(index_dir=index_dir)
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return QueryResult(
            answer="I'm having trouble accessing my knowledge base. Please try again later.",
            sources=[],
            confidence=0.0,
            query=query,
            has_relevant_context=False,
            classification=classification,
        )

    # Create query engine
    query_engine = create_query_engine(
        index=index,
        top_k=top_k,
        similarity_cutoff=similarity_cutoff,
    )

    # Execute query
    try:
        response = query_engine.query(query)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResult(
            answer="I encountered an error while searching for information. Please try again.",
            sources=[],
            confidence=0.0,
            query=query,
            has_relevant_context=False,
            classification=classification,
        )

    # Extract sources
    sources = extract_sources(response)

    # Calculate confidence
    confidence = calculate_confidence(sources)

    # Check if we have relevant context
    has_relevant_context = len(sources) > 0 and confidence >= similarity_cutoff

    # Get the response text
    answer = str(response)

    # Handle low confidence or no relevant context
    if not has_relevant_context:
        answer = NO_INFORMATION_RESPONSE
    elif confidence < 0.7:
        answer = LOW_CONFIDENCE_PREFIX + answer

    # Append source citations if requested
    if include_sources and sources:
        answer += format_sources(sources)

    return QueryResult(
        answer=answer,
        sources=sources,
        confidence=confidence,
        query=query,
        has_relevant_context=has_relevant_context,
        raw_response=response,
        classification=classification,
    )


async def query_with_history(
    query: str,
    chat_history: List[Dict[str, str]],
    index_dir: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
) -> QueryResult:
    """
    Query with conversation history for follow-up questions.

    Args:
        query: The current user question
        chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        index_dir: Optional path to the index directory
        top_k: Number of documents to retrieve

    Returns:
        QueryResult with answer, sources, and confidence
    """
    # Format history for context
    history_text = ""
    for msg in chat_history[-5:]:  # Last 5 messages for context
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text += f"{role.capitalize()}: {content}\n"

    # Enhance query with context if it seems like a follow-up
    followup_indicators = ["it", "that", "this", "they", "them", "more", "also", "what about"]
    is_followup = any(query.lower().startswith(ind) for ind in followup_indicators)

    if is_followup and history_text:
        # Combine context with query
        enhanced_query = f"Context:\n{history_text}\nCurrent question: {query}"
    else:
        enhanced_query = query

    # Use standard query
    result = await query_knowledge_base(
        query=enhanced_query,
        index_dir=index_dir,
        top_k=top_k,
    )

    # Store original query in result
    result.query = query

    return result


class RAGQueryEngine:
    """
    Stateful query engine wrapper for session-based interactions.

    This class maintains state between queries and handles
    session-based conversation history.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        top_k: int = DEFAULT_TOP_K,
        similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
    ):
        """
        Initialize the RAG query engine.

        Args:
            index_dir: Path to the index directory
            top_k: Default number of documents to retrieve
            similarity_cutoff: Default similarity threshold
        """
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.top_k = top_k
        self.similarity_cutoff = similarity_cutoff
        self._index: Optional[VectorStoreIndex] = None
        self._query_engine: Optional[RetrieverQueryEngine] = None

    async def initialize(self) -> None:
        """Initialize the index and query engine."""
        self._index = await get_or_create_index(index_dir=self.index_dir)
        self._query_engine = create_query_engine(
            index=self._index,
            top_k=self.top_k,
            similarity_cutoff=self.similarity_cutoff,
        )

    async def query(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Execute a query.

        Args:
            query: The user's question
            session_id: Optional session ID for tracking

        Returns:
            QueryResult with answer, sources, and confidence
        """
        if self._index is None:
            await self.initialize()

        return await query_knowledge_base(
            query=query,
            index_dir=self.index_dir,
            top_k=self.top_k,
            similarity_cutoff=self.similarity_cutoff,
        )

    async def refresh(self) -> None:
        """Refresh the index to pick up new documents."""
        from .indexer import refresh_index

        self._index = await refresh_index(index_dir=self.index_dir)
        self._query_engine = create_query_engine(
            index=self._index,
            top_k=self.top_k,
            similarity_cutoff=self.similarity_cutoff,
        )


# Module-level singleton for convenience
_engine: Optional[RAGQueryEngine] = None


async def get_query_engine(
    index_dir: Optional[Path] = None,
) -> RAGQueryEngine:
    """
    Get or create a singleton query engine instance.

    Args:
        index_dir: Optional path to the index directory

    Returns:
        Configured RAGQueryEngine instance
    """
    global _engine

    if _engine is None:
        _engine = RAGQueryEngine(index_dir=index_dir)
        await _engine.initialize()

    return _engine
