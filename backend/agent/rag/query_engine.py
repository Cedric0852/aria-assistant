"""
RAG query engine for the Citizen Support Assistant.

This module provides the query functionality for the RAG pipeline,
including retrieval, response generation, and source tracking.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI

from .indexer import get_or_create_index, DEFAULT_INDEX_DIR
from .prompts import (
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
    REFINE_PROMPT_TEMPLATE,
    NO_INFORMATION_RESPONSE,
    LOW_CONFIDENCE_PREFIX,
    format_sources,
)
from .domain_classifier import (
    classify_query,
    QueryCategory,
    OFF_TOPIC_RESPONSE,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_CUTOFF = 0.6  # Minimum 40% relevance required
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "query": self.query,
            "has_relevant_context": self.has_relevant_context,
        }


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
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,  # Always filter by default
    llm: Optional[OpenAI] = None,
    streaming: bool = False,
) -> RetrieverQueryEngine:
    """
    Create a query engine from a vector store index.

    Args:
        index: The VectorStoreIndex to query
        top_k: Number of top similar documents to retrieve
        similarity_cutoff: Minimum similarity score threshold (default 0.4)
        llm: The LLM to use for response generation
        streaming: Whether to enable streaming responses

    Returns:
        Configured RetrieverQueryEngine
    """
    if llm is None:
        llm = get_llm()
    Settings.llm = llm

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # ALWAYS use SimilarityPostprocessor to filter low-relevance nodes
    # This prevents the LLM from seeing irrelevant context
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    ]

    # Use custom prompt templates to enforce RAG-only answers
    qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)
    refine_prompt = PromptTemplate(REFINE_PROMPT_TEMPLATE)

    # Use "refine" mode for more careful response synthesis
    # This iterates through contexts and refines the answer, reducing hallucination
    response_synthesizer = get_response_synthesizer(
        response_mode="refine",
        streaming=streaming,
        text_qa_template=qa_prompt,
        refine_template=refine_prompt,
    )

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


def is_off_topic_query(query: str) -> bool:
    """
    Pre-retrieval domain classifier to detect obviously off-topic queries.
    This runs BEFORE vector search to save compute and prevent hallucination.

    Returns True if the query is clearly NOT about Irembo government services.
    """
    query_lower = query.lower().strip()

    # Off-topic keywords that indicate non-Irembo queries
    off_topic_patterns = [
        # Cooking/Food
        "cook", "recipe", "ingredient", "bake", "fry", "boil", "food", "meal",
        "rice", "chicken", "beef", "vegetable", "soup", "cake", "bread",
        # Beauty/Cosmetics
        "hair", "dye", "makeup", "cosmetic", "skincare", "nail", "salon",
        "shampoo", "haircut", "hairstyle", "beauty",
        # Math/Coding
        "calculate", "math", "equation", "code", "program", "python", "javascript",
        "algorithm", "function", "variable", "loop", "array",
        # General knowledge
        "capital of", "president of", "history of", "weather", "news",
        "movie", "song", "game", "sport", "football", "basketball",
        # Health (not government health services)
        "symptom", "medicine", "cure", "disease", "doctor", "hospital",
        # Other off-topic
        "joke", "story", "poem", "write me", "translate", "summarize",
        "how to make", "how do i", "what is the best", "recommend",
    ]

    # Check if query contains off-topic patterns
    for pattern in off_topic_patterns:
        if pattern in query_lower:
            # But exclude if it also contains Irembo-related terms
            irembo_terms = ["passport", "visa", "permit", "certificate", "license",
                          "registration", "irembo", "government", "apply", "document",
                          "fee", "requirement", "rwanda", "birth", "marriage", "death",
                          "driving", "business", "tax", "id", "national"]
            if not any(term in query_lower for term in irembo_terms):
                logger.info(f"Query '{query}' classified as off-topic (pattern: {pattern})")
                return True

    return False


async def query_knowledge_base(
    query: str,
    session: Optional[Any] = None,
    index_dir: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
    include_sources: bool = True,
) -> QueryResult:
    """
    Query the knowledge base and return a grounded response.

    This is the main query function that:
    1. LAYER 0: Pydantic AI classifier (gpt-5-nano reasoning) for domain classification
    2. LAYER 1-4: RAG pipeline with hallucination prevention
    3. Retrieves relevant documents from the vector store
    4. Generates a response using the LLM
    5. Tracks sources and confidence

    Args:
        query: The user's question
        session: Optional session for conversation history
        index_dir: Optional path to the index directory
        top_k: Number of documents to retrieve
        similarity_cutoff: Minimum similarity threshold
        include_sources: Whether to include source citations in response

    Returns:
        QueryResult with answer, sources, and confidence
    """
    # ==========================================================================
    # LAYER 0: Pydantic AI Domain Classifier (gpt-5-nano with reasoning)
    # ==========================================================================
    # Uses intelligent reasoning to classify queries before RAG retrieval
    logger.info(f"[QUERY_ENGINE] Processing query: '{query[:80]}'")

    classification = await classify_query(query)
    logger.info(
        f"[QUERY_ENGINE] Classification result: {classification.category.value} "
        f"(reasoning: {classification.reasoning[:50]}...)"
    )

    # Check if classifier failed (reasoning starts with "Classification failed")
    classifier_failed = classification.reasoning.startswith("Classification failed")

    if classifier_failed:
        logger.warning(f"[QUERY_ENGINE] Classifier failed, using fallback pattern matching")
        # Fallback to simple pattern matching if classifier fails
        if is_off_topic_query(query):
            logger.info(f"[QUERY_ENGINE] Query '{query}' rejected by fallback pattern classifier")
            return QueryResult(
                answer=NO_INFORMATION_RESPONSE,
                sources=[],
                confidence=0.0,
                query=query,
                has_relevant_context=False,
            )
        else:
            logger.info(f"[QUERY_ENGINE] Query passed fallback check, continuing to RAG")
    else:
        # Handle greetings - direct response, no RAG needed
        if classification.category == QueryCategory.GREETING:
            logger.info(f"[QUERY_ENGINE] Greeting detected, returning direct response")
            return QueryResult(
                answer=classification.direct_response or "Hello! How can I help you with Irembo services?",
                sources=[],
                confidence=1.0,
                query=query,
                has_relevant_context=True,  # Greeting is always relevant
            )

        # Handle small talk - direct response, no RAG needed
        if classification.category == QueryCategory.SMALL_TALK:
            logger.info(f"[QUERY_ENGINE] Small talk detected, returning direct response")
            return QueryResult(
                answer=classification.direct_response or "I'm here to help with Irembo government services!",
                sources=[],
                confidence=1.0,
                query=query,
                has_relevant_context=True,
            )

        # Handle off-topic queries - polite decline
        if classification.category == QueryCategory.OFF_TOPIC:
            logger.info(f"[QUERY_ENGINE] Query '{query}' rejected by Pydantic AI classifier (off-topic)")
            return QueryResult(
                answer=OFF_TOPIC_RESPONSE,
                sources=[],
                confidence=0.0,
                query=query,
                has_relevant_context=False,
            )

        # If category is IREMBO_SERVICE, continue to RAG pipeline below
        logger.info(f"[QUERY_ENGINE] Irembo service query, continuing to RAG pipeline")

    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

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
        )

    # Extract sources
    sources = extract_sources(response)

    # Calculate confidence
    confidence = calculate_confidence(sources)

    # STRICT RELEVANCE CHECK: Check the maximum individual source score
    # If the best matching document is below the threshold, don't use LLM answer
    max_source_score = max((s.get("score", 0) for s in sources), default=0)

    # Check if we have relevant context - require max score >= threshold (40%)
    has_relevant_context = len(sources) > 0 and max_source_score >= similarity_cutoff

    # Get the response text
    answer = str(response)

    # Handle low confidence or no relevant context - STRICTLY return NO_INFORMATION_RESPONSE
    if not has_relevant_context:
        logger.info(f"Query '{query}' rejected: max_score={max_source_score}, threshold={similarity_cutoff}")
        answer = NO_INFORMATION_RESPONSE
    elif max_source_score < 0.6:
        # Medium confidence - still answer but with prefix
        answer = LOW_CONFIDENCE_PREFIX + answer

    # Only append source citations if we have relevant context
    if include_sources and sources and has_relevant_context:
        answer += format_sources(sources)

    return QueryResult(
        answer=answer,
        sources=sources if has_relevant_context else [],  # Don't return misleading sources
        confidence=max_source_score,  # Use max score as confidence metric
        query=query,
        has_relevant_context=has_relevant_context,
        raw_response=response,
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
