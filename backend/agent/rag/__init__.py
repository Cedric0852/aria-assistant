"""
RAG (Retrieval-Augmented Generation) Pipeline for the Citizen Support Assistant.

This module provides the complete RAG pipeline for loading, indexing, and querying
knowledge base documents to provide grounded, accurate responses to citizen queries.

Main Components:
- loaders: Document loading for various formats (JSON, PDF, DOCX, MD, TXT)
- indexer: Vector store indexing with OpenAI embeddings
- query_engine: RAG query with source tracking and confidence scoring
- prompts: System prompts with hallucination prevention

Example Usage:
    from agent.rag import (
        query_knowledge_base,
        create_index,
        refresh_index,
        UnifiedDocumentLoader,
    )

    # Create or refresh index
    await create_index()

    # Query the knowledge base
    result = await query_knowledge_base("How do I apply for a visa?")
    print(result.answer)
    print(result.sources)
"""

# Document loaders
from .loaders import (
    IremboJSONReader,
    UnifiedDocumentLoader,
    get_document_loader,
)

# Indexing functions
from .indexer import (
    create_index,
    load_index,
    refresh_index,
    add_documents,
    delete_document,
    get_index_stats,
    get_or_create_index,
    get_embedding_model,
    configure_settings,
    DEFAULT_KNOWLEDGE_DIR,
    DEFAULT_INDEX_DIR,
)

# Query engine
from .query_engine import (
    QueryResult,
    create_query_engine,
    query_knowledge_base,
    query_with_history,
    RAGQueryEngine,
    get_query_engine,
    get_llm,
    extract_sources,
    calculate_confidence,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_CUTOFF,
    DEFAULT_LLM_MODEL,
)

# Prompts
from .prompts import (
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
    FOLLOWUP_PROMPT_TEMPLATE,
    NO_INFORMATION_RESPONSE,
    CLARIFICATION_PROMPT,
    LOW_CONFIDENCE_PREFIX,
    SOURCE_CITATION_TEMPLATE,
    SYNTHESIS_PROMPT,
    format_qa_prompt,
    format_followup_prompt,
    format_clarification,
    format_sources,
)

__all__ = [
    # Loaders
    "IremboJSONReader",
    "UnifiedDocumentLoader",
    "get_document_loader",

    # Indexer
    "create_index",
    "load_index",
    "refresh_index",
    "add_documents",
    "delete_document",
    "get_index_stats",
    "get_or_create_index",
    "get_embedding_model",
    "configure_settings",
    "DEFAULT_KNOWLEDGE_DIR",
    "DEFAULT_INDEX_DIR",

    # Query Engine
    "QueryResult",
    "create_query_engine",
    "query_knowledge_base",
    "query_with_history",
    "RAGQueryEngine",
    "get_query_engine",
    "get_llm",
    "extract_sources",
    "calculate_confidence",
    "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_CUTOFF",
    "DEFAULT_LLM_MODEL",

    # Prompts
    "SYSTEM_PROMPT",
    "QA_PROMPT_TEMPLATE",
    "FOLLOWUP_PROMPT_TEMPLATE",
    "NO_INFORMATION_RESPONSE",
    "CLARIFICATION_PROMPT",
    "LOW_CONFIDENCE_PREFIX",
    "SOURCE_CITATION_TEMPLATE",
    "SYNTHESIS_PROMPT",
    "format_qa_prompt",
    "format_followup_prompt",
    "format_clarification",
    "format_sources",
]
