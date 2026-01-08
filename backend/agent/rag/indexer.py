"""
Vector indexing functionality for the RAG pipeline.

This module handles the creation, loading, and updating of the vector store index
used for semantic search over the knowledge base documents.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import redis.asyncio as aioredis
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from .loaders import UnifiedDocumentLoader, get_document_loader

logger = logging.getLogger(__name__)

EMBEDDING_CACHE_TTL = 86400  # 24 hours

DEFAULT_KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge"
DEFAULT_INDEX_DIR = Path(__file__).parent.parent.parent / "storage" / "index"

_cached_index: Optional[VectorStoreIndex] = None
_cached_index_dir: Optional[Path] = None

_embedding_redis: Optional[aioredis.Redis] = None


async def get_embedding_redis() -> aioredis.Redis:
    """Get Redis connection for embedding cache."""
    global _embedding_redis
    if _embedding_redis is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _embedding_redis = aioredis.from_url(redis_url, decode_responses=True)
        logger.info("Connected to Redis for embedding cache")
    return _embedding_redis


class CachedOpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding wrapper with Redis caching.

    Caches embedding vectors in Redis to avoid redundant API calls.
    Cache key is based on MD5 hash of the text + model name.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        cache_ttl: int = EMBEDDING_CACHE_TTL,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._cache_ttl = cache_ttl
        self._base_embed = OpenAIEmbedding(model=model_name, api_key=api_key) if api_key else OpenAIEmbedding(model=model_name)
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embed:{self._model_name}:{text_hash}"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Sync version - no caching (LlamaIndex requires sync for query)."""
        return self._base_embed._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Sync version - no caching."""
        return self._base_embed._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async query embedding with Redis caching."""
        cache_key = self._get_cache_key(query)

        try:
            redis = await get_embedding_redis()
            cached = await redis.get(cache_key)
            if cached:
                self._cache_hits += 1
                logger.debug(f"Embedding cache HIT (hits={self._cache_hits})")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Embedding cache read failed: {e}")

        # Cache miss - generate embedding
        self._cache_misses += 1
        logger.debug(f"Embedding cache MISS (misses={self._cache_misses})")

        embedding = await self._base_embed._aget_query_embedding(query)

        # Cache the result
        try:
            redis = await get_embedding_redis()
            await redis.setex(cache_key, self._cache_ttl, json.dumps(embedding))
        except Exception as e:
            logger.warning(f"Embedding cache write failed: {e}")

        return embedding

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async text embedding with Redis caching."""
        cache_key = self._get_cache_key(text)

        try:
            redis = await get_embedding_redis()
            cached = await redis.get(cache_key)
            if cached:
                self._cache_hits += 1
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Embedding cache read failed: {e}")

        # Cache miss
        self._cache_misses += 1
        embedding = await self._base_embed._aget_text_embedding(text)

        try:
            redis = await get_embedding_redis()
            await redis.setex(cache_key, self._cache_ttl, json.dumps(embedding))
        except Exception as e:
            logger.warning(f"Embedding cache write failed: {e}")

        return embedding

    def get_cache_stats(self) -> dict:
        """Get cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%"
        }


def get_embedding_model(
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> BaseEmbedding:
    """
    Get the embedding model for vector indexing.

    Args:
        model_name: The OpenAI embedding model to use
        api_key: Optional API key (uses env var if not provided)
        use_cache: Whether to use Redis caching for embeddings (default: True)

    Returns:
        Configured embedding instance (cached or direct)
    """
    if use_cache:
        logger.info(f"Using cached embedding model: {model_name}")
        return CachedOpenAIEmbedding(model_name=model_name, api_key=api_key)

    # Non-cached version
    kwargs = {"model": model_name}
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbedding(**kwargs)


def configure_settings(
    embedding_model: Optional[BaseEmbedding] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    use_cached_embeddings: bool = True,
) -> None:
    """
    Configure global LlamaIndex settings.

    Args:
        embedding_model: The embedding model to use
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between chunks
        use_cached_embeddings: Whether to use Redis-cached embeddings (default: True)
    """
    if embedding_model:
        Settings.embed_model = embedding_model
    else:
        Settings.embed_model = get_embedding_model(use_cache=use_cached_embeddings)

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap


async def create_index(
    knowledge_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    loader: Optional[UnifiedDocumentLoader] = None,
    embedding_model: Optional[OpenAIEmbedding] = None,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """
    Create a new vector store index from documents in the knowledge directory.

    This function loads all supported documents, creates embeddings, and
    builds a searchable vector index. The index is persisted to disk.

    Args:
        knowledge_dir: Directory containing knowledge documents
        index_dir: Directory to persist the index
        loader: Document loader to use (creates default if not provided)
        embedding_model: Embedding model to use
        show_progress: Whether to show progress during indexing

    Returns:
        The created VectorStoreIndex
    """
    knowledge_dir = Path(knowledge_dir or DEFAULT_KNOWLEDGE_DIR)
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    # Ensure directories exist
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Configure settings
    configure_settings(embedding_model)

    # Load documents
    loader = loader or get_document_loader()
    logger.info(f"Loading documents from {knowledge_dir}")
    documents = loader.load_directory(knowledge_dir)

    if not documents:
        logger.warning(f"No documents found in {knowledge_dir}")
        # Create empty index
        documents = [Document(text="No documents loaded yet.", doc_id="empty")]

    logger.info(f"Creating index with {len(documents)} documents")

    # Create the index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=show_progress,
    )

    # Persist to disk
    index.storage_context.persist(persist_dir=str(index_dir))
    logger.info(f"Index persisted to {index_dir}")

    return index


def load_index(
    index_dir: Optional[Path] = None,
    embedding_model: Optional[OpenAIEmbedding] = None,
) -> Optional[VectorStoreIndex]:
    """
    Load a persisted index from storage.

    Args:
        index_dir: Directory containing the persisted index
        embedding_model: Embedding model to use (must match index creation)

    Returns:
        The loaded VectorStoreIndex, or None if not found
    """
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    if not index_dir.exists():
        logger.warning(f"Index directory does not exist: {index_dir}")
        return None

    # Check for required files
    docstore_path = index_dir / "docstore.json"
    if not docstore_path.exists():
        logger.warning(f"No index found at {index_dir}")
        return None

    # Configure settings
    configure_settings(embedding_model)

    try:
        logger.info(f"Loading index from {index_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context)
        logger.info("Index loaded successfully")
        return index
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return None


async def refresh_index(
    knowledge_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    loader: Optional[UnifiedDocumentLoader] = None,
    embedding_model: Optional[OpenAIEmbedding] = None,
) -> VectorStoreIndex:
    """
    Refresh the index with new or changed documents.

    This function uses doc_id to identify documents, only re-indexing
    documents that are new or have changed. Existing unchanged documents
    are preserved.

    Args:
        knowledge_dir: Directory containing knowledge documents
        index_dir: Directory containing the persisted index
        loader: Document loader to use
        embedding_model: Embedding model to use

    Returns:
        The refreshed VectorStoreIndex
    """
    global _cached_index, _cached_index_dir

    knowledge_dir = Path(knowledge_dir or DEFAULT_KNOWLEDGE_DIR)
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    # Clear cache before refresh
    clear_index_cache()

    # Configure settings
    configure_settings(embedding_model)

    # Try to load existing index
    existing_index = load_index(index_dir, embedding_model)

    if existing_index is None:
        # No existing index, create new one
        logger.info("No existing index found, creating new index")
        return await create_index(
            knowledge_dir=knowledge_dir,
            index_dir=index_dir,
            loader=loader,
            embedding_model=embedding_model,
        )

    # Load current documents
    loader = loader or get_document_loader()
    documents = loader.load_directory(knowledge_dir)

    if not documents:
        logger.warning("No documents found for refresh")
        return existing_index

    logger.info(f"Refreshing index with {len(documents)} documents")

    # Use refresh_ref_docs to update only changed documents
    # This method uses doc_id to identify documents
    try:
        existing_index.refresh_ref_docs(documents)

        # Persist updated index
        existing_index.storage_context.persist(persist_dir=str(index_dir))
        logger.info(f"Index refreshed and persisted to {index_dir}")

        # Cache the refreshed index
        _cached_index = existing_index
        _cached_index_dir = index_dir

    except Exception as e:
        logger.error(f"Failed to refresh index: {e}")
        # Fall back to full rebuild
        logger.info("Falling back to full index rebuild")
        index = await create_index(
            knowledge_dir=knowledge_dir,
            index_dir=index_dir,
            loader=loader,
            embedding_model=embedding_model,
        )
        # Cache the rebuilt index
        _cached_index = index
        _cached_index_dir = index_dir
        return index

    return existing_index


async def add_documents(
    documents: List[Document],
    index_dir: Optional[Path] = None,
    embedding_model: Optional[OpenAIEmbedding] = None,
) -> VectorStoreIndex:
    """
    Add new documents to an existing index.

    Args:
        documents: List of documents to add
        index_dir: Directory containing the persisted index
        embedding_model: Embedding model to use

    Returns:
        The updated VectorStoreIndex
    """
    global _cached_index, _cached_index_dir

    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    # Configure settings
    configure_settings(embedding_model)

    # Load existing index (use cache if available)
    index = _cached_index if _cached_index is not None and _cached_index_dir == index_dir else load_index(index_dir, embedding_model)

    if index is None:
        # Create new index with the documents
        logger.info("No existing index, creating new one with provided documents")
        index = VectorStoreIndex.from_documents(documents)
    else:
        # Insert new documents
        for doc in documents:
            index.insert(doc)

    # Persist
    index.storage_context.persist(persist_dir=str(index_dir))
    logger.info(f"Added {len(documents)} documents to index")

    # Update cache
    _cached_index = index
    _cached_index_dir = index_dir

    return index


async def delete_document(
    doc_id: str,
    index_dir: Optional[Path] = None,
    embedding_model: Optional[OpenAIEmbedding] = None,
) -> bool:
    """
    Delete a document from the index by its doc_id.

    Args:
        doc_id: The document ID to delete
        index_dir: Directory containing the persisted index
        embedding_model: Embedding model to use

    Returns:
        True if document was deleted, False otherwise
    """
    global _cached_index, _cached_index_dir

    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    # Configure settings
    configure_settings(embedding_model)

    # Load existing index (use cache if available)
    index = _cached_index if _cached_index is not None and _cached_index_dir == index_dir else load_index(index_dir, embedding_model)

    if index is None:
        logger.warning("No index found")
        return False

    try:
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
        index.storage_context.persist(persist_dir=str(index_dir))
        logger.info(f"Deleted document {doc_id} from index")

        # Update cache with modified index
        _cached_index = index
        _cached_index_dir = index_dir

        return True
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        return False


def get_index_stats(index_dir: Optional[Path] = None) -> dict:
    """
    Get statistics about the current index.

    Args:
        index_dir: Directory containing the persisted index

    Returns:
        Dictionary with index statistics
    """
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR)

    index = load_index(index_dir)

    if index is None:
        return {
            "exists": False,
            "document_count": 0,
            "index_dir": str(index_dir),
        }

    # Get document count from docstore
    try:
        docstore = index.storage_context.docstore
        doc_count = len(docstore.docs)
    except Exception:
        doc_count = 0

    return {
        "exists": True,
        "document_count": doc_count,
        "index_dir": str(index_dir),
    }


async def get_or_create_index(
    knowledge_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    embedding_model: Optional[OpenAIEmbedding] = None,
    force_reload: bool = False,
) -> VectorStoreIndex:
    """
    Get existing index or create a new one if it doesn't exist.

    This function caches the index in memory to avoid reloading from disk
    on every request. Use force_reload=True to bypass the cache.

    Args:
        knowledge_dir: Directory containing knowledge documents
        index_dir: Directory for the persisted index
        embedding_model: Embedding model to use
        force_reload: Force reload from disk even if cached

    Returns:
        The VectorStoreIndex (existing or newly created)
    """
    global _cached_index, _cached_index_dir

    # Resolve path to ensure consistent comparison
    index_dir = Path(index_dir or DEFAULT_INDEX_DIR).resolve()

    # Debug cache state
    logger.info(f"Index cache check: cached={_cached_index is not None}, cached_dir={_cached_index_dir}, requested_dir={index_dir}")

    # Return cached index if available and matches requested directory
    if not force_reload and _cached_index is not None and _cached_index_dir == index_dir:
        logger.info("Using cached index from memory (cache hit)")
        return _cached_index

    # Cache miss - explain why
    if _cached_index is None:
        logger.info("Cache miss: no cached index")
    elif _cached_index_dir != index_dir:
        logger.info(f"Cache miss: directory mismatch (cached={_cached_index_dir} vs requested={index_dir})")
    else:
        logger.info("Cache miss: force_reload requested")

    # Try to load existing from disk
    logger.info(f"Loading index from disk: {index_dir}")
    index = load_index(index_dir, embedding_model)

    if index is not None:
        # Cache the loaded index
        _cached_index = index
        _cached_index_dir = index_dir
        logger.info("Index loaded and cached in memory")
        return index

    # Create new
    index = await create_index(
        knowledge_dir=knowledge_dir,
        index_dir=index_dir,
        embedding_model=embedding_model,
    )

    # Cache the new index
    _cached_index = index
    _cached_index_dir = index_dir
    logger.info("New index created and cached in memory")

    return index


def clear_index_cache() -> None:
    """Clear the in-memory index cache. Call this after refreshing or modifying the index."""
    global _cached_index, _cached_index_dir
    _cached_index = None
    _cached_index_dir = None
    logger.info("Index cache cleared")


async def get_index_status() -> bool:
    """
    Check if a valid index exists (either cached or on disk).

    Returns:
        True if index is available, False otherwise
    """
    global _cached_index, _cached_index_dir

    # Check memory cache first
    if _cached_index is not None:
        return True

    # Check disk
    index_dir = DEFAULT_INDEX_DIR
    if not index_dir.exists():
        return False

    docstore_path = index_dir / "docstore.json"
    return docstore_path.exists()
