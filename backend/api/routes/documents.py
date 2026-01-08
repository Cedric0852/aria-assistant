"""Document management endpoints for knowledge base."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import json
import hashlib
import os
import aiofiles
import aiofiles.os

router = APIRouter(prefix="/api", tags=["documents"])

KNOWLEDGE_DIR = Path(os.getenv(
    "KNOWLEDGE_DIR",
    Path(__file__).parent.parent.parent / "data" / "knowledge"
))

SUPPORTED_EXTENSIONS = {".json", ".md", ".txt", ".pdf", ".docx"}


class DocumentUploadResult(BaseModel):
    """Result of a document upload operation."""
    status: str
    doc_id: str
    filename: str
    file_type: str


class DocumentUploadError(BaseModel):
    """Error result for failed document upload."""
    filename: str
    error: str


class BatchUploadResponse(BaseModel):
    """Response for batch document upload."""
    uploaded: list[DocumentUploadResult]
    errors: list[DocumentUploadError]
    total_success: int
    total_failed: int


class DocumentInfo(BaseModel):
    """Information about a document in the knowledge base."""
    filename: str
    file_type: str
    size_bytes: int
    doc_id: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: list[DocumentInfo]
    total: int


class RefreshIndexResponse(BaseModel):
    """Response for index refresh operation."""
    status: str
    documents_indexed: int
    message: Optional[str] = None


def generate_doc_id(filename: str) -> str:
    """Generate a unique document ID from filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def extract_json_doc_id(content: bytes) -> Optional[str]:
    """Extract article_id from JSON content if available."""
    try:
        data = json.loads(content)
        return data.get("article_id")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


async def save_document(file: UploadFile) -> DocumentUploadResult:
    """
    Save an uploaded document to the knowledge directory.

    Args:
        file: Uploaded file

    Returns:
        DocumentUploadResult with upload details

    Raises:
        HTTPException for validation errors
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {list(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if suffix == ".json":
        try:
            data = json.loads(content)
            if "content_markdown" not in data and "content" not in data:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must have 'content_markdown' or 'content' field",
                )
            doc_id = data.get("article_id") or generate_doc_id(file.filename)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON: {str(e)}",
            )
    else:
        doc_id = generate_doc_id(file.filename)

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    # Save file with doc_id as prefix for uniqueness
    save_path = KNOWLEDGE_DIR / f"{doc_id}{suffix}"

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    return DocumentUploadResult(
        status="uploaded",
        doc_id=doc_id,
        filename=file.filename,
        file_type=suffix[1:],  # Remove leading dot
    )


@router.post("/documents/upload", response_model=DocumentUploadResult)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload"),
) -> DocumentUploadResult:
    """
    Upload a single document to the knowledge base.

    Supported formats:
    - .json - IremboGov JSON format (must have 'content_markdown' or 'content')
    - .md - Markdown files
    - .txt - Plain text files
    - .pdf - PDF documents
    - .docx - Word documents

    The document is saved to the knowledge directory. Use /documents/refresh
    to rebuild the vector index after uploading.

    Args:
        file: Document file to upload

    Returns:
        DocumentUploadResult with doc_id and upload status
    """
    return await save_document(file)


@router.post("/documents/upload-batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    files: list[UploadFile] = File(..., description="Documents to upload"),
) -> BatchUploadResponse:
    """
    Upload multiple documents at once.

    Processes each file independently - failures on individual files
    don't prevent other files from being uploaded.

    Args:
        files: List of document files to upload

    Returns:
        BatchUploadResponse with results and errors for each file
    """
    results: list[DocumentUploadResult] = []
    errors: list[DocumentUploadError] = []

    for file in files:
        try:
            result = await save_document(file)
            results.append(result)
        except HTTPException as e:
            errors.append(DocumentUploadError(
                filename=file.filename or "unknown",
                error=str(e.detail),
            ))
        except Exception as e:
            errors.append(DocumentUploadError(
                filename=file.filename or "unknown",
                error=str(e),
            ))

    return BatchUploadResponse(
        uploaded=results,
        errors=errors,
        total_success=len(results),
        total_failed=len(errors),
    )


@router.post("/documents/refresh", response_model=RefreshIndexResponse)
async def refresh_index() -> RefreshIndexResponse:
    """
    Rebuild the vector index with all documents in the knowledge directory.

    This will:
    1. Load all supported documents from the knowledge directory
    2. Parse and chunk the documents
    3. Generate embeddings using OpenAI
    4. Build/update the vector index
    5. Persist the index to disk

    Documents with the same doc_id will be updated rather than duplicated.

    Returns:
        RefreshIndexResponse with indexing results
    """
    try:
        # Try to import and run the actual indexer
        from agent.rag.indexer import refresh_index as do_refresh

        index = await do_refresh(KNOWLEDGE_DIR)

        # Get document count from the index
        try:
            docstore = index.storage_context.docstore
            count = len(docstore.docs)
        except Exception:
            count = 0

        return RefreshIndexResponse(
            status="index_refreshed",
            documents_indexed=count,
            message=f"Successfully indexed {count} documents",
        )
    except ImportError:
        # Indexer not yet implemented - count files as placeholder
        count = sum(
            1
            for ext in SUPPORTED_EXTENSIONS
            for _ in KNOWLEDGE_DIR.glob(f"*{ext}")
        )
        return RefreshIndexResponse(
            status="pending",
            documents_indexed=count,
            message=f"Found {count} documents. Indexer not yet implemented.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Index refresh failed: {str(e)}",
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """
    List all documents in the knowledge base.

    Returns:
        DocumentListResponse with list of documents and their metadata
    """
    documents: list[DocumentInfo] = []

    if not KNOWLEDGE_DIR.exists():
        return DocumentListResponse(documents=[], total=0)

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in KNOWLEDGE_DIR.glob(f"*{ext}"):
            if file_path.is_file():
                stat = file_path.stat()

                # Try to extract doc_id from filename (assuming doc_id prefix pattern)
                stem = file_path.stem
                doc_id = stem if len(stem) == 12 else None

                # For JSON files, try to get article_id
                if ext == ".json" and doc_id is None:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            doc_id = data.get("article_id")
                    except Exception:
                        pass

                documents.append(DocumentInfo(
                    filename=file_path.name,
                    file_type=ext[1:],
                    size_bytes=stat.st_size,
                    doc_id=doc_id,
                ))

    documents.sort(key=lambda d: d.filename)

    return DocumentListResponse(
        documents=documents,
        total=len(documents),
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """
    Delete a document from the knowledge base.

    Args:
        doc_id: Document ID to delete

    Returns:
        Deletion status
    """
    if not KNOWLEDGE_DIR.exists():
        raise HTTPException(status_code=404, detail="Knowledge directory not found")

    # Find file with matching doc_id prefix
    deleted = False
    for ext in SUPPORTED_EXTENSIONS:
        file_path = KNOWLEDGE_DIR / f"{doc_id}{ext}"
        if file_path.exists():
            await aiofiles.os.remove(file_path)
            deleted = True
            break

    if not deleted:
        # Also check for files where doc_id might be the article_id in JSON
        for json_file in KNOWLEDGE_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("article_id") == doc_id:
                        await aiofiles.os.remove(json_file)
                        deleted = True
                        break
            except Exception:
                continue

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    return {
        "status": "deleted",
        "doc_id": doc_id,
        "message": "Document deleted. Run /documents/refresh to update the index.",
    }
