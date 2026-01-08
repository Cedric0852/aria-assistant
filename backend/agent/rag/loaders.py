"""
Document loaders for the RAG pipeline.

This module provides document loading capabilities for various file formats,
with a focus on the IremboGov JSON format that includes markdown content.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

logger = logging.getLogger(__name__)


class IremboJSONReader(BaseReader):
    """
    Custom reader for IremboGov JSON document format.

    Expected JSON structure:
    {
        "url": "https://support.irembo.gov.rw/...",
        "title": "How to Apply for a Visa",
        "content_markdown": "# How to Apply...",
        "content": "plain text version",
        "metadata": {
            "updated": "...",
            "related_articles": [...]
        },
        "article_id": "47001155415"
    }
    """

    def load_data(self, file_path: Path, **kwargs) -> List[Document]:
        """
        Load a single JSON document and convert to LlamaIndex Document.

        Args:
            file_path: Path to the JSON file

        Returns:
            List containing a single Document with extracted content and metadata
        """
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            raise ValueError(f"Invalid JSON file: {file_path}") from e
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

        # Prefer content_markdown, fall back to content
        content = data.get("content_markdown") or data.get("content", "")

        if not content:
            logger.warning(f"No content found in {file_path}")
            return []

        # Extract metadata
        raw_metadata = data.get("metadata", {})
        article_id = data.get("article_id", "")

        metadata = {
            "title": data.get("title", file_path.stem),
            "url": data.get("url", ""),
            "article_id": article_id,
            "source_file": str(file_path.absolute()),
            "file_type": "json",
            "updated": raw_metadata.get("updated", ""),
            "related_articles": raw_metadata.get("related_articles", []),
        }

        # Use article_id as doc_id for deduplication, fall back to file path
        doc_id = article_id if article_id else self._generate_doc_id(file_path)

        doc = Document(
            text=content,
            metadata=metadata,
            doc_id=doc_id,
        )

        return [doc]

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique doc_id from the file path."""
        return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()


class UnifiedDocumentLoader:
    """
    Unified document loader supporting multiple file formats.

    Supported formats:
    - .json: IremboGov JSON format (using IremboJSONReader)
    - .md: Markdown files
    - .txt: Plain text files
    - .pdf: PDF documents
    - .docx: Word documents (Office Open XML)
    - .doc: Legacy Word documents (best effort)
    """

    SUPPORTED_EXTENSIONS = {".json", ".md", ".txt", ".pdf", ".docx", ".doc"}

    def __init__(self):
        """Initialize the unified loader with format-specific readers."""
        self.json_reader = IremboJSONReader()
        self._pdf_reader = None
        self._docx_reader = None
        self._md_reader = None

    @property
    def pdf_reader(self):
        """Lazy-load PDF reader."""
        if self._pdf_reader is None:
            try:
                from llama_index.readers.file import PDFReader
                self._pdf_reader = PDFReader()
            except ImportError:
                logger.warning("PDFReader not available. Install with: pip install llama-index-readers-file pypdf")
                self._pdf_reader = False
        return self._pdf_reader

    @property
    def docx_reader(self):
        """Lazy-load DOCX reader."""
        if self._docx_reader is None:
            try:
                from llama_index.readers.file import DocxReader
                self._docx_reader = DocxReader()
            except ImportError:
                logger.warning("DocxReader not available. Install with: pip install llama-index-readers-file python-docx")
                self._docx_reader = False
        return self._docx_reader

    @property
    def md_reader(self):
        """Lazy-load Markdown reader."""
        if self._md_reader is None:
            try:
                from llama_index.readers.file import MarkdownReader
                self._md_reader = MarkdownReader()
            except ImportError:
                # Fall back to plain text reading for markdown
                logger.info("MarkdownReader not available, using plain text reading for .md files")
                self._md_reader = False
        return self._md_reader

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique doc_id from the file path using MD5 hash."""
        return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

    def _read_text_file(self, file_path: Path) -> str:
        """Read a text file with encoding detection."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        # Last resort: read with errors ignored
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file to load

        Returns:
            List of Document objects (usually one, may be more for PDFs)

        Raises:
            ValueError: If the file type is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )

        documents: List[Document] = []

        try:
            if suffix == ".json":
                return self.json_reader.load_data(file_path)

            elif suffix == ".pdf":
                if self.pdf_reader:
                    documents = self.pdf_reader.load_data(file_path)
                else:
                    logger.warning(f"PDF reader not available, skipping {file_path}")
                    return []

            elif suffix in {".docx", ".doc"}:
                if self.docx_reader:
                    documents = self.docx_reader.load_data(file_path)
                else:
                    # Try alternative docx reading with docx2txt
                    try:
                        import docx2txt
                        content = docx2txt.process(str(file_path))
                        documents = [Document(text=content)]
                    except ImportError:
                        logger.warning(f"No DOCX reader available, skipping {file_path}")
                        return []

            elif suffix == ".md":
                if self.md_reader:
                    documents = self.md_reader.load_data(file_path)
                else:
                    # Fall back to plain text
                    content = self._read_text_file(file_path)
                    documents = [Document(text=content)]

            elif suffix == ".txt":
                content = self._read_text_file(file_path)
                documents = [Document(text=content)]

            else:
                raise ValueError(f"Unhandled file type: {suffix}")

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

        # Add common metadata and doc_id to all documents
        for doc in documents:
            doc.metadata = doc.metadata or {}
            doc.metadata.update({
                "source_file": str(file_path.absolute()),
                "title": doc.metadata.get("title", file_path.stem),
                "file_type": suffix[1:],  # Remove the leading dot
            })

            # Set doc_id if not already set
            if not doc.doc_id:
                doc.doc_id = self._generate_doc_id(file_path)

        return documents

    def load_directory(
        self,
        dir_path: Path,
        recursive: bool = True,
        extensions: Optional[set] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            dir_path: Path to the directory to scan
            recursive: If True, scan subdirectories recursively
            extensions: Optional set of extensions to filter (default: all supported)

        Returns:
            List of all loaded Document objects
        """
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        extensions = extensions or self.SUPPORTED_EXTENSIONS
        documents: List[Document] = []

        glob_pattern = "**/*" if recursive else "*"

        for ext in extensions:
            for file_path in dir_path.glob(f"{glob_pattern}{ext}"):
                if file_path.is_file():
                    try:
                        file_docs = self.load_file(file_path)
                        documents.extend(file_docs)
                        logger.info(f"Loaded {len(file_docs)} document(s) from {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                        continue

        logger.info(f"Loaded {len(documents)} total documents from {dir_path}")
        return documents

    def load_files(self, file_paths: List[Path]) -> List[Document]:
        """
        Load a list of specific files.

        Args:
            file_paths: List of file paths to load

        Returns:
            List of all loaded Document objects
        """
        documents: List[Document] = []

        for file_path in file_paths:
            try:
                file_docs = self.load_file(Path(file_path))
                documents.extend(file_docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        return documents


def get_document_loader() -> UnifiedDocumentLoader:
    """Factory function to get a configured document loader."""
    return UnifiedDocumentLoader()
