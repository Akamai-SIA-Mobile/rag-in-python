"""
JSON document indexing functionality for chunking by JSON entries.
This module handles JSON files by treating each top-level entry as a separate document chunk.
"""

import json
import logging
from pathlib import Path
from typing import Any

from llama_index.core import Document

from .base_indexer import BaseIndexer

logger = logging.getLogger(__name__)


class JSONDocumentIndexer(BaseIndexer):
    """
    Handles JSON document loading and indexing with FAISS vector store.
    Each top-level JSON entry becomes a separate document for indexing.
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store_path: Path | None = None,
        dimension: int = 1536,  # OpenAI embedding dimension
    ) -> None:
        """
        Initialize the JSON document indexer.

        Args:
            embedding_model: Embedding model to use for document encoding.
            vector_store_path: Path to persist/load FAISS vector store.
            dimension: Vector dimension for FAISS index.
        """
        super().__init__(embedding_model, vector_store_path, dimension)

    def load_documents(self, json_file_path: Path) -> list[Document]:
        """
        Implementation of abstract method to load JSON documents.

        Args:
            json_file_path: Path to JSON file to load.

        Returns:
            List of documents created from JSON entries.
        """
        return self.load_json_documents(json_file_path)

    def load_json_documents(self, json_file_path: Path) -> list[Document]:
        """
        Load JSON file and create documents from each top-level entry.

        Args:
            json_file_path: Path to the JSON file.

        Returns:
            List of documents, one per JSON entry.
        """
        if not json_file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")

        documents = []

        try:
            with open(json_file_path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError(
                    f"Expected JSON object at root level, got {type(data)}"
                )

            for entry_id, entry_data in data.items():
                # Create document text from the entry
                doc_text = self._create_document_text(entry_id, entry_data)

                # Create metadata
                metadata = {
                    "entry_id": entry_id,
                    "source_file": str(json_file_path),
                    "doc_type": "json_entry",
                }

                # Add additional metadata from the entry if available
                if isinstance(entry_data, dict):
                    if "title" in entry_data:
                        metadata["title"] = entry_data["title"]
                    if "description" in entry_data:
                        metadata["description"] = entry_data["description"]

                # Create LlamaIndex Document
                document = Document(
                    text=doc_text,
                    metadata=metadata,
                    doc_id=f"{json_file_path.stem}_{entry_id}",
                )

                documents.append(document)

            logger.info(f"Created {len(documents)} documents from {json_file_path}")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {json_file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file_path}: {e}")
            raise

        return documents

    def _create_document_text(self, entry_id: str, entry_data: Any) -> str:
        """
        Create readable text from a JSON entry for embedding and retrieval.

        Args:
            entry_id: The ID/key of the JSON entry.
            entry_data: The data associated with the entry.

        Returns:
            Formatted text representation of the entry.
        """
        text_parts = [f"Entry ID: {entry_id}"]

        if isinstance(entry_data, dict):
            # Handle structured entry data
            for key, value in entry_data.items():
                if key == "title":
                    text_parts.append(f"Title: {value}")
                elif key == "description":
                    text_parts.append(f"Description: {value}")
                elif key == "body":
                    # Clean HTML tags from body content if present
                    clean_body = self._clean_html(str(value))
                    text_parts.append(f"Content: {clean_body}")
                else:
                    # Handle other fields
                    text_parts.append(f"{key.title()}: {value}")
        else:
            # Handle simple value
            text_parts.append(f"Content: {entry_data}")

        return "\n\n".join(text_parts)

    def _clean_html(self, html_text: str) -> str:
        """
        Remove HTML tags and clean up text content.

        Args:
            html_text: HTML text to clean.

        Returns:
            Cleaned plain text.
        """
        import re

        # Remove HTML tags
        clean_text = re.sub(r"<[^>]+>", "", html_text)

        # Replace HTML entities
        clean_text = clean_text.replace("&amp;", "&")
        clean_text = clean_text.replace("&lt;", "<")
        clean_text = clean_text.replace("&gt;", ">")
        clean_text = clean_text.replace("&quot;", '"')
        clean_text = clean_text.replace("&#39;", "'")
        clean_text = clean_text.replace("&nbsp;", " ")

        # Clean up whitespace
        clean_text = re.sub(r"\s+", " ", clean_text)
        clean_text = clean_text.strip()

        return clean_text

    def process_json_file(self, json_file_path: Path, index_path: Path) -> Any:
        """
        Complete pipeline to process a JSON file and create an index.

        Args:
            json_file_path: Path to the JSON file to process.
            index_path: Path where to save the index.

        Returns:
            Created vector store index.
        """
        # Load documents from JSON
        documents = self.load_json_documents(json_file_path)

        # Create index
        index = self.index_documents(documents)

        # Save index
        self.save_index(index, index_path)

        return index
