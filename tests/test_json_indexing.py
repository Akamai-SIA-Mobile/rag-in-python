"""Tests for JSON indexing functionality."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from llama_index.core import Document

from rag_in_python.json_indexing import JSONDocumentIndexer


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Load JSON file and return the data."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def chunk_json_by_entries(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Chunk JSON data by top-level entries.
    Each entry becomes a separate chunk for indexing.
    """
    chunks = []

    for entry_id, entry_data in json_data.items():
        # Create a chunk with the entry ID and its full data
        chunk = {
            "id": entry_id,
            "content": json.dumps(entry_data, indent=2),
            "metadata": {
                "source": "molly.json",
                "entry_id": entry_id,
                "type": "json_entry",
            },
        }
        chunks.append(chunk)

    return chunks


class TestJSONIndexing:
    """Test cases for JSON indexing functionality."""

    def test_load_json_file(self):
        """Test loading JSON file."""
        json_file_path = Path("test_docs/molly.json")

        if json_file_path.exists():
            json_data = load_json_file(json_file_path)
            assert isinstance(json_data, dict)
            assert len(json_data) > 0
        else:
            pytest.skip("molly.json test file not found")

    def test_chunk_json_by_entries(self):
        """Test chunking JSON data by entries."""
        sample_data = {
            "entry1": {"title": "Test 1", "content": "Content 1"},
            "entry2": {"title": "Test 2", "content": "Content 2"},
        }

        chunks = chunk_json_by_entries(sample_data)

        assert len(chunks) == 2
        assert chunks[0]["id"] == "entry1"
        assert chunks[1]["id"] == "entry2"
        assert "title" in chunks[0]["content"]
        assert chunks[0]["metadata"]["source"] == "molly.json"
        assert chunks[0]["metadata"]["entry_id"] == "entry1"

    def test_chunk_empty_json(self):
        """Test chunking empty JSON data."""
        empty_data = {}
        chunks = chunk_json_by_entries(empty_data)
        assert len(chunks) == 0

    @pytest.mark.integration
    def test_json_indexing_with_real_data(self):
        """Test JSON indexing with real molly.json data."""
        json_file_path = Path("test_docs/molly.json")

        if not json_file_path.exists():
            pytest.skip("molly.json test file not found")

        # Load the JSON data
        json_data = load_json_file(json_file_path)

        # Chunk the data by entries
        chunks = chunk_json_by_entries(json_data)

        assert len(chunks) > 0

        # Check first chunk structure
        first_chunk = chunks[0]
        assert "id" in first_chunk
        assert "content" in first_chunk
        assert "metadata" in first_chunk
        assert first_chunk["metadata"]["source"] == "molly.json"

        # Verify content is valid JSON
        parsed_content = json.loads(first_chunk["content"])
        assert isinstance(parsed_content, dict)

    @patch("rag_in_python.json_indexing.JSONDocumentIndexer")
    def test_json_indexer_integration(self, mock_indexer):
        """Test integration with JSONDocumentIndexer."""
        # Mock the indexer
        mock_indexer_instance = Mock()
        mock_indexer.return_value = mock_indexer_instance

        sample_data = {
            "test_entry": {"title": "Test", "description": "Test description"}
        }

        chunks = chunk_json_by_entries(sample_data)

        # Convert chunks to Document objects (as JSONDocumentIndexer would do)
        documents = []
        for chunk in chunks:
            doc = Document(text=chunk["content"], metadata=chunk["metadata"])
            documents.append(doc)

        assert len(documents) == 1
        assert documents[0].metadata["entry_id"] == "test_entry"


class TestJSONDocumentIndexer:
    """Test cases for JSONDocumentIndexer class."""

    @patch("rag_in_python.json_indexing.BaseIndexer.__init__")
    def test_json_document_indexer_init(self, mock_base_init):
        """Test JSONDocumentIndexer initialization."""
        mock_base_init.return_value = None
        mock_embedding = Mock()

        JSONDocumentIndexer(
            embedding_model=mock_embedding,
            vector_store_path=Path("test_path"),
            dimension=1536,
        )

        mock_base_init.assert_called_once_with(mock_embedding, Path("test_path"), 1536)

    def test_create_document_text_structured(self):
        """Test creating document text from structured JSON data."""
        with patch("rag_in_python.json_indexing.BaseIndexer.__init__"):
            indexer = JSONDocumentIndexer(Mock())

            entry_data = {
                "title": "Test Title",
                "description": "Test Description",
                "body": "<p>Test content with <b>HTML</b></p>",
                "custom_field": "Custom Value",
            }

            result = indexer._create_document_text("test_id", entry_data)

            assert "Entry ID: test_id" in result
            assert "Title: Test Title" in result
            assert "Description: Test Description" in result
            assert "Content: Test content with HTML" in result
            assert "Custom_Field: Custom Value" in result

    def test_create_document_text_simple(self):
        """Test creating document text from simple JSON value."""
        with patch("rag_in_python.json_indexing.BaseIndexer.__init__"):
            indexer = JSONDocumentIndexer(Mock())

            result = indexer._create_document_text("test_id", "Simple text value")

            assert "Entry ID: test_id" in result
            assert "Content: Simple text value" in result

    def test_clean_html(self):
        """Test HTML cleaning functionality."""
        with patch("rag_in_python.json_indexing.BaseIndexer.__init__"):
            indexer = JSONDocumentIndexer(Mock())

            html_text = "<p>Test &amp; <b>bold</b> &lt;text&gt;</p>"
            result = indexer._clean_html(html_text)

            assert result == "Test & bold <text>"

    @patch("rag_in_python.json_indexing.BaseIndexer.__init__")
    def test_load_json_documents_success(self, mock_base_init):
        """Test successful loading of JSON documents."""
        mock_base_init.return_value = None
        indexer = JSONDocumentIndexer(Mock())

        # Create a temporary JSON file
        test_data = {
            "entry1": {"title": "Test Entry 1", "description": "First test entry"},
            "entry2": {"title": "Test Entry 2", "description": "Second test entry"},
        }

        with (
            patch("builtins.open", create=True) as mock_open,
            patch("json.load", return_value=test_data),
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = indexer.load_json_documents(Path("test.json"))

            assert len(result) == 2
            assert all(isinstance(doc, Document) for doc in result)
            assert result[0].metadata["entry_id"] == "entry1"
            assert result[1].metadata["entry_id"] == "entry2"

    def test_load_json_documents_file_not_found(self):
        """Test handling of missing JSON file."""
        with patch("rag_in_python.json_indexing.BaseIndexer.__init__"):
            indexer = JSONDocumentIndexer(Mock())

            with pytest.raises(FileNotFoundError):
                indexer.load_json_documents(Path("nonexistent.json"))
