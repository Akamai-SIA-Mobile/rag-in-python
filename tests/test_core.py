"""Tests for the core RAG system."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from rag_in_python.core import RAGSystem
from llama_index.core import Document


class TestRAGSystem:
    """Test cases for RAGSystem class."""
    
    def test_initialization(self):
        """Test RAGSystem initialization with default parameters."""
        rag_system = RAGSystem()
        
        assert rag_system.llm is not None
        assert rag_system.embedding_model is not None
        assert rag_system.indexer is not None
        assert rag_system.retriever is not None
        assert rag_system.generator is not None
        assert rag_system.index is None
        
    def test_initialization_with_custom_path(self):
        """Test RAGSystem initialization with custom vector store path."""
        custom_path = Path("./custom_index")
        rag_system = RAGSystem(vector_store_path=custom_path)
        
        assert rag_system.vector_store_path == custom_path
        
    @patch('rag_in_python.core.DocumentIndexer')
    def test_index_documents(self, mock_indexer):
        """Test document indexing functionality."""
        # Mock the indexer
        mock_index = Mock()
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.return_value = mock_index
        mock_indexer.return_value = mock_indexer_instance
        
        rag_system = RAGSystem()
        documents = [Document(text="Test document 1"), Document(text="Test document 2")]
        
        rag_system.index_documents(documents)
        
        assert rag_system.index == mock_index
        mock_indexer_instance.index_documents.assert_called_once_with(documents, batch_size=100)
        
    def test_query_without_index_raises_error(self):
        """Test that querying without an index raises ValueError."""
        rag_system = RAGSystem()
        
        with pytest.raises(ValueError, match="No documents indexed"):
            rag_system.query("test query")
            
    @patch('rag_in_python.core.DocumentIndexer')
    @patch('rag_in_python.core.HybridRetriever')
    @patch('rag_in_python.core.ResponseGenerator')
    def test_query_with_index(self, mock_generator, mock_retriever, mock_indexer):
        """Test querying with a properly initialized index."""
        # Set up mocks
        mock_index = Mock()
        mock_indexer_instance = Mock()
        mock_indexer_instance.index_documents.return_value = mock_index
        mock_indexer.return_value = mock_indexer_instance
        
        mock_retrieved_docs = [Document(text="Retrieved doc")]
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = mock_retrieved_docs
        mock_retriever.return_value = mock_retriever_instance
        
        mock_response = "Generated response"
        mock_generator_instance = Mock()
        mock_generator_instance.generate.return_value = mock_response
        mock_generator.return_value = mock_generator_instance
        
        # Test the query
        rag_system = RAGSystem()
        rag_system.index_documents([Document(text="Test doc")])
        
        result = rag_system.query("test query")
        
        assert result["response"] == mock_response
        assert result["retrieved_documents"] == 1
        mock_retriever_instance.retrieve.assert_called_once()
        mock_generator_instance.generate.assert_called_once()


@pytest.mark.integration
class TestRAGSystemIntegration:
    """Integration tests for RAGSystem (require API keys)."""
    
    @pytest.mark.skip(reason="Requires OpenAI API key")
    def test_end_to_end_workflow(self):
        """Test complete workflow from indexing to querying."""
        # This would test the actual integration with OpenAI and FAISS
        # Skip by default as it requires API keys and external dependencies
        pass