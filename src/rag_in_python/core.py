"""
Core RAG system implementation using LlamaIndex and FAISS.
"""

import logging
from pathlib import Path
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .generation import ResponseGenerator
from .indexing import DocumentIndexer
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Main RAG system orchestrating indexing, retrieval, and generation.

    This class provides a high-level interface for building RAG applications
    with LlamaIndex and FAISS vector storage.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        embedding_model: BaseEmbedding | None = None,
        vector_store_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the RAG system.

        Args:
            llm: Language model for generation. Defaults to OpenAI "gpt-4".
            embedding_model: Embedding model. Defaults to OpenAI text-embedding-ada-002.
            vector_store_path: Path to persist/load FAISS vector store.
            **kwargs: Additional configuration options.
        """
        self.llm = llm or OpenAI(model="gpt-4")
        self.embedding_model = embedding_model or OpenAIEmbedding()
        self.vector_store_path = vector_store_path

        # Initialize components
        self.indexer = DocumentIndexer(
            embedding_model=self.embedding_model,
            vector_store_path=vector_store_path,
        )
        self.retriever = HybridRetriever()
        self.generator = ResponseGenerator(llm=self.llm)

        # Will be set after indexing documents
        self.index: BaseIndex[Any] | None = None

    def index_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> None:
        """
        Index documents into the vector store.

        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.
        """
        logger.info(f"Indexing {len(documents)} documents...")
        self.index = self.indexer.index_documents(documents, batch_size=batch_size)
        logger.info("Document indexing completed.")

    def load_and_index_files(
        self,
        file_paths: list[Path],
        batch_size: int = 100,
    ) -> None:
        """
        Load documents from files and index them.

        Args:
            file_paths: List of file paths to load and index.
            batch_size: Number of documents to process in each batch.
        """
        documents = self.indexer.load_documents(file_paths)
        self.index_documents(documents, batch_size=batch_size)

    def query(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve.
            similarity_threshold: Minimum similarity threshold for retrieval.
            **kwargs: Additional query options.

        Returns:
            Dictionary containing the response and metadata.
        """
        if not self.index:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # Retrieve relevant documents
        assert isinstance(self.index, VectorStoreIndex)
        retrieved_docs = self.retriever.retrieve(
            query=query,
            index=self.index,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Generate response
        response = self.generator.generate(
            query=query,
            context_docs=retrieved_docs,
            **kwargs,
        )

        return {
            "response": response,
            "retrieved_documents": len(retrieved_docs),
            "sources": [doc.metadata for doc in retrieved_docs if doc.metadata],
        }

    def save_index(self, path: Path | None = None) -> None:
        """
        Save the vector index to disk.

        Args:
            path: Path to save the index. Uses default path if not provided.
        """
        if not self.index:
            raise ValueError("No index to save. Index documents first.")

        save_path = path or self.vector_store_path
        if not save_path:
            raise ValueError("No save path specified.")

        assert isinstance(self.index, VectorStoreIndex)
        self.indexer.save_index(self.index, save_path)
        logger.info(f"Index saved to {save_path}")

    def load_index(self, path: Path | None = None) -> None:
        """
        Load a vector index from disk.

        Args:
            path: Path to load the index from. Uses default path if not provided.
        """
        load_path = path or self.vector_store_path
        if not load_path:
            raise ValueError("No load path specified.")

        self.index = self.indexer.load_index(load_path)
        logger.info(f"Index loaded from {load_path}")
