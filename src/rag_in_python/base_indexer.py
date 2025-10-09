"""
Base indexer functionality shared between different indexer types.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension
DEFAULT_BATCH_SIZE = 100


class BaseIndexer(ABC):
    """
    Base class for document indexers with common FAISS vector store operations.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_store_path: Path | None = None,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        """
        Initialize the base indexer.

        Args:
            embedding_model: Embedding model to use for document encoding.
            vector_store_path: Path to persist/load FAISS vector store.
            dimension: Vector dimension for FAISS index.
        """
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        self.dimension = dimension

    def create_vector_store(self) -> FaissVectorStore:
        """
        Create a new FAISS vector store.

        Returns:
            Initialized FAISS vector store.
        """
        faiss_index = faiss.IndexFlatIP(self.dimension)
        return FaissVectorStore(faiss_index=faiss_index)

    def index_documents(
        self,
        documents: list[Document],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> VectorStoreIndex:
        """
        Index documents into FAISS vector store with batching.

        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.

        Returns:
            Vector store index.
        """
        vector_store = self.create_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info(f"Creating index for {len(documents)} documents...")

        if len(documents) > batch_size:
            # Create index with first batch
            first_batch = documents[:batch_size]
            index = VectorStoreIndex.from_documents(
                first_batch,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True,
            )

            # Add remaining documents in batches
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 2}: documents {i}-{min(i+batch_size, len(documents))}"
                )
                for doc in batch:
                    index.insert(doc)
        else:
            # Create index with all documents at once
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True,
            )

        logger.info("Document indexing completed.")
        return index

    def save_index(self, index: VectorStoreIndex, path: Path) -> None:
        """
        Save vector index to disk.

        Args:
            index: Vector store index to save.
            path: Path to save the index.
        """
        path.mkdir(parents=True, exist_ok=True)

        vector_store = index.vector_store
        if isinstance(vector_store, FaissVectorStore):
            faiss.write_index(vector_store._faiss_index, str(path / "index.faiss"))

        index.storage_context.persist(persist_dir=str(path))
        logger.info(f"Index saved to {path}")

    def load_index(self, path: Path) -> BaseIndex[Any]:
        """
        Load vector index from disk.

        Args:
            path: Path to load the index from.

        Returns:
            Loaded vector store index.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index path does not exist: {path}")

        faiss_index = faiss.read_index(str(path / "index.faiss"))
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(path),
        )

        from llama_index.core import load_index_from_storage

        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.embedding_model,
        )

        logger.info(f"Index loaded from {path}")
        return index

    @abstractmethod
    def load_documents(self, source: Any) -> list[Document]:
        """
        Abstract method for loading documents. Must be implemented by subclasses.

        Args:
            source: Source to load documents from (file paths, JSON files, etc.)

        Returns:
            List of loaded documents.
        """
        pass
