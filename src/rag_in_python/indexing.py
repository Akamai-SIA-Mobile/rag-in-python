"""
Document indexing functionality using LlamaIndex and FAISS.
"""

import logging
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding

from .base_indexer import BaseIndexer

logger = logging.getLogger(__name__)


class DocumentIndexer(BaseIndexer):
    """
    Handles document loading and indexing with FAISS vector store.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_store_path: Path | None = None,
        dimension: int = 1536,  # OpenAI embedding dimension
    ) -> None:
        """
        Initialize the document indexer.

        Args:
            embedding_model: Embedding model to use for document encoding.
            vector_store_path: Path to persist/load FAISS vector store.
            dimension: Vector dimension for FAISS index.
        """
        super().__init__(embedding_model, vector_store_path, dimension)

    def load_documents(self, file_paths: list[Path]) -> list[Document]:
        """
        Load documents from file paths.

        Args:
            file_paths: List of file paths to load.

        Returns:
            List of loaded documents.
        """
        documents = []

        for file_path in file_paths:
            if file_path.is_file():
                try:
                    # Use SimpleDirectoryReader for single file
                    reader = SimpleDirectoryReader(input_files=[str(file_path)])
                    file_docs = reader.load_data()
                    documents.extend(file_docs)
                    logger.info(f"Loaded {len(file_docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            elif file_path.is_dir():
                try:
                    # Use SimpleDirectoryReader for directory
                    reader = SimpleDirectoryReader(str(file_path))
                    dir_docs = reader.load_data()
                    documents.extend(dir_docs)
                    logger.info(f"Loaded {len(dir_docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading directory {file_path}: {e}")
            else:
                logger.warning(f"Path does not exist: {file_path}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
