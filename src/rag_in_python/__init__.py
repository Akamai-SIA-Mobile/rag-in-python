"""
RAG in Python - A comprehensive RAG toolkit built with LlamaIndex and FAISS.

This package provides tools and utilities for building Retrieval-Augmented Generation
systems using modern Python practices, LlamaIndex framework, and FAISS vector storage.
"""

__version__ = "0.1.0"
__author__ = "SIA Mobile AI Ambassador Team"
__email__ = "zcui@akamai.com"

from .base_indexer import BaseIndexer
from .core import RAGSystem
from .generation import ResponseGenerator
from .indexing import DocumentIndexer
from .json_indexing import JSONDocumentIndexer
from .retrieval import HybridRetriever

__all__ = [
    "BaseIndexer",
    "DocumentIndexer",
    "HybridRetriever",
    "JSONDocumentIndexer",
    "RAGSystem",
    "ResponseGenerator",
]
