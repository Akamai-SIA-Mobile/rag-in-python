"""
RAG in Python - A comprehensive RAG toolkit built with LlamaIndex and FAISS.

This package provides tools and utilities for building Retrieval-Augmented Generation
systems using modern Python practices, LlamaIndex framework, and FAISS vector storage.
"""

__version__ = "0.1.0"
__author__ = "SIA Mobile AI Ambassador Team"
__email__ = "zcui@akamai.com"

from .core import RAGSystem
from .base_indexer import BaseIndexer
from .indexing import DocumentIndexer
from .retrieval import HybridRetriever
from .generation import ResponseGenerator
from .json_indexing import JSONDocumentIndexer

__all__ = [
    "RAGSystem",
    "BaseIndexer",
    "DocumentIndexer", 
    "JSONDocumentIndexer",
    "HybridRetriever",
    "ResponseGenerator",
]