"""
Response generation functionality using LLMs.
"""

from typing import List, Dict, Any, Optional
import logging

from llama_index.core import Document
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates responses using retrieved documents and LLM.
    """
    
    def __init__(self, llm: LLM) -> None:
        """
        Initialize the response generator.
        
        Args:
            llm: Language model for generation.
        """
        self.llm = llm
        
    def generate(
        self,
        query: str,
        context_docs: List[Document],
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """
        Generate response based on query and context documents.
        
        Args:
            query: User query.
            context_docs: Retrieved documents for context.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated response string.
        """
        # Prepare context from retrieved documents
        context = self._prepare_context(context_docs)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        response = self.llm.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        logger.info(f"Generated response of length {len(response.text)}")
        return response.text
        
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        if not documents:
            return "No relevant context found."
            
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Add document with source information
            source_info = ""
            if doc.metadata:
                if "filename" in doc.metadata:
                    source_info = f" (Source: {doc.metadata['filename']})"
                elif "source" in doc.metadata:
                    source_info = f" (Source: {doc.metadata['source']})"
                    
            context_parts.append(f"Document {i}{source_info}:\n{doc.text}\n")
            
        return "\n".join(context_parts)
        
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create the final prompt for the LLM.
        
        Args:
            query: User query.
            context: Context from retrieved documents.
            
        Returns:
            Formatted prompt string.
        """
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
        
    def generate_with_citations(
        self,
        query: str,
        context_docs: List[Document],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate response with citations to source documents.
        
        Args:
            query: User query.
            context_docs: Retrieved documents for context.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens in response.
            
        Returns:
            Dictionary with response and citation information.
        """
        # Generate response
        response = self.generate(
            query=query,
            context_docs=context_docs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Extract citations from context documents
        citations = []
        for i, doc in enumerate(context_docs, 1):
            citation = {
                "index": i,
                "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                "metadata": doc.metadata,
            }
            
            # Add score if available
            if "score" in doc.metadata:
                citation["similarity_score"] = doc.metadata["score"]
                
            citations.append(citation)
            
        return {
            "response": response,
            "citations": citations,
            "num_sources": len(context_docs),
        }