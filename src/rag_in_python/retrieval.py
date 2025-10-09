"""
Document retrieval functionality with hybrid search capabilities.
"""

import logging
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from .debug_utils import RetrievalDebugger

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval system combining vector search with additional filtering.
    """

    def __init__(self) -> None:
        """Initialize the hybrid retriever."""
        pass

    def retrieve(
        self,
        query: str,
        index: VectorStoreIndex,
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Retrieve relevant documents using vector similarity.

        Args:
            query: Query string.
            index: Vector store index to search.
            top_k: Number of top documents to retrieve.
            similarity_threshold: Minimum similarity threshold.
            **kwargs: Additional retrieval parameters.

        Returns:
            List of retrieved documents.
        """
        logger.info(
            f"Starting vector retrieval - Query: '{query}', "
            f"Top-k: {top_k}, Similarity threshold: {similarity_threshold}"
        )

        # Create retriever from index
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

        # Retrieve nodes
        retrieved_nodes: list[NodeWithScore] = retriever.retrieve(query)

        # Use debug utilities for logging
        RetrievalDebugger.print_retrieval_summary(
            query, retrieved_nodes, similarity_threshold
        )
        RetrievalDebugger.print_node_details(retrieved_nodes)

        # Filter by similarity threshold and convert to documents
        documents = []
        for i, node_with_score in enumerate(retrieved_nodes):
            score = node_with_score.score if node_with_score.score is not None else 0.0

            if score >= similarity_threshold:
                text = node_with_score.node.get_content()
                logger.info(
                    f"Retrieved document #{i+1}: "
                    f"Score={score:.4f}, "
                    f"Node ID={node_with_score.node.node_id}, "
                    f"Text length={len(text)} chars"
                )

                # Convert node to document
                doc = Document(
                    text=text,
                    metadata={
                        **node_with_score.node.metadata,
                        "score": score,
                        "node_id": node_with_score.node.node_id,
                    },
                )
                documents.append(doc)

        # Print filtering results
        accepted_count, filtered_count = RetrievalDebugger.print_filtering_results(
            retrieved_nodes, similarity_threshold
        )
        RetrievalDebugger.print_final_summary(
            accepted_count, filtered_count, len(retrieved_nodes)
        )

        logger.info(
            f"Retrieved {len(documents)} documents (out of {len(retrieved_nodes)} candidates) "
            f"above similarity threshold {similarity_threshold}"
        )

        # Provide helpful suggestions if no documents were retrieved
        if len(documents) == 0 and len(retrieved_nodes) > 0:
            all_scores = [
                node.score for node in retrieved_nodes if node.score is not None
            ]
            if all_scores:
                max_score = max(all_scores)
                logger.warning(
                    f"No documents retrieved! Consider lowering similarity_threshold from {similarity_threshold} "
                    f"to {max_score:.4f} or lower. Highest candidate score was {max_score:.4f}"
                )

        # Log summary of retrieved documents
        if documents:
            scores = [doc.metadata.get("score", 0) for doc in documents]
            logger.info(
                f"Retrieval summary - Top score: {max(scores):.4f}, "
                f"Lowest score: {min(scores):.4f}, "
                f"Average score: {sum(scores)/len(scores):.4f}"
            )

        return documents

    def retrieve_with_metadata_filter(
        self,
        query: str,
        index: VectorStoreIndex,
        metadata_filters: dict | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.1,
    ) -> list[Document]:
        """
        Retrieve documents with additional metadata filtering.

        Args:
            query: Query string.
            index: Vector store index to search.
            metadata_filters: Dictionary of metadata key-value pairs to filter by.
            top_k: Number of top documents to retrieve.
            similarity_threshold: Minimum similarity threshold.

        Returns:
            List of retrieved and filtered documents.
        """
        logger.info(
            f"Starting hybrid retrieval - Query: '{query}', "
            f"Top-k: {top_k}, Similarity threshold: {similarity_threshold}, "
            f"Metadata filters: {metadata_filters}"
        )

        # First retrieve based on similarity
        documents = self.retrieve(
            query=query,
            index=index,
            top_k=top_k * 2,  # Get more candidates for filtering
            similarity_threshold=similarity_threshold,
        )

        # Apply metadata filters if provided
        if metadata_filters:
            filtered_documents = []
            logger.info(f"Applying metadata filters: {metadata_filters}")

            for i, doc in enumerate(documents):
                if self._matches_metadata_filters(doc.metadata, metadata_filters):
                    filtered_documents.append(doc)
                    logger.debug(f"Document #{i+1} passed metadata filters")
                else:
                    logger.debug(
                        f"Document #{i+1} filtered out by metadata. "
                        f"Document metadata: {doc.metadata}, "
                        f"Required filters: {metadata_filters}"
                    )

            documents = filtered_documents[:top_k]  # Limit to top_k after filtering

            logger.info(
                f"Applied metadata filters, {len(documents)} documents remaining out of {len(filtered_documents)} that passed filters"
            )

        # Log final summary
        if documents:
            scores = [doc.metadata.get("score", 0) for doc in documents]
            logger.info(
                f"Final retrieval summary - {len(documents)} documents, "
                f"Top score: {max(scores):.4f}, "
                f"Lowest score: {min(scores):.4f}, "
                f"Average score: {sum(scores)/len(scores):.4f}"
            )

        return documents[:top_k]

    def _matches_metadata_filters(self, metadata: dict, filters: dict) -> bool:
        """
        Check if document metadata matches the provided filters.

        Args:
            metadata: Document metadata.
            filters: Metadata filters to apply.

        Returns:
            True if metadata matches all filters, False otherwise.
        """
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def debug_vector_store_contents(
        self,
        index: VectorStoreIndex,
        sample_query: str = "test query",
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        show_full_content: bool = True,
    ) -> None:
        """
        Debug method to inspect what's actually in your vector store.
        """
        RetrievalDebugger.debug_vector_store_contents(
            index=index,
            sample_query=sample_query,
            top_k=top_k,
            show_full_content=show_full_content,
        )

    def test_query_variations(
        self,
        index: VectorStoreIndex,
        base_topic: str = "data connection speed",
        similarity_threshold: float = 0.1,
        top_k: int = 3,
    ) -> None:
        """
        Test different query variations to find the best way to retrieve relevant documents.

        Args:
            index: Vector store index to search.
            base_topic: The topic you're trying to find information about.
            similarity_threshold: Similarity threshold to use.
            top_k: Number of top documents to retrieve.
        """
        # Generate different query variations
        query_variations = [
            base_topic,
            f"how to {base_topic}",
            f"adjust {base_topic}",
            f"configure {base_topic}",
            f"optimize {base_topic}",
            f"improve {base_topic}",
            f"settings for {base_topic}",
            base_topic.replace(" ", " and "),
            # Split the topic into individual terms
            *base_topic.split(),
        ]

        print(f"\n{'='*80}")
        print(f"TESTING QUERY VARIATIONS FOR: '{base_topic}'")
        print(f"{'='*80}")

        best_query = None
        best_score = 0.0
        best_count = 0

        for query in query_variations:
            print(f"\nTesting query: '{query}'")
            print(f"{'-'*50}")

            try:
                # Use the retrieve method with our enhanced logging
                documents = self.retrieve(
                    query=query,
                    index=index,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )

                if documents:
                    scores = [doc.metadata.get("score", 0) for doc in documents]
                    max_score = max(scores)
                    print(
                        f"✅ Found {len(documents)} documents, best score: {max_score:.4f}"
                    )

                    if max_score > best_score:
                        best_query = query
                        best_score = max_score
                        best_count = len(documents)
                else:
                    print(
                        f"❌ No documents found above threshold {similarity_threshold}"
                    )

            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")

        print(f"\n{'='*80}")
        print("BEST QUERY RESULTS:")
        print(f"{'='*80}")
        if best_query:
            print(f"🏆 Best query: '{best_query}'")
            print(f"   Documents found: {best_count}")
            print(f"   Best score: {best_score:.4f}")
            print(f"\n💡 RECOMMENDATION: Use '{best_query}' for your retrieval")
        else:
            print(
                f"❌ No queries returned results above threshold {similarity_threshold}"
            )
            print("💡 RECOMMENDATIONS:")
            print("   1. Lower the similarity threshold (try 0.05 or 0.01)")
            print(f"   2. Add documents about '{base_topic}' to your vector store")
            print("   3. Check if your documents use different terminology")
        print(f"{'='*80}\n")
