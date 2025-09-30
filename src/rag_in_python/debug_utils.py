"""
Debug utilities for retrieval operations.
"""

from typing import List, Optional
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class RetrievalDebugger:
    """
    Utility class for debugging retrieval operations.
    """
    
    @staticmethod
    def print_retrieval_summary(query: str, retrieved_nodes: List[NodeWithScore], similarity_threshold: float) -> None:
        """
        Print a summary of retrieval results.
        
        Args:
            query: The search query.
            retrieved_nodes: List of retrieved nodes with scores.
            similarity_threshold: The similarity threshold used.
        """
        print(f"\n{'='*60}")
        print(f"VECTOR STORE RETRIEVAL RESULTS FOR QUERY: '{query}'")
        print(f"{'='*60}")
        print(f"Total candidates found: {len(retrieved_nodes)}")
        
        if retrieved_nodes:
            all_scores = [node.score for node in retrieved_nodes if node.score is not None]
            print(f"Candidate scores: {[f'{score:.4f}' for score in all_scores]}")
    
    @staticmethod
    def print_node_details(retrieved_nodes: List[NodeWithScore]) -> None:
        """
        Print detailed information about retrieved nodes.
        
        Args:
            retrieved_nodes: List of retrieved nodes with scores.
        """
        print(f"\nDETAILS OF ALL CANDIDATES:")
        print(f"{'-'*50}")
        
        for i, node_with_score in enumerate(retrieved_nodes):
            score = node_with_score.score if node_with_score.score is not None else 0.0
            print(f"\nCandidate #{i+1}:")
            print(f"  Score: {score:.4f}")
            print(f"  Node ID: {node_with_score.node.node_id}")
            print(f"  Text length: {len(node_with_score.node.text)} characters")
            print(f"  Metadata: {node_with_score.node.metadata}")
            
            # Show FULL document content
            print(f"  FULL CONTENT:")
            print(f"  {'-'*40}")
            content_lines = node_with_score.node.text.split('\n')
            for line in content_lines:
                print(f"  {line}")
            print(f"  {'-'*40}")
    
    @staticmethod
    def print_filtering_results(retrieved_nodes: List[NodeWithScore], similarity_threshold: float) -> tuple:
        """
        Print filtering results and return accepted/filtered counts.
        
        Args:
            retrieved_nodes: List of retrieved nodes with scores.
            similarity_threshold: The similarity threshold used.
            
        Returns:
            Tuple of (accepted_count, filtered_count).
        """
        print(f"\n{'-'*50}")
        print(f"FILTERING WITH THRESHOLD: {similarity_threshold}")
        print(f"{'-'*50}")
        
        accepted_count = 0
        filtered_count = 0
        
        for i, node_with_score in enumerate(retrieved_nodes):
            score = node_with_score.score if node_with_score.score is not None else 0.0
            
            if score >= similarity_threshold:
                print(f"✅ Document #{i+1} ACCEPTED (Score: {score:.4f} >= {similarity_threshold})")
                accepted_count += 1
            else:
                print(f"❌ Document #{i+1} REJECTED (Score: {score:.4f} < {similarity_threshold})")
                filtered_count += 1
                
        return accepted_count, filtered_count
    
    @staticmethod
    def print_final_summary(accepted_count: int, filtered_count: int, total_candidates: int) -> None:
        """
        Print final retrieval summary.
        
        Args:
            accepted_count: Number of accepted documents.
            filtered_count: Number of filtered documents.
            total_candidates: Total number of candidate documents.
        """
        print(f"\n{'='*60}")
        print(f"FINAL RETRIEVAL SUMMARY")
        print(f"{'='*60}")
        print(f"Documents accepted: {accepted_count}")
        print(f"Documents rejected: {filtered_count}")
        print(f"Total candidates: {total_candidates}")
        
        if accepted_count == 0:
            print("❌ NO DOCUMENTS ACCEPTED!")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def debug_vector_store_contents(
        index: VectorStoreIndex,
        sample_query: str = "test query",
        top_k: int = 10,
        show_full_content: bool = True
    ) -> None:
        """
        Debug method to inspect what's actually in your vector store.
        
        Args:
            index: Vector store index to inspect.
            sample_query: Query to use for retrieval.
            top_k: Number of documents to retrieve for inspection.
            show_full_content: If True, shows full document content.
        """
        print(f"\n{'='*80}")
        print(f"VECTOR STORE CONTENT INSPECTION")
        print(f"{'='*80}")
        print(f"Using query: '{sample_query}'")
        
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        
        try:
            retrieved_nodes: List[NodeWithScore] = retriever.retrieve(sample_query)
            
            print(f"Total documents found: {len(retrieved_nodes)}")
            
            if not retrieved_nodes:
                print("❌ NO DOCUMENTS FOUND IN VECTOR STORE!")
                print("   This means your vector store is empty or not properly indexed.")
                return
            
            print(f"\nDOCUMENT INVENTORY:")
            print(f"{'-'*60}")
            
            for i, node_with_score in enumerate(retrieved_nodes):
                score = node_with_score.score if node_with_score.score is not None else 0.0
                
                print(f"\nDocument #{i+1}:")
                print(f"  Score: {score:.4f}")
                print(f"  Node ID: {node_with_score.node.node_id}")
                print(f"  Text length: {len(node_with_score.node.text)} characters")
                print(f"  Metadata: {node_with_score.node.metadata}")
                
                if show_full_content:
                    print(f"  FULL CONTENT:")
                    print(f"  {'-'*40}")
                    content_lines = node_with_score.node.text.split('\n')
                    for line in content_lines:
                        print(f"    {line}")
                    print(f"  {'-'*40}")
                else:
                    text_preview = node_with_score.node.text[:200]
                    if len(node_with_score.node.text) > 200:
                        text_preview += "..."
                    print(f"  Content preview: {text_preview}")
            
            print(f"\n{'='*80}")
            print(f"INSPECTION COMPLETE")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ ERROR inspecting vector store: {e}")
            print("   Check if your index is properly initialized.")