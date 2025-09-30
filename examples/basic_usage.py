#!/usr/bin/env python3
"""
Basic usage example for RAG in Python.

This example demonstrates how to:
1. Set up the RAG system
2. Index some sample documents
3. Query the system
4. Display results
"""

import os
from pathlib import Path
from rag_in_python import RAGSystem
from llama_index.core import Document


def main():
    """Run the basic usage example."""
    print("🚀 RAG in Python - Basic Usage Example")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Sample documents
    sample_docs = [
        Document(
            text="Python is a high-level programming language known for its simplicity and readability. "
                 "It's widely used in web development, data science, artificial intelligence, and automation.",
            metadata={"source": "python_intro.txt", "topic": "programming"}
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms "
                 "that can learn and make predictions from data without being explicitly programmed.",
            metadata={"source": "ml_basics.txt", "topic": "ai"}
        ),
        Document(
            text="Vector databases are specialized databases designed to store and query high-dimensional "
                 "vectors efficiently. They're essential for similarity search and recommendation systems.",
            metadata={"source": "vector_db.txt", "topic": "databases"}
        ),
        Document(
            text="Large Language Models (LLMs) like GPT are trained on vast amounts of text data "
                 "and can generate human-like text for various tasks including question answering.",
            metadata={"source": "llm_overview.txt", "topic": "ai"}
        ),
    ]
    
    print(f"📚 Sample documents: {len(sample_docs)}")
    
    # Initialize RAG system
    print("\n🔧 Initializing RAG system...")
    rag_system = RAGSystem(vector_store_path=Path("./example_index"))
    
    # Index the documents
    print("📝 Indexing documents...")
    try:
        rag_system.index_documents(sample_docs)
        print("✅ Documents indexed successfully!")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
        return
    
    # Example queries
    queries = [
        "What is Python used for?",
        "How do machine learning algorithms work?", 
        "What are vector databases good for?",
        "Tell me about Large Language Models",
    ]
    
    print(f"\n🤖 Running {len(queries)} example queries...")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 30)
        
        try:
            result = rag_system.query(query, top_k=2)
            
            print(f"💡 Answer: {result['response']}")
            print(f"📊 Sources: {result['retrieved_documents']} documents")
            
            # Show sources if available
            if result.get('sources'):
                print("🔗 Sources:")
                for j, source in enumerate(result['sources'], 1):
                    source_name = source.get('source', 'Unknown')
                    topic = source.get('topic', 'N/A')
                    print(f"   {j}. {source_name} (topic: {topic})")
                    
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Tips:")
    print("   • Set different top_k values to retrieve more/fewer documents")
    print("   • Experiment with different similarity thresholds")
    print("   • Try the CLI: rag-cli interactive --index-path ./example_index")
    

if __name__ == "__main__":
    main()