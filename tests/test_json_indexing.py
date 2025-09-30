#!/usr/bin/env python3
"""
Test script for JSON indexing functionality.
This demonstrates how to chunk and index the molly.json file by entries.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file and return the data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_json_by_entries(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk JSON data by top-level entries.
    Each entry becomes a separate chunk for indexing.
    """
    chunks = []
    
    for entry_id, entry_data in json_data.items():
        # Create a chunk with the entry ID and its full data
        chunk = {
            "id": entry_id,
            "content": json.dumps(entry_data, indent=2),
            "metadata": {
                "source": "molly.json",
                "entry_id": entry_id,
                "type": "json_entry"
            }
        }
        chunks.append(chunk)
    
    return chunks

def main():
    """Main function to test JSON chunking."""
    # Path to the molly.json file
    json_file_path = Path("test_docs/molly.json")
    
    if not json_file_path.exists():
        print(f"Error: {json_file_path} not found!")
        return
    
    print(f"Loading JSON file: {json_file_path}")
    
    # Load the JSON data
    json_data = load_json_file(json_file_path)
    
    # Chunk the data by entries
    chunks = chunk_json_by_entries(json_data)
    
    print(f"Total entries found: {len(chunks)}")
    print("\nFirst 3 entry IDs:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  {i+1}. {chunk['id']}")
    
    # Show example of first chunk
    if chunks:
        first_chunk = chunks[0]
        print(f"\nExample chunk for entry '{first_chunk['id']}':")
        print(f"Content preview (first 200 chars):")
        print(first_chunk['content'][:200] + "..." if len(first_chunk['content']) > 200 else first_chunk['content'])
        print(f"Metadata: {first_chunk['metadata']}")

if __name__ == "__main__":
    main()