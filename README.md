# RAG in Python

A comprehensive RAG (Retrieval-Augmented Generation) toolkit built with LlamaIndex and FAISS, following Python community best practices.

## Features

- 🚀 **Modern Python 3.12** - Built with the latest Python features
- 📚 **LlamaIndex Integration** - Powerful document indexing and retrieval
- ⚡ **FAISS Vector Store** - High-performance similarity search
- 🗄️ **JSON Document Support** - Specialized indexing for JSON files by entries
- 🔧 **CLI Interface** - Easy-to-use command-line tools
- 🏗️ **Modular Architecture** - Clean separation of concerns with inheritance-based design
- 📦 **uv & hatchling** - Modern Python packaging and dependency management
- 🧪 **Type Safety** - Full type hints and mypy support
- 🎯 **Developer Experience** - Pre-commit hooks, linting, and testing setup

## Installation

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/Akamai-SIA-Mobile/rag-in-python.git
cd rag-in-python
```

2. Install `uv` if not already installed
```bash
# MacOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Install the package with development dependencies:
```bash
uv sync --extra dev
```

4. Activate `uv` environment, so that no need using `uv run` before commands:
```bash
# MacOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
source .\.venv\Scripts\activate
```

## Quick Start

### 1. Set up your environment

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Index your documents

**Regular documents:**
```bash
uv run rag-cli index /path/to/documents/folder --index-path ./vector_index
```

**JSON files (by entries):**
```bash
uv run rag-cli index-json data.json --index-path ./json_index
```

### 3. Query your documents

**Using Regular index:**
```bash
uv run rag-cli query "What is the main topic?" --index-path ./vector_index
```

**Using JSON index:**
```bash
uv run rag-cli query-json "Find information about..." --index-path ./json_index --citations
```

### 4. Interactive mode

**Using Regular index:**
```bash
uv run rag-cli interactive --index-path ./vector_index
```

**Using JSON index**
```bash
uv run rag-cli interactive --index-path ./json_index
```

## Project Structure

```
rag-in-python/
├── src/
│   └── rag_in_python/
│       ├── __init__.py          # Main package exports
│       ├── core.py              # Core RAG system orchestration
│       ├── base_indexer.py      # Base class for indexers
│       ├── indexing.py          # Document loading and indexing
│       ├── json_indexing.py     # JSON-specific document indexing
│       ├── retrieval.py         # Document retrieval with FAISS
│       ├── generation.py        # Response generation with LLMs
│       ├── debug_utils.py       # Debug utilities for retrieval
│       └── cli.py               # Command-line interface
├── tests/                       # Test suite
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── pyproject.toml              # Project configuration
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Architecture

### Core Components

1. **RAGSystem** (`core.py`) - Main orchestrator that coordinates all components
2. **BaseIndexer** (`base_indexer.py`) - Abstract base class with shared vector store operations
3. **DocumentIndexer** (`indexing.py`) - Handles general document loading and FAISS indexing
4. **JSONDocumentIndexer** (`json_indexing.py`) - Specialized indexer for JSON files by entries
5. **HybridRetriever** (`retrieval.py`) - Retrieves relevant documents using vector similarity
6. **ResponseGenerator** (`generation.py`) - Generates responses using retrieved context
7. **RetrievalDebugger** (`debug_utils.py`) - Debug utilities for retrieval operations

### Design Principles

- **Modularity**: Each component has a single responsibility
- **Inheritance**: Base classes provide shared functionality to reduce code duplication
- **Type Safety**: Full type hints for better IDE support and fewer bugs
- **Extensibility**: Easy to swap out components (e.g., different LLMs, vector stores)
- **Performance**: Batch processing and efficient vector operations
- **Observability**: Comprehensive logging throughout the system

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Supported Document Types

- **Text files**: `.txt`, `.md`, `.rst`
- **PDFs**: `.pdf`
- **Microsoft Office**: `.docx`, `.pptx`
- **Web content**: `.html`, `.xml`
- **JSON files**: `.json` (indexed by top-level entries)
- **Code files**: `.py`, `.js`, `.json`, etc.

## Development

### Setting up the development environment

```bash
# Clone the repository
git clone https://github.com/Akamai-SIA-Mobile/rag-in-python.git
cd rag-in-python

# Install with development dependencies
uv sync --extra dev

# Set up pre-commit hooks
uv run pre-commit install
```

### Running tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=rag_in_python

# Run specific test categories
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m "unit"      # Unit tests only
```

### Code quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
uv run ruff check --fix src/ tests/

# Type checking
uv run mypy src/

# Run all checks
uv run pre-commit run --all-files
```

### Building the package

```bash
# Using uv (recommended)
uv build

# Using standard tools
python -m build
```

## Python API Usage

### Basic RAG System

```python
from pathlib import Path
from rag_in_python import RAGSystem

# Initialize the RAG system
rag = RAGSystem(vector_store_path=Path("./vector_index"))

# Index documents
document_paths = [Path("doc1.txt"), Path("doc2.pdf")]
rag.load_and_index_files(document_paths)

# Save the index
rag.save_index()

# Query the system
result = rag.query("What is the main topic?", top_k=5)
print(result["response"])
print(f"Sources: {result['retrieved_documents']} documents")
```

### JSON Document Indexing

```python
from pathlib import Path
from llama_index.embeddings.openai import OpenAIEmbedding
from rag_in_python import JSONDocumentIndexer

# Initialize JSON indexer
embedding_model = OpenAIEmbedding()
json_indexer = JSONDocumentIndexer(
    embedding_model=embedding_model,
    vector_store_path=Path("./json_index")
)

# Load and index JSON file (each top-level entry becomes a document)
documents = json_indexer.load_json_documents(Path("data.json"))
index = json_indexer.index_documents(documents)
json_indexer.save_index(index, Path("./json_index"))

# Query the JSON index
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("Find entry about...")
print(response)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) - For the powerful RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - For efficient similarity search
- [OpenAI](https://openai.com/) - For the language models and embeddings

## Support

- 📖 [Documentation](https://github.com/Akamai-SIA-Mobile/rag-in-python/blob/main/README.md)
- 🐛 [Issue Tracker](https://github.com/Akamai-SIA-Mobile/rag-in-python/issues)
- 💬 [Discussions](https://github.com/Akamai-SIA-Mobile/rag-in-python/discussions)

---

Built with ❤️ using modern Python practices and tools.
