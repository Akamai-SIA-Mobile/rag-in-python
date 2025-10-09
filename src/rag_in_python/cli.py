"""
Command-line interface for the RAG system.
"""

import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .core import RAGSystem
from .json_indexing import JSONDocumentIndexer

# Set up rich console
console = Console()
app = typer.Typer(help="RAG in Python - A comprehensive RAG toolkit")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()  # type: ignore
def index(
    data_dir: Path = typer.Argument(
        ..., help="Directory containing documents to index"
    ),
    index_path: Path = typer.Option(
        Path("./vector_index"),
        "--index-path",
        "-i",
        help="Path to save the vector index",
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", "-b", help="Batch size for indexing"
    ),
) -> None:
    """Index documents from a directory."""

    if not data_dir.exists():
        console.print(f"[red]Error: Directory {data_dir} does not exist[/red]")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Indexing documents from {data_dir}[/blue]")

    # Initialize RAG system
    rag_system = RAGSystem(vector_store_path=index_path)

    # Get all files in directory
    file_paths = list(data_dir.rglob("*"))
    file_paths = [p for p in file_paths if p.is_file()]

    if not file_paths:
        console.print(f"[yellow]No files found in {data_dir}[/yellow]")
        return

    # Separate JSON files from other files
    json_files = [p for p in file_paths if p.suffix.lower() == ".json"]
    other_files = [p for p in file_paths if p.suffix.lower() != ".json"]

    console.print(f"Found {len(file_paths)} files to index:")
    if json_files:
        console.print(f"  • {len(json_files)} JSON files (will be chunked by entries)")
    if other_files:
        console.print(f"  • {len(other_files)} other document files")

    # Index regular documents
    if other_files:
        console.print(f"[blue]Indexing {len(other_files)} regular documents...[/blue]")
        with console.status(f"[bold green]Indexing {len(other_files)} files..."):
            rag_system.load_and_index_files(other_files, batch_size=batch_size)

    # Index JSON files separately
    if json_files:
        console.print(
            f"[blue]Indexing {len(json_files)} JSON files by entries...[/blue]"
        )
        from llama_index.embeddings.openai import OpenAIEmbedding

        embedding_model = OpenAIEmbedding()
        json_indexer = JSONDocumentIndexer(embedding_model=embedding_model)

        all_json_documents = []
        for json_file in json_files:
            try:
                with console.status(f"[bold green]Processing {json_file.name}..."):
                    documents = json_indexer.load_json_documents(json_file)
                    all_json_documents.extend(documents)
                    console.print(f"  • {json_file.name}: {len(documents)} entries")
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not process {json_file}: {e}[/yellow]"
                )

        if all_json_documents:
            # Add JSON documents to the main RAG system
            with console.status(
                f"[bold green]Indexing {len(all_json_documents)} JSON entries..."
            ):
                # Add JSON documents to existing index or create new one
                if rag_system.index is not None:
                    # Add to existing index
                    for doc in all_json_documents:
                        rag_system.index.insert(doc)
                else:
                    # Create new index with JSON documents
                    import faiss
                    from llama_index.core import VectorStoreIndex
                    from llama_index.core.storage.storage_context import StorageContext
                    from llama_index.vector_stores.faiss import FaissVectorStore

                    # Create vector store
                    faiss_index = faiss.IndexFlatIP(1536)  # OpenAI embedding dimension
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store
                    )

                    rag_system.index = VectorStoreIndex.from_documents(
                        all_json_documents,
                        storage_context=storage_context,
                        embed_model=embedding_model,
                    )

    # Save index
    rag_system.save_index()

    console.print(f"[green]✓ Successfully indexed documents to {index_path}[/green]")


@app.command()  # type: ignore
def index_json(
    json_file: Path = typer.Argument(..., help="Path to JSON file to index"),
    index_path: Path = typer.Option(
        Path("./json_vector_index"),
        "--index-path",
        "-i",
        help="Path to save the vector index",
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", "-b", help="Batch size for indexing"
    ),
) -> None:
    """Index a JSON file by treating each top-level entry as a separate document."""

    if not json_file.exists():
        console.print(f"[red]Error: JSON file {json_file} does not exist[/red]")
        raise typer.Exit(1)

    if json_file.suffix.lower() != ".json":
        console.print(f"[red]Error: File {json_file} is not a JSON file[/red]")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Indexing JSON file {json_file}[/blue]")

    # Initialize JSON indexer
    from llama_index.embeddings.openai import OpenAIEmbedding

    embedding_model = OpenAIEmbedding()
    json_indexer = JSONDocumentIndexer(embedding_model=embedding_model)

    try:
        # Load and process JSON documents
        with console.status(f"[bold green]Loading JSON entries from {json_file}..."):
            documents = json_indexer.load_json_documents(json_file)

        console.print(f"Found {len(documents)} JSON entries to index")

        # Index documents
        with console.status(
            f"[bold green]Creating vector index for {len(documents)} entries..."
        ):
            index = json_indexer.index_documents(documents, batch_size=batch_size)

        # Save index
        with console.status(f"[bold green]Saving index to {index_path}..."):
            json_indexer.save_index(index, index_path)

        console.print(
            f"[green]✓ Successfully indexed {len(documents)} JSON entries to {index_path}[/green]"
        )

        # Display sample entries
        console.print("\n[blue]Sample indexed entries:[/blue]")
        for i, doc in enumerate(documents[:3]):
            entry_id = doc.metadata.get("entry_id", f"Entry {i+1}")
            console.print(f"  • {entry_id}")

        if len(documents) > 3:
            console.print(f"  ... and {len(documents) - 3} more entries")

    except Exception as e:
        console.print(f"[red]Error indexing JSON file: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()  # type: ignore
def query_json(
    question: str = typer.Argument(..., help="Question to ask"),
    index_path: Path = typer.Option(
        Path("./json_vector_index"),
        "--index-path",
        "-i",
        help="Path to the JSON vector index",
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of JSON entries to retrieve"
    ),
    with_citations: bool = typer.Option(
        False, "--citations", "-c", help="Include citations in the response"
    ),
) -> None:
    """Query the indexed JSON documents."""

    if not index_path.exists():
        console.print(f"[red]Error: Index path {index_path} does not exist[/red]")
        console.print("Run 'rag-cli index-json' first to create a JSON index")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Question:[/blue] {question}")
    console.print()

    # Initialize JSON indexer and load index
    from llama_index.embeddings.openai import OpenAIEmbedding

    embedding_model = OpenAIEmbedding()
    json_indexer = JSONDocumentIndexer(embedding_model=embedding_model)

    try:
        with console.status("[bold green]Loading JSON index..."):
            index = json_indexer.load_index(index_path)

        with console.status(f"[bold green]Searching {top_k} relevant JSON entries..."):
            # Create a query engine
            query_engine = index.as_query_engine(
                similarity_top_k=top_k, response_mode="tree_summarize"
            )

            # Execute query
            response = query_engine.query(question)

        # Display response
        console.print("[green]Answer:[/green]")
        console.print(str(response))
        console.print()

        # Display metadata
        console.print("[dim]Retrieved from JSON entries[/dim]")

        # Display citations if requested
        if with_citations and hasattr(response, "source_nodes"):
            console.print("\n[blue]Sources:[/blue]")
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("#", style="dim", width=3)
            table.add_column("Entry ID", style="cyan")
            table.add_column("Source", style="yellow")
            table.add_column("Score", style="green", width=8)

            for i, node in enumerate(response.source_nodes, 1):
                entry_id = node.metadata.get("entry_id", "Unknown")
                source_file = node.metadata.get("source_file", "Unknown")
                score = getattr(node, "score", "N/A")

                if isinstance(score, float):
                    score = f"{score:.3f}"

                table.add_row(str(i), str(entry_id), str(source_file), str(score))

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error querying JSON index: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()  # type: ignore
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    index_path: Path = typer.Option(
        Path("./vector_index"), "--index-path", "-i", help="Path to the vector index"
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of documents to retrieve"
    ),
    with_citations: bool = typer.Option(
        False, "--citations", "-c", help="Include citations in the response"
    ),
) -> None:
    """Query the indexed documents."""

    if not index_path.exists():
        console.print(f"[red]Error: Index path {index_path} does not exist[/red]")
        console.print("Run 'rag-cli index' first to create an index")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Question:[/blue] {question}")
    console.print()

    # Initialize RAG system and load index
    rag_system = RAGSystem(vector_store_path=index_path)

    with console.status("[bold green]Loading index..."):
        rag_system.load_index()

    with console.status(f"[bold green]Searching {top_k} relevant documents..."):
        result = rag_system.query(question, top_k=top_k)

    # Display response
    console.print("[green]Answer:[/green]")
    console.print(result["response"])
    console.print()

    # Display metadata
    console.print(
        f"[dim]Retrieved {result['retrieved_documents']} relevant documents[/dim]"
    )

    # Display citations if requested
    if with_citations and result.get("sources"):
        console.print("\n[blue]Sources:[/blue]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Score", style="yellow", width=8)

        for i, source in enumerate(result["sources"], 1):
            score = source.get("score", "N/A")
            source_name = source.get("filename", source.get("source", "Unknown"))
            if isinstance(score, float):
                score = f"{score:.3f}"
            table.add_row(str(i), str(source_name), str(score))

        console.print(table)


@app.command()  # type: ignore
def interactive(
    index_path: Path = typer.Option(
        Path("./vector_index"), "--index-path", "-i", help="Path to the vector index"
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of documents to retrieve"
    ),
) -> None:
    """Start an interactive query session."""

    if not index_path.exists():
        console.print(f"[red]Error: Index path {index_path} does not exist[/red]")
        console.print("Run 'rag-cli index' first to create an index")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)

    # Initialize RAG system and load index
    console.print("[blue]Loading RAG system...[/blue]")
    rag_system = RAGSystem(vector_store_path=index_path)

    with console.status("[bold green]Loading index..."):
        rag_system.load_index()

    console.print("[green]✓ RAG system ready![/green]")
    console.print("Type 'quit' or 'exit' to end the session")
    console.print("=" * 60)

    while True:
        try:
            question = typer.prompt("\n🤖 Ask a question")

            if question.lower() in ["quit", "exit", "q"]:
                console.print("[blue]Goodbye![/blue]")
                break

            if not question.strip():
                continue

            # Query the system
            result = rag_system.query(question, top_k=top_k)

            console.print("\n[green]Answer:[/green]")
            console.print(result["response"])
            console.print(
                f"\n[dim]Sources: {result['retrieved_documents']} documents[/dim]"
            )

        except KeyboardInterrupt:
            console.print("\n[blue]Goodbye![/blue]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()  # type: ignore
def list_commands() -> None:
    """List all available commands and their descriptions."""

    console.print("[bold blue]RAG in Python - Available Commands[/bold blue]")
    console.print("=" * 60)

    commands_info = [
        ("index", "Index documents from a directory", "rag-cli index /path/to/docs"),
        ("index-json", "Index a JSON file by entries", "rag-cli index-json molly.json"),
        ("query", "Query indexed documents", "rag-cli query 'What is this about?'"),
        (
            "query-json",
            "Query indexed JSON entries",
            "rag-cli query-json 'Find entry about X'",
        ),
        ("interactive", "Start interactive query session", "rag-cli interactive"),
        ("list-commands", "Show this help", "rag-cli list-commands"),
    ]

    # Create table
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Command", style="cyan", width=15)
    table.add_column("Description", style="white", width=30)
    table.add_column("Example", style="dim", width=30)

    for cmd, desc, example in commands_info:
        table.add_row(cmd, desc, example)

    console.print(table)

    console.print("\n[yellow]💡 Tips:[/yellow]")
    console.print("  • Set OPENAI_API_KEY environment variable before using")
    console.print("  • Use --help with any command for detailed options")
    console.print(
        "  • JSON indexing treats each top-level entry as a separate document"
    )


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
