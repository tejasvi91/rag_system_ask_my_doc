import sys
from src.ingestion.ingest import ingest_files
from src.ingestion.vectorstore import store_chunks


def main(file_paths: list):
    print("=== RAG Ingestion Pipeline ===\n")

    # Step 1: Load and chunk
    chunks = ingest_files(file_paths)

    # Step 2: Embed and store
    store_chunks(chunks)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    # Pass file paths as command line arguments
    # e.g. python -m src.ingestion.run_ingest data/sample.md
    files = sys.argv[1:] if len(sys.argv) > 1 else ["data/sample.md"]
    main(files)