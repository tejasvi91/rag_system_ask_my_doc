import hashlib
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_document(source: str) -> List[Document]:
    """Load a single PDF or Markdown file into LangChain Document objects."""
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix == ".md":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf or .md")

    docs = loader.load()

    # Attach a clean source name to every page for citations later
    for doc in docs:
        doc.metadata["source_name"] = path.stem

    print(f"  Loaded '{path.name}' — {len(docs)} page(s)")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.
    RecursiveCharacterTextSplitter tries to split on paragraphs first,
    then sentences, then words — preserving semantic coherence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Give every chunk a stable unique ID for citations
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_name", "doc")
        chunk_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:6]
        chunk.metadata["chunk_id"] = f"{source}_chunk{i}_{chunk_hash}"

    print(f"  Split into {len(chunks)} chunks")
    return chunks


def ingest_files(file_paths: List[str]) -> List[Document]:
    """Full ingestion: load all files and chunk them. Returns all chunks."""
    all_chunks = []

    for path in file_paths:
        print(f"Processing: {path}")
        docs = load_document(path)
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks ready for embedding: {len(all_chunks)}")
    return all_chunks