from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
from src.config import TOP_K
from src.ingestion.vectorstore import load_vector_store


def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[Document]:
    """
    Convert the query to a vector and fetch the top_k most similar chunks.
    Returns a list of Document objects with content and metadata.
    """
    vector_store = load_vector_store()

    results = vector_store.similarity_search(query, k=top_k)

    print(f"Retrieved {len(results)} chunks for query: '{query}'")
    return results


def format_context(chunks: List[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.
    Each chunk is clearly labelled with its source and chunk ID.
    """
    formatted = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_name", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i}")

        formatted.append(
            f"[Chunk {i+1} | Source: {source} | ID: {chunk_id}]\n"
            f"{chunk.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted)