from typing import List
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from src.ingestion.vectorstore import load_vector_store
from src.config import TOP_K


def get_all_chunks() -> List[Document]:
    """Fetch every chunk stored in ChromaDB for BM25 indexing."""
    vector_store = load_vector_store()
    collection = vector_store._collection
    results = collection.get(include=["documents", "metadatas"])

    chunks = []
    for i, text in enumerate(results["documents"]):
        doc = Document(
            page_content=text,
            metadata=results["metadatas"][i]
        )
        chunks.append(doc)

    return chunks


def bm25_search(query: str, chunks: List[Document], top_k: int) -> List[Document]:
    """
    Keyword search using BM25.
    Tokenises by whitespace — fast and effective for technical terms,
    model names, numbers, and exact phrases vector search often misses.
    """
    tokenised_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenised_corpus)

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    # Pair each chunk with its score and sort descending
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def hybrid_search(query: str, top_k: int = TOP_K) -> List[Document]:
    """
    Combine BM25 keyword search and vector semantic search.
    Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists
    into a single ranked result without needing score normalisation.
    """
    # Load all chunks once for BM25
    all_chunks = get_all_chunks()

    # BM25 results
    bm25_results = bm25_search(query, all_chunks, top_k)

    # Vector results
    vector_store = load_vector_store()
    vector_results = vector_store.similarity_search(query, k=top_k)

    # Reciprocal Rank Fusion — standard way to merge ranked lists
    # Score = sum of 1/(rank + 60) across both lists for each chunk
    rrf_scores = {}
    chunk_map = {}

    for rank, doc in enumerate(bm25_results):
        key = doc.metadata.get("chunk_id", doc.page_content[:50])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
        chunk_map[key] = doc

    for rank, doc in enumerate(vector_results):
        key = doc.metadata.get("chunk_id", doc.page_content[:50])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
        chunk_map[key] = doc

    # Sort by combined RRF score
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [chunk_map[k] for k in sorted_keys[:top_k]]