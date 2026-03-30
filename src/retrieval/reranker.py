from typing import List
from langchain.schema import Document
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model once and reuse it.
    First call downloads the model (~80MB) — subsequent calls are instant.
    """
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model '{MODEL_NAME}'...")
        _reranker = CrossEncoder(MODEL_NAME)
        print("Reranker ready.")
    return _reranker


def rerank(query: str, chunks: List[Document], top_k: int = 3) -> List[Document]:
    """
    Re-score retrieved chunks by evaluating the query and each chunk
    together as a pair. Much more precise than initial retrieval because
    the model sees the full query-chunk relationship, not just similarity.

    Returns top_k chunks sorted by reranker score descending.
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Build query-chunk pairs for the cross-encoder
    pairs = [[query, doc.page_content] for doc in chunks]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    scored_chunks = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    # Store reranker score in metadata for transparency
    results = []
    for score, doc in scored_chunks[:top_k]:
        doc.metadata["reranker_score"] = round(float(score), 4)
        results.append(doc)

    return results