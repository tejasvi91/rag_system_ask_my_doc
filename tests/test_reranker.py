import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank

print("=== Reranker Test ===\n")

query = "What optimizer was used with what parameters?"
print(f"Query: {query}\n")

# Step 1: Hybrid retrieval — gets 5 candidates
candidates = hybrid_search(query, top_k=5)
print(f"Candidates from hybrid search: {len(candidates)}")
for i, c in enumerate(candidates):
    print(f"  {i+1}. {c.metadata.get('chunk_id')} ")

# Step 2: Reranker — narrows to top 3
print("\nReranking...")
reranked = rerank(query, candidates, top_k=3)

print(f"\nTop {len(reranked)} after reranking:")
for i, c in enumerate(reranked):
    score = c.metadata.get('reranker_score', 'N/A')
    print(f"  {i+1}. [{score}] {c.metadata.get('chunk_id')}")
    print(f"     Preview: {c.page_content[:100].strip()}...")