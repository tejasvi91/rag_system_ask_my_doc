import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.retriever import format_context

print("=== Hybrid Retrieval Test ===\n")

query = "What optimizer was used with what parameters?"
print(f"Query: {query}\n")

chunks = hybrid_search(query)
print(f"Retrieved {len(chunks)} chunks\n")
print(format_context(chunks))