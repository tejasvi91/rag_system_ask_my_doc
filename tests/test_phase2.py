import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.answer_v2 import answer_question_v2

questions = [
    "What optimizer was used and what were its parameters?",
    "What is the BLEU score on English to German translation?",
    "What is the capital of France?",
]

print("=== Phase 2 RAG Pipeline Test ===")

for q in questions:
    result = answer_question_v2(q)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Citation audit: {result['citation_audit']}")
    print("-" * 60)