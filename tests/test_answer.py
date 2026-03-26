import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.answer import answer_question

questions = [
    "What is machine learning?",
    "What are the three types of machine learning?",
    "What is the capital of France?",  # should be declined — not in documents
]

for q in questions:
    print(f"\nQ: {q}")
    print("-" * 50)
    result = answer_question(q)
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")