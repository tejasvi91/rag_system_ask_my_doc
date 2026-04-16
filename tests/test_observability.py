import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.answer_v3 import answer_question_v3

questions = [
    "What optimizer was used and what were its parameters?",
    "What is the BLEU score on English to German translation?",
    "What is the capital of France?",
]

print("=== Observability Pipeline Test ===\n")

for q in questions:
    result = answer_question_v3(q)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Trace ID: {result['trace_id']}")
    print(f"Metrics: {result['metrics']}")
    print(f"View trace: http://localhost:3000")
    print("-" * 60)