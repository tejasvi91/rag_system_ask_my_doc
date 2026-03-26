import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.answer import answer_question

questions = [
    "What is the attention mechanism?",
    "What optimizer was used to train the model?",
    "What is the BLEU score achieved on WMT 2014 English-to-German translation?",
]

for q in questions:
    print(f"\nQ: {q}")
    print("-" * 50)
    result = answer_question(q)
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")