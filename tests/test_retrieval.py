import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import retrieve_chunks, format_context

chunks = retrieve_chunks('What is machine learning?')
print(format_context(chunks))