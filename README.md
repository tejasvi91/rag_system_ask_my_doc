# Production RAG System

A production-grade Retrieval Augmented Generation (RAG) system built in Python.
Ingests PDFs and Markdown documents, stores them as vector embeddings, and answers
questions with inline citations — declining to answer when the context doesn't support it.

## Architecture

**Phase 1 (complete)** — Core pipeline
- Document ingestion: PDF and Markdown support
- Chunking: 700 token window with 100 token overlap
- Vector store: ChromaDB with OpenAI embeddings
- Retrieval: Top-K semantic search
- Answer generation: GPT-4o-mini with citation enforcement
- Versioned prompt config in `prompts/prompts.yaml`

**Phase 2 (upcoming)** — Production quality
- Hybrid retrieval: BM25 keyword + vector semantic search
- Cross-encoder reranker for improved precision
- Hard citation enforcement — declines unsupported answers

**Phase 3 (upcoming)** — Evaluation & CI
- Golden dataset: 50–200 manually verified Q&A pairs
- RAGAS faithfulness evaluation
- CI pipeline — build fails if quality drops below threshold

## Tech Stack

- LangChain — orchestration
- ChromaDB — vector store
- OpenAI — embeddings and LLM
- RAGAS — evaluation (Phase 3)

## Setup
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
```

## Usage
```bash
# Ingest documents
python -m src.ingestion.run_ingest data/your_document.pdf

# Run Q&A
python tests/test_answer.py
```
