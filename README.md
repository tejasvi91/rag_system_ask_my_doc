# Ask My Doc — RAG System

A production-grade Retrieval Augmented Generation (RAG) system built in Python.
Upload your documents and ask questions — the system retrieves the most relevant
content and answers with inline citations, refusing to guess when the answer
isn't in the documents.

## What it does

- Ingests PDF and Markdown documents
- Chunks them intelligently with overlapping windows to preserve context
- Stores chunks as vector embeddings in ChromaDB
- Retrieves the most semantically relevant chunks for any question
- Generates answers with inline citations like [Source: document, chunk N]
- Declines to answer when the documents don't support the question

## Why the last point matters

Most RAG demos hallucinate confidently when the answer isn't in the documents.
This system doesn't. If the retrieved context doesn't support an answer,
it says so explicitly. That's the difference between a demo and a
system you can actually trust.

## Tech Stack

- Python 3.11
- LangChain — orchestration
- ChromaDB — vector store
- OpenAI text-embedding-3-small — embeddings
- GPT-4o-mini — answer generation
- Versioned prompt config via YAML

## Project Structure
```
rag-system/
├── src/
│   ├── config.py              # central config and prompt loader
│   ├── ingestion/
│   │   ├── ingest.py          # document loading and chunking
│   │   ├── vectorstore.py     # embedding and ChromaDB storage
│   │   └── run_ingest.py      # ingestion runner
│   └── retrieval/
│       ├── retriever.py       # semantic search
│       └── answer.py          # LLM answer generation
├── prompts/
│   └── prompts.yaml           # versioned prompt config
├── data/                      # place your documents here
└── tests/
    ├── test_env.py
    ├── test_retrieval.py
    └── test_answer.py
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # add your OPENAI_API_KEY
```

## Usage
```bash
# Ingest a document
python -m src.ingestion.run_ingest data/your_document.pdf

# Ask questions
python tests/test_answer.py
```

## Example
```
Q: What optimizer was used to train the model?
A: The optimizer used was Adam with β1=0.9, β2=0.98 and ε=10⁻⁹
   [Source: attention_paper, chunk 11]

Q: What is the capital of France?
A: I cannot answer this from the available documents.
```
