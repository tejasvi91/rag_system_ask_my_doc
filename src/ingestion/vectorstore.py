from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_DIR


COLLECTION_NAME = "rag_documents"


def get_embeddings():
    """Return the embedding model. Single place to swap models later."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )


def store_chunks(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and persist them to ChromaDB.
    If the collection already exists, this adds to it.
    """
    print(f"Embedding {len(chunks)} chunks with model '{EMBEDDING_MODEL}'...")

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Stored successfully in: {CHROMA_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """Load an already-persisted vector store from disk."""
    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )