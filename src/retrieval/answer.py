from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config import OPENAI_API_KEY, CHAT_MODEL
from src.retrieval.retriever import retrieve_chunks, format_context
from src.config import load_prompts


def answer_question(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Format them as context
    3. Send to LLM with citation-enforcing prompt
    4. Return answer + source chunks
    """
    # Step 1: Retrieve
    chunks = retrieve_chunks(question)

    if not chunks:
        return {
            "question": question,
            "answer": "I cannot answer this from the available documents.",
            "sources": []
        }

    # Step 2: Format context
    context = format_context(chunks)

    # Step 3: Load versioned prompts
    prompts = load_prompts()
    system_prompt = prompts["rag_answer"]["system"]
    user_prompt = prompts["rag_answer"]["user"].format(
        context=context,
        question=question
    )

    # Step 4: Call LLM
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0  # deterministic — no creativity, just facts from context
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    # Step 5: Collect source IDs for transparency
    sources = [
        chunk.metadata.get("chunk_id", "unknown")
        for chunk in chunks
    ]

    return {
        "question": question,
        "answer": response.content,
        "sources": sources
    }