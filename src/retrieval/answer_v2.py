from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config import OPENAI_API_KEY, CHAT_MODEL, load_prompts
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.retriever import format_context
from src.retrieval.citation_enforcer import enforce_citations


def answer_question_v2(question: str) -> dict:
    """
    Full Phase 2 RAG pipeline:
    1. Hybrid retrieval (BM25 + vector)
    2. Cross-encoder reranking
    3. LLM answer generation
    4. Citation enforcement audit
    """
    print(f"\nProcessing: '{question}'")

    # Step 1: Hybrid retrieval — cast wide net
    candidates = hybrid_search(question, top_k=5)
    print(f"  Hybrid retrieval: {len(candidates)} candidates")

    if not candidates:
        return {
            "question": question,
            "answer": "I cannot answer this from the available documents.",
            "sources": [],
            "citation_audit": None
        }

    # Step 2: Rerank — narrow to best 3
    reranked = rerank(question, candidates, top_k=3)
    print(f"  After reranking: {len(reranked)} chunks")
    for c in reranked:
        print(f"    [{c.metadata.get('reranker_score')}] {c.metadata.get('chunk_id')}")

    # Step 3: Format context and generate answer
    context = format_context(reranked)
    prompts = load_prompts()
    system_prompt = prompts["rag_answer"]["system"]
    user_prompt = prompts["rag_answer"]["user"].format(
        context=context,
        question=question
    )

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content

    # Step 4: Citation enforcement audit
    print("  Running citation audit...")
    audit = enforce_citations(answer, context)
    print(f"  Audit result: supported={audit['supported']}")

    # Block the answer if audit fails
    if not audit["supported"]:
        answer = (
            "I cannot answer this from the available documents. "
            f"Unsupported claims detected: {audit['unsupported_claims']}"
        )

    sources = [c.metadata.get("chunk_id") for c in reranked]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "citation_audit": audit
    }