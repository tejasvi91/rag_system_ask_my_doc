import time
import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config import OPENAI_API_KEY, CHAT_MODEL, load_prompts
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.retriever import format_context
from src.retrieval.citation_enforcer import enforce_citations
from src.observability.tracer import get_langfuse, flush
from src.observability.metrics import log_request_metric


def answer_question_v3(question: str) -> dict:
    """
    Full Phase 2 pipeline with complete Langfuse observability.
    Every step is traced — chunks, scores, prompt, response, tokens, latency.
    """
    langfuse = get_langfuse()
    total_start = time.time()

    # Create a top-level trace for this entire request
    trace = langfuse.trace(
        name="rag-request",
        input={"question": question},
        metadata={"pipeline_version": "v3", "model": CHAT_MODEL}
    )

    try:
        # ── Step 1: Hybrid retrieval ──────────────────────────────
        t0 = time.time()
        retrieval_span = trace.span(
            name="hybrid-retrieval",
            input={"query": question, "top_k": 5}
        )
        candidates = hybrid_search(question, top_k=5)
        retrieval_latency = time.time() - t0

        retrieval_span.end(
            output={
                "num_candidates": len(candidates),
                "chunk_ids": [c.metadata.get("chunk_id") for c in candidates]
            },
            metadata={"latency_seconds": round(retrieval_latency, 3)}
        )

        if not candidates:
            trace.update(
                output={"answer": "I cannot answer this from the available documents."},
                metadata={"status": "no_candidates"}
            )
            flush()
            return {
                "question": question,
                "answer": "I cannot answer this from the available documents.",
                "sources": [],
                "trace_id": trace.id
            }

        # ── Step 2: Reranking ─────────────────────────────────────
        t0 = time.time()
        rerank_span = trace.span(
            name="reranking",
            input={"num_candidates": len(candidates)}
        )
        reranked = rerank(question, candidates, top_k=3)
        rerank_latency = time.time() - t0

        rerank_span.end(
            output={
                "top_chunks": [
                    {
                        "chunk_id": c.metadata.get("chunk_id"),
                        "score": c.metadata.get("reranker_score"),
                        "preview": c.page_content[:100]
                    }
                    for c in reranked
                ]
            },
            metadata={"latency_seconds": round(rerank_latency, 3)}
        )

        # ── Step 3: LLM answer generation ─────────────────────────
        context = format_context(reranked)
        prompts = load_prompts()
        system_prompt = prompts["rag_answer"]["system"]
        user_prompt = prompts["rag_answer"]["user"].format(
            context=context,
            question=question
        )

        t0 = time.time()
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )

        # Log the generation to Langfuse with full token tracking
        generation = trace.generation(
            name="llm-answer",
            model=CHAT_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            metadata={"prompt_version": prompts["rag_answer"]["version"]}
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        llm_latency = time.time() - t0

        # Calculate token counts
        input_tokens = llm.get_num_tokens(system_prompt + user_prompt)
        output_tokens = llm.get_num_tokens(response.content)

        # Cost estimate — gpt-4o-mini pricing
        cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)

        generation.end(
            output=response.content,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            metadata={
                "latency_seconds": round(llm_latency, 3),
                "estimated_cost_usd": round(cost, 6)
            }
        )

        # ── Step 4: Citation enforcement ──────────────────────────
        t0 = time.time()
        citation_span = trace.span(
            name="citation-audit",
            input={"answer": response.content}
        )
        audit = enforce_citations(response.content, context)
        citation_latency = time.time() - t0

        citation_span.end(
            output=audit,
            metadata={"latency_seconds": round(citation_latency, 3)}
        )

        # Block if unsupported
        final_answer = response.content
        if not audit["supported"]:
            final_answer = (
                "I cannot answer this from the available documents. "
                f"Unsupported claims: {audit['unsupported_claims']}"
            )

        # ── Final trace update ─────────────────────────────────────
        total_latency = time.time() - total_start
        sources = [c.metadata.get("chunk_id") for c in reranked]

        trace.update(
            output={"answer": final_answer, "sources": sources},
            metadata={
                "total_latency_seconds": round(total_latency, 3),
                "total_tokens": input_tokens + output_tokens,
                "estimated_cost_usd": round(cost, 6),
                "citation_supported": audit["supported"],
                "status": "success"
            }
        )

        log_request_metric(
            question=question,
            latency_seconds=round(total_latency, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 6),
            citation_supported=audit["supported"],
            status="success",
            trace_id=trace.id
        )

        flush()

        return {
            "question": question,
            "answer": final_answer,
            "sources": sources,
            "trace_id": trace.id,
            "metrics": {
                "total_latency": round(total_latency, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": round(cost, 6),
                "citation_supported": audit["supported"]
            }
        }

    except Exception as e:
        trace.update(
            metadata={"status": "error", "error": str(e)}
        )
        flush()
        raise