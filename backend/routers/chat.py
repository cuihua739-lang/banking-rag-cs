"""Chat router: main Q&A, SSE streaming, and suggestions endpoints."""

import json
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.schemas import ChatRequest, ChatResponse, Citation, RetrievalInfo
from services.retrieval_pipeline import round1_retrieval, round2_retrieval
from services.query_expander import expand_queries
from services.relevance_judge import judge_relevance
from services.response_generator import generate_answer
from services.llm_client import get_client
from config.settings import (
    RELEVANCE_THRESHOLD,
    RELEVANCE_MIN_HIGH_COUNT, RELEVANCE_MIN_BEST, RELEVANCE_FALLBACK,
    RERANK_TOP_K,
)

router = APIRouter(tags=["chat"])


async def _run_full_pipeline(query: str) -> ChatResponse:
    """Run the full RAG pipeline: Round 1 → relevance → Round 2 if needed → answer."""

    round_num = 1
    r2 = None

    # Round 1: retrieval
    r1 = await round1_retrieval(query, top_k=RERANK_TOP_K)
    candidates = r1["candidates"]

    if not candidates:
        return ChatResponse(
            answer="很抱歉，我暂时无法找到与您问题相关的信息。建议您拨打我行24小时客服热线 955XX 或前往就近网点咨询。",
            citations=[],
            retrieval_info=RetrievalInfo(
                round=1, average_score=0.0,
                chunks_considered=0, chunks_used=0,
                retrieval_time_ms=r1["elapsed_ms"],
            ),
        )

    # Try to judge relevance
    avg_score = 7.0
    high_count = len(candidates)
    best_score = 7.0
    try:
        judgment = await judge_relevance(query, candidates)
        avg_score = judgment.get("average_score", 0)
        high_count = judgment.get("high_count", 0)
        best_score = judgment.get("best_score", 0)
    except Exception:
        pass

    # Round 2: deep retrieval if relevance insufficient
    should_go_deep = (
        avg_score < RELEVANCE_THRESHOLD
        or high_count < RELEVANCE_MIN_HIGH_COUNT
        or best_score < RELEVANCE_MIN_BEST
    )

    if should_go_deep:
        try:
            expanded = await expand_queries(query)
            r2 = await round2_retrieval(query, expanded)
            candidates = r2["candidates"]
            round_num = 2

            try:
                judgment = await judge_relevance(query, candidates)
                avg_score = judgment.get("average_score", 0)
            except Exception:
                avg_score = 5.0

            if not candidates or avg_score < RELEVANCE_FALLBACK:
                total_ms = r1["elapsed_ms"] + r2["elapsed_ms"]
                return ChatResponse(
                    answer="您的问题较为专业，我暂时无法提供满意的解答。建议您拨打我行24小时客服热线 955XX 或通过手机银行APP联系在线客服，我们的专业客服人员将为您详细解答。",
                    citations=[],
                    retrieval_info=RetrievalInfo(
                        round=2, average_score=round(avg_score, 2),
                        chunks_considered=len(candidates), chunks_used=0,
                        retrieval_time_ms=total_ms,
                    ),
                )
        except Exception:
            pass

    # Generate final answer
    best_answer = ""
    best_citations: list[Citation] = []
    try:
        gen_result = await generate_answer(query, candidates)
        best_answer = gen_result.get("answer", "")
        used_chunks = gen_result.get("used_chunks", candidates[:3])
        for ch in used_chunks:
            best_citations.append(Citation(
                doc_title=ch.get("doc_title", ""),
                section_title=ch.get("section_title", ""),
                chunk_id=ch.get("chunk_id", ""),
                excerpt=ch.get("text", "")[:200],
            ))
    except Exception:
        pass

    # Fallback answer if LLM generation failed
    if not best_answer and candidates:
        best_answer = "根据我们的知识库，为您找到以下相关信息：\n\n" + "\n\n".join(
            f"**{c.get('doc_title', '')}**\n{c.get('text', '')[:500]}"
            for c in candidates[:3]
        )
        best_citations = [
            Citation(
                doc_title=c.get("doc_title", ""),
                section_title=c.get("section_title", ""),
                chunk_id=c.get("chunk_id", ""),
                excerpt=c.get("text", "")[:200],
            )
            for c in candidates[:3]
        ]

    total_ms = r1["elapsed_ms"]
    if r2 is not None:
        total_ms += r2["elapsed_ms"]

    return ChatResponse(
        answer=best_answer,
        citations=best_citations,
        retrieval_info=RetrievalInfo(
            round=round_num,
            average_score=round(avg_score, 2),
            chunks_considered=len(candidates),
            chunks_used=len(best_citations),
            retrieval_time_ms=total_ms,
        ),
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_pipeline_stream(query: str):
    """Generator-based pipeline that yields SSE events at each step."""

    round_num = 1
    r2 = None
    candidates = []

    # Step 1: Retrieving
    yield _sse({"step": "retrieving", "message": "正在检索相关知识..."})
    r1 = await round1_retrieval(query, top_k=RERANK_TOP_K)
    candidates = r1["candidates"]
    yield _sse({"step": "retrieved", "hits": r1["dense_hits"] + r1["sparse_hits"] + r1["kg_hits"], "candidates": len(candidates)})

    if not candidates:
        yield _sse({"step": "done", "answer": "很抱歉，我暂时无法找到与您问题相关的信息。建议您拨打我行24小时客服热线 955XX 或前往就近网点咨询。", "citations": [], "round": 1})
        return

    # Step 2: Judging relevance
    yield _sse({"step": "judging", "message": "正在评估检索质量..."})
    avg_score = 7.0
    high_count = len(candidates)
    best_score = 7.0
    try:
        judgment = await judge_relevance(query, candidates)
        avg_score = judgment.get("average_score", 0)
        high_count = judgment.get("high_count", 0)
        best_score = judgment.get("best_score", 0)
        yield _sse({"step": "judged", "average_score": avg_score, "high_count": high_count})
    except Exception:
        yield _sse({"step": "judged", "average_score": 7.0, "note": "使用默认评分"})

    # Step 3: Round 2 if needed
    should_go_deep = (
        avg_score < RELEVANCE_THRESHOLD
        or high_count < RELEVANCE_MIN_HIGH_COUNT
        or best_score < RELEVANCE_MIN_BEST
    )

    if should_go_deep:
        try:
            yield _sse({"step": "expanding", "message": "关联度不足，正在进行深度检索..."})
            expanded = await expand_queries(query)
            yield _sse({"step": "expanded", "variants": len(expanded)})

            yield _sse({"step": "retrieving_deep", "message": "正在扩大检索范围..."})
            r2 = await round2_retrieval(query, expanded)
            candidates = r2["candidates"]
            round_num = 2
            yield _sse({"step": "retrieved_deep", "hits": r2["dense_hits"] + r2["sparse_hits"] + r2["kg_hits"], "candidates": len(candidates)})

            try:
                judgment = await judge_relevance(query, candidates)
                avg_score = judgment.get("average_score", 0)
            except Exception:
                avg_score = 5.0
                yield _sse({"step": "judged", "average_score": avg_score, "note": "深度检索后评分"})

            if not candidates or avg_score < RELEVANCE_FALLBACK:
                yield _sse({"step": "done", "answer": "您的问题较为专业，我暂时无法提供满意的解答。建议您拨打我行24小时客服热线 955XX 或通过手机银行APP联系在线客服。", "citations": [], "round": 2})
                return
        except Exception:
            yield _sse({"step": "deep_failed", "message": "深度检索失败，使用首轮结果"})

    # Step 4: Generating answer
    yield _sse({"step": "generating", "message": "正在生成回答..."})

    best_answer = ""
    best_citations: list[dict] = []
    try:
        gen_result = await generate_answer(query, candidates)
        best_answer = gen_result.get("answer", "")
        used_chunks = gen_result.get("used_chunks", candidates[:3])
        for ch in used_chunks:
            best_citations.append({
                "doc_title": ch.get("doc_title", ""),
                "section_title": ch.get("section_title", ""),
                "chunk_id": ch.get("chunk_id", ""),
                "excerpt": ch.get("text", "")[:200],
            })
    except Exception:
        pass

    if not best_answer and candidates:
        best_answer = "根据我们的知识库，为您找到以下相关信息：\n\n" + "\n\n".join(
            f"**{c.get('doc_title', '')}**\n{c.get('text', '')[:500]}"
            for c in candidates[:3]
        )

    yield _sse({
        "step": "done",
        "answer": best_answer,
        "citations": best_citations,
        "round": round_num,
        "average_score": round(avg_score, 2),
        "candidates_used": len(best_citations),
    })


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat endpoint showing each pipeline step."""
    return StreamingResponse(
        _run_pipeline_stream(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/suggestions")
async def suggestions(req: ChatRequest):
    """Generate follow-up question suggestions based on the conversation."""
    client = get_client()
    if client is None:
        return {"suggestions": ["请问还有其他问题吗？"]}

    try:
        response = await client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=512,
            temperature=0.5,
            system="你是一个银行客服助手。根据客户刚刚咨询的问题，生成3个用户可能感兴趣的追问建议。以JSON格式输出：{\"suggestions\": [\"建议1\", \"建议2\", \"建议3\"]}。建议要具体、实用，与银行业务相关。",
            messages=[{"role": "user", "content": f"客户刚问了这个问题：{req.query}\n\n请生成3个追问建议。"}],
        )
        text = ""
        for block in response.content:
            if getattr(block, "text", None):
                text += block.text

        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            return {"suggestions": data.get("suggestions", [])}
    except Exception:
        pass

    return {"suggestions": ["请问还有其他问题吗？"]}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint: question → retrieval → relevance → answer."""
    return await _run_full_pipeline(req.query)
