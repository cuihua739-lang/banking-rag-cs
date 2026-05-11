"""LLM batch relevance judgment service."""

import re
import json
from services.llm_client import get_client
from prompts.relevance_judge import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
from config.settings import JUDGE_BATCH_SIZE


async def judge_relevance(query: str, chunks: list[dict]) -> dict:
    """Judge relevance of chunks to the query using LLM.

    Returns dict with: average_score, high_count, best_score, evaluations
    """
    client = get_client()
    if client is None or not chunks:
        return {"average_score": 0, "high_count": 0, "best_score": 0, "evaluations": []}

    # Batch chunks for evaluation
    batch = chunks[:JUDGE_BATCH_SIZE]
    chunks_text_parts = []
    for i, ch in enumerate(batch):
        chunks_text_parts.append(
            f"[{i}] 标题：{ch.get('doc_title', '')} | 章节：{ch.get('section_title', '')}\n{ch.get('text', '')[:600]}"
        )
    chunks_text = "\n\n---\n\n".join(chunks_text_parts)

    user_msg = USER_MESSAGE_TEMPLATE.format(query=query, chunks_text=chunks_text)

    try:
        response = await client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=2048,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = ""
        for block in response.content:
            if getattr(block, "text", None):
                text += block.text

        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            scores = [e.get("score", 0) for e in data.get("evaluations", [])]
            avg = sum(scores) / len(scores) if scores else 0
            high_count = sum(1 for s in scores if s >= 7)
            best = max(scores) if scores else 0
            return {
                "average_score": round(avg, 2),
                "high_count": high_count,
                "best_score": best,
                "evaluations": data.get("evaluations", []),
                "missing_info": data.get("missing_info", ""),
            }
    except Exception:
        pass

    return {"average_score": 0, "high_count": 0, "best_score": 0, "evaluations": []}
