"""Final answer generation service."""

import re
import json
from services.llm_client import get_client
from prompts.response_gen import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
from config.settings import RESPONSE_TOP_CHUNKS


async def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate a final answer from retrieved chunks.

    Returns dict with: answer, used_chunks (list of chunk dicts)
    """
    client = get_client()
    if client is None or not chunks:
        return {"answer": "", "used_chunks": []}

    # Use top chunks for context
    top_chunks = chunks[:RESPONSE_TOP_CHUNKS]
    chunks_text_parts = []
    for i, ch in enumerate(top_chunks):
        chunks_text_parts.append(
            f"[{i}] **{ch.get('doc_title', '')}** ({ch.get('category', '')})\n{ch.get('text', '')}"
        )
    chunks_text = "\n\n---\n\n".join(chunks_text_parts)

    user_msg = USER_MESSAGE_TEMPLATE.format(query=query, chunks_text=chunks_text)

    try:
        response = await client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=2048,
            temperature=0.3,
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
            answer = data.get("answer", text)
            used_indices = data.get("used_chunk_indices", list(range(min(3, len(top_chunks)))))
            used_chunks = [top_chunks[i] for i in used_indices if i < len(top_chunks)]
            return {"answer": answer, "used_chunks": used_chunks}

        # If JSON parsing fails, treat entire response as answer
        return {"answer": text.strip(), "used_chunks": top_chunks[:3]}
    except Exception:
        return {"answer": "", "used_chunks": []}
