"""Query expansion for Round 2 deep retrieval."""

import re
import json
from services.llm_client import get_client
from prompts.query_expansion import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE


async def expand_queries(query: str) -> list[str]:
    """Expand a query into multiple variants for deeper retrieval.

    Returns list of expanded query strings (up to 3).
    """
    client = get_client()
    if client is None:
        return [query]

    user_msg = USER_MESSAGE_TEMPLATE.format(query=query)

    try:
        response = await client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=1024,
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
            variants = data.get("variants", [])
            return variants[:3] if variants else [query]
    except Exception:
        pass

    return [query]
