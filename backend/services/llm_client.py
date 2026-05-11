import os
import asyncio
import anthropic
from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


def get_client() -> anthropic.AsyncAnthropic | None:
    if not ANTHROPIC_API_KEY:
        return None
    return anthropic.AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
        base_url=ANTHROPIC_BASE_URL,
    )


async def generate_text(
    system_prompt: str,
    user_message: str,
    *,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("ANTHROPIC_API_KEY 环境变量未设置")

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if getattr(block, "text", None):
                    return block.text
            raise RuntimeError("模型未返回文本内容")
        except anthropic.RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        except anthropic.APIError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    raise RuntimeError(f"AI 服务暂时不可用: {last_error}")
