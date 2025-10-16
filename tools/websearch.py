"""Wrapper for invoking OpenAI web search capability."""
from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from config import settings

LOGGER = logging.getLogger(__name__)


async def run_web_search(client: AsyncOpenAI, query: str) -> Optional[str]:
    """Execute web search via OpenAI Responses API and return summarized findings."""
    if not settings.web_search_enabled:
        return None
    if settings.test_mode:
        return f"Test web search results for: {query}"
    try:
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Use web search to gather current macro/fundamental insights for the "
                                f"following query and return a concise, source-cited summary.\nQuery: {query}"
                            ),
                        }
                    ],
                }
            ],
            tools=[{"type": "web_search"}],
            temperature=0.1,
        )
        return getattr(response, "output_text", "").strip() or None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Web search failed", exc_info=exc)
        return None
