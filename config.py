"""Application configuration management."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    """Return environment variable value as boolean."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Container for runtime configuration values."""

    telegram_token: str
    openai_api_key: str
    database_url: str
    bot_admin_id: Optional[int]
    price_provider: str
    binance_api_key: Optional[str]
    binance_api_secret: Optional[str]
    oanda_account_id: Optional[str]
    oanda_api_key: Optional[str]
    scheduler_timezone: str
    test_mode: bool
    fastapi_host: str
    fastapi_port: int
    memory_enabled: bool
    memory_path: str
    embedding_model: str
    memory_top_k: int
    knowledge_path: str
    audio_enabled: bool
    audio_voice: str
    audio_path: str
    web_search_enabled: bool
    code_analysis_enabled: bool


def get_settings() -> Settings:
    """Parse environment variables into a Settings object."""
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/zones.db")
    price_provider = os.getenv("PRICE_PROVIDER", "binance").lower()
    scheduler_timezone = os.getenv("SCHEDULER_TIMEZONE", "UTC")
    fastapi_host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    fastapi_port = int(os.getenv("FASTAPI_PORT", "8000"))
    memory_enabled = _get_bool("MEMORY_ENABLED", True)
    memory_path = os.getenv("MEMORY_PATH", "data/memory")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    memory_top_k = int(os.getenv("MEMORY_TOP_K", "3"))
    knowledge_path = os.getenv("KNOWLEDGE_PATH", "data/docs")
    audio_enabled = _get_bool("AUDIO_ENABLED", True)
    audio_voice = os.getenv("AUDIO_VOICE", "alloy")
    audio_path = os.getenv("AUDIO_PATH", "data/audio")
    web_search_enabled = _get_bool("WEB_SEARCH_ENABLED", True)
    code_analysis_enabled = _get_bool("CODE_ANALYSIS_ENABLED", True)

    return Settings(
        telegram_token=telegram_token,
        openai_api_key=openai_api_key,
        database_url=database_url,
        bot_admin_id=int(os.getenv("BOT_ADMIN_ID", "0")) or None,
        price_provider=price_provider,
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_secret=os.getenv("BINANCE_API_SECRET"),
        oanda_account_id=os.getenv("OANDA_ACCOUNT_ID"),
        oanda_api_key=os.getenv("OANDA_API_KEY"),
        scheduler_timezone=scheduler_timezone,
        test_mode=_get_bool("TEST_MODE", False),
        fastapi_host=fastapi_host,
        fastapi_port=fastapi_port,
        memory_enabled=memory_enabled,
        memory_path=memory_path,
        embedding_model=embedding_model,
        memory_top_k=memory_top_k,
        knowledge_path=knowledge_path,
        audio_enabled=audio_enabled,
        audio_voice=audio_voice,
        audio_path=audio_path,
        web_search_enabled=web_search_enabled,
        code_analysis_enabled=code_analysis_enabled,
    )


settings = get_settings()
