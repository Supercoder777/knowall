"""Audio text-to-speech and transcription helpers."""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from config import settings

LOGGER = logging.getLogger(__name__)


async def synthesize_speech(client: AsyncOpenAI, text: str, *, voice: Optional[str] = None) -> Optional[bytes]:
    """Generate speech audio from text using OpenAI TTS."""
    if not settings.audio_enabled:
        return None
    if settings.test_mode:
        return text.encode("utf-8")
    try:
        voice_name = voice or settings.audio_voice
        async with client.audio.speech.with_streaming_response.create(  # type: ignore[attr-defined]
            model="gpt-4o-mini-tts",
            voice=voice_name,
            input=text,
        ) as response:
            return await response.read()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Speech synthesis failed", exc_info=exc)
        return None


async def transcribe_audio(client: AsyncOpenAI, audio_path: Path) -> Optional[str]:
    """Transcribe audio content to text via OpenAI."""
    if settings.test_mode:
        return f"Transcribed (test) from {audio_path.name}"
    try:
        with audio_path.open("rb") as audio_file:
            result = await client.audio.transcriptions.create(  # type: ignore[attr-defined]
                model="gpt-4o-mini-transcribe",
                file=audio_file,
            )
        return getattr(result, "text", None)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Audio transcription failed", exc_info=exc)
        return None
