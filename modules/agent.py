"""LLM agent orchestrating GPT-4o chart analysis."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from tools.audio import synthesize_speech, transcribe_audio
from tools.extract import (PromptLoader, ZoneAnalysis, build_json_schema,
                           build_system_prompt, encode_image,
                           format_user_prompt, parse_analysis_payload)
from tools.memory import EmbeddingMemory, MemoryItem, NullMemory
from tools.websearch import run_web_search

LOGGER = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Raised when the trading agent fails to complete a task."""


class TradingAgent:
    """Trading agent wrapping GPT-4o calls and schema validation."""

    def __init__(self, prompts_path: Path) -> None:
        self._prompts = PromptLoader(prompts_path)
        self._client: Optional[AsyncOpenAI] = None
        self._system_prompt = build_system_prompt(
            self._prompts.load("trading_rules.md"),
            self._prompts.load("labeler.md"),
        )
        self._lock = asyncio.Lock()
        self._embedding_model = settings.embedding_model
        self._memory_top_k = settings.memory_top_k
        self._memory = self._init_memory()

    async def _client_instance(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.openai_api_key and not settings.test_mode:
                raise AgentError("OpenAI API key is not configured")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key or "test")
        return self._client

    async def analyze_chart(
        self,
        image_bytes: bytes,
        caption: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ZoneAnalysis:
        """Run GPT-4o analysis against a chart screenshot."""
        if settings.test_mode:
            return self._test_response(caption, metadata)

        encoded_image = encode_image(image_bytes)
        memory_context = await self._memory_context(caption, metadata)
        web_context = await self._web_search_context(caption)
        user_segments: List[str] = []
        if memory_context:
            user_segments.append(memory_context)
        if web_context:
            user_segments.append(web_context)
        user_segments.append(caption or "Analyze the attached chart.")
        if metadata:
            user_segments.append(json.dumps(metadata, ensure_ascii=False))
        user_prompt = format_user_prompt(user_segments)
        schema = build_json_schema()
        LOGGER.info("Submitting analysis request to OpenAI", extra={"meta": metadata})

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
        ):
            with attempt:
                raw_output = await self._invoke_model(
                    text=user_prompt,
                    image_b64=encoded_image,
                    schema=schema,
                    temperature=0.3,
                )
                LOGGER.debug("Received LLM response", extra={"output": raw_output})
                return parse_analysis_payload(raw_output)

        raise AgentError("LLM analysis attempt failed")

    async def run_feedback_loop(
        self,
        zone: Dict[str, Any],
        price: float,
    ) -> ZoneAnalysis:
        """Request textual re-evaluation when price reaches the zone."""
        if settings.test_mode:
            response = zone.copy()
            response["comment"] = f"Re-evaluated at price {price:.5f}"
            return ZoneAnalysis.model_validate(response)

        context = json.dumps(zone, ensure_ascii=False)
        user_prompt = format_user_prompt(
            [
                "Re-evaluate the stored trading zone given the live price.",
                f"Current price: {price}",
                f"Zone data: {context}",
            ]
        )
        schema = build_json_schema()
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
        ):
            with attempt:
                raw_output = await self._invoke_model(
                    text=user_prompt,
                    image_b64=None,
                    schema=schema,
                    temperature=0.3,
                )
                return parse_analysis_payload(raw_output)
        raise AgentError("LLM feedback loop failed")

    @staticmethod
    def _test_response(caption: Optional[str], metadata: Optional[Dict[str, Any]]) -> ZoneAnalysis:
        """Return deterministic analysis result when test mode is enabled."""
        base_pair = "GBPUSD"
        if caption and "eur" in caption.lower():
            base_pair = "EURUSD"
        payload = {
            "pair": base_pair,
            "bias": "buy",
            "entry": 1.2675,
            "stop": 1.2630,
            "target": 1.2800,
            "zone_type": "DBR",
            "eq_alignment": "buy",
            "enhancer_score": 8.5,
            "comment": "Test mode response.",
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k in payload})
        return ZoneAnalysis.model_validate(payload)

    async def close(self) -> None:
        """Close the OpenAI client session."""
        if self._client:
            await self._client.close()

    async def converse(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Provide a natural language response for interactive chat."""
        if settings.test_mode:
            meta_hint = f" ({metadata})" if metadata else ""
            return f"Test mode reply to: {message}{meta_hint}"

        memory_context = await self._memory_context(message, metadata)
        web_context = await self._web_search_context(message)
        segments: List[str] = []
        if memory_context:
            segments.append(memory_context)
        if web_context:
            segments.append(web_context)
        segments.append(message)
        if metadata:
            segments.append(json.dumps(metadata, ensure_ascii=False))
        user_prompt = format_user_prompt(segments)

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
        ):
            with attempt:
                result = await self._invoke_model(
                    text=user_prompt,
                    image_b64=None,
                    schema=None,
                    temperature=0.4,
                )
                return result.strip()

        raise AgentError("Conversation request failed")

    async def remember_zone(
        self,
        zone_id: int,
        analysis: ZoneAnalysis,
        metadata: Dict[str, Any],
    ) -> None:
        """Persist analysis outcome into vector memory."""
        if isinstance(self._memory, NullMemory):
            return
        summary = self._analysis_summary(analysis)
        vector = await self._embed_text(summary)
        if vector is None:
            return
        enriched_metadata = {
            "zone_id": zone_id,
            "pair": analysis.pair,
            "bias": analysis.bias,
            "entry": analysis.entry,
            "stop": analysis.stop,
            "target": analysis.target,
            "zone_type": analysis.zone_type,
            "eq_alignment": analysis.eq_alignment,
            "enhancer_score": analysis.enhancer_score,
            "comment": analysis.comment,
            "type": "zone",
            "snippet": analysis.comment,
            **{k: v for k, v in metadata.items() if k not in {"comment"}},
        }
        await self._memory.add_entry(summary, vector, enriched_metadata)

    async def ingest_document(
        self,
        title: str,
        chunks: List[str],
        metadata: Dict[str, Any],
    ) -> int:
        """Store document chunks in vector memory."""
        if isinstance(self._memory, NullMemory):
            return 0
        stored = 0
        for index, chunk in enumerate(chunks):
            vector = await self._embed_text(chunk)
            if vector is None:
                continue
            summary = f"{title} section {index + 1}: {chunk[:120]}"
            enriched_metadata = {
                "type": "document",
                "doc_title": title,
                "chunk_index": index,
                "snippet": chunk[:400],
                "content": chunk,
                **metadata,
            }
            await self._memory.add_entry(summary, vector, enriched_metadata)
            stored += 1
        return stored

    def _init_memory(self) -> EmbeddingMemory | NullMemory:
        if not settings.memory_enabled:
            return NullMemory()
        try:
            dimension = self._embedding_dimension(self._embedding_model)
            return EmbeddingMemory(Path(settings.memory_path), dimension)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vector memory unavailable; disabling", exc_info=exc)
            return NullMemory()

    @staticmethod
    def _embedding_dimension(model: str) -> int:
        mapping = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return mapping.get(model, 1536)

    async def _memory_context(
        self,
        caption: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if isinstance(self._memory, NullMemory):
            return None
        query_parts: List[str] = []
        if caption:
            query_parts.append(caption)
        if metadata:
            pair = metadata.get("pair")
            if pair:
                query_parts.append(f"PAIR {pair}")
            caption_text = metadata.get("caption")
            if caption_text:
                query_parts.append(caption_text)
        if not query_parts:
            return None
        query_text = " | ".join(query_parts)
        vector = await self._embed_text(query_text)
        if vector is None:
            return None
        items = await self._memory.search(vector, top_k=self._memory_top_k)
        if not items:
            return None
        return self._format_memory_prompt(items)

    async def _web_search_context(self, text: Optional[str]) -> Optional[str]:
        if not text or not self._should_web_search(text):
            return None
        client = await self._client_instance()
        summary = await run_web_search(client, text)
        if not summary:
            return None
        return f"Web search summary:\n{summary}"

    @staticmethod
    def _should_web_search(text: str) -> bool:
        lowered = text.lower()
        triggers = ["fundamental", "macro", "sentiment", "news", "economic"]
        return any(trigger in lowered for trigger in triggers)

    @staticmethod
    def _format_memory_prompt(items: List[MemoryItem]) -> str:
        lines = ["Historical references:"]
        for item in items:
            meta = item.metadata
            entry_type = meta.get("type", "zone")
            if entry_type == "document":
                title = meta.get("doc_title", "Document")
                chunk_index = meta.get("chunk_index", "?")
                snippet = meta.get("snippet", "")
                lines.append(
                    f"- Doc {title} [section {chunk_index}] :: {snippet}"
                )
            else:
                pair = meta.get("pair", "?")
                bias = meta.get("bias", "?")
                entry = meta.get("entry", "?")
                comment = meta.get("comment", meta.get("snippet", ""))
                zone_id = meta.get("zone_id", "?")
                lines.append(
                    f"- Case {zone_id} | {pair} {bias} entry {entry} | {comment}"
                )
        return "\n".join(lines)

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        if settings.test_mode:
            return self._hash_embedding(text)
        client = await self._client_instance()
        response = await client.embeddings.create(
            model=self._embedding_model,
            input=[text],
        )
        embedding = response.data[0].embedding
        return list(embedding)

    def _hash_embedding(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rand = random.Random(digest)
        dimension = self._embedding_dimension(self._embedding_model)
        return [rand.random() for _ in range(dimension)]

    @staticmethod
    def _analysis_summary(analysis: ZoneAnalysis) -> str:
        return (
            f"{analysis.pair} {analysis.bias} setup | entry {analysis.entry:.5f} | "
            f"stop {analysis.stop:.5f} | target {analysis.target:.5f} | "
            f"zone {analysis.zone_type} | eq {analysis.eq_alignment} | "
            f"enhancer {analysis.enhancer_score:.1f} | comment {analysis.comment}"
        )

    async def transcribe_voice(self, audio_path: Path) -> Optional[str]:
        """Transcribe a voice note to text."""
        client = await self._client_instance()
        return await transcribe_audio(client, audio_path)

    async def synthesize_voice(self, text: str) -> Optional[bytes]:
        """Generate TTS audio for alerting."""
        client = await self._client_instance()
        return await synthesize_speech(client, text)

    async def analyze_csv(self, csv_path: Path) -> str:
        """Run code-interpreter analysis on uploaded CSV trades."""
        if settings.test_mode:
            return "Test CSV analysis summary: win rate 55%, avg RR 1.8."
        client = await self._client_instance()
        try:
            with csv_path.open("rb") as handle:
                uploaded = await client.files.create(file=handle, purpose="assistants")
            prompt = (
                "Review the attached CSV containing trade history. "
                "Calculate total trades, win rate, average risk-to-reward, max drawdown, "
                "and provide 2 bullet recommendations."
            )
            response = await client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                tools=[{"type": "code_interpreter"}],
                attachments=[
                    {
                        "file_id": uploaded.id,
                        "tools": [{"type": "code_interpreter"}],
                    }
                ],
                temperature=0.2,
            )
            return getattr(response, "output_text", "").strip() or "No analysis produced."
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("CSV analysis failed", exc_info=exc)
            raise AgentError("CSV analysis failed") from exc

    async def _invoke_model(
        self,
        *,
        text: str,
        image_b64: Optional[str],
        schema: Optional[Dict[str, Any]],
        temperature: float,
    ) -> str:
        """Call OpenAI API handling both Responses and ChatCompletion interfaces."""
        client = await self._client_instance()

        if hasattr(client, "responses"):
            input_blocks = [{"type": "input_text", "text": text}]
            if image_b64:
                input_blocks.append({"type": "input_image", "image": {"b64_json": image_b64}})
            kwargs: Dict[str, Any] = {
                "model": "gpt-4o",
                "temperature": temperature,
                "system": self._system_prompt,
                "input": [{"role": "user", "content": input_blocks}],
            }
            if schema:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
            response = await client.responses.create(**kwargs)
            return getattr(response, "output_text", "").strip()

        # Fallback for older SDK versions lacking the Responses API
        message_content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        if image_b64:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )
        kwargs_chat: Dict[str, Any] = {
            "model": "gpt-4o",
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": message_content},
            ],
        }
        if schema:
            kwargs_chat["response_format"] = {"type": "json_schema", "json_schema": schema}
        completion = await client.chat.completions.create(**kwargs_chat)
        message = completion.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text_val = item.get("text") or item.get("content")
                    if text_val:
                        parts.append(text_val)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)
