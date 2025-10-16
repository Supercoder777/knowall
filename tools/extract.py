"""LLM response extraction and validation utilities."""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.config import ConfigDict


LOGGER = logging.getLogger(__name__)


class ZoneAnalysis(BaseModel):
    """Validated representation of a trading zone analysis."""

    model_config = ConfigDict(extra="forbid")

    pair: str = Field(..., description="Market pair symbol, e.g. GBPUSD.")
    bias: Literal["buy", "sell"] = Field(..., description="Directional bias for the setup.")
    entry: float = Field(..., description="Proposed entry price.")
    stop: float = Field(..., description="Protective stop price.")
    target: float = Field(..., description="Target price for the trade.")
    zone_type: str = Field(..., description="Smart Money zone classification.")
    eq_alignment: Literal["buy", "sell", "neutral"] = Field(
        ..., description="Directional bias from equilibrium alignment."
    )
    enhancer_score: float = Field(
        ..., ge=0.0, le=10.0, description="Score reflecting confluence enhancers."
    )
    comment: str = Field(..., description="Supporting commentary from the model.")

    @field_validator("pair")
    @classmethod
    def _upper_pair(cls, value: str) -> str:
        return value.upper().replace("/", "")

    def to_dict(self) -> Dict[str, Any]:
        """Return data as standard dictionary."""
        return self.model_dump()


class ExtractionError(RuntimeError):
    """Raised when LLM response parsing fails."""


@dataclass
class PromptLoader:
    """Load prompt content from the prompts directory."""

    base_path: Path

    def load(self, name: str) -> str:
        """Read a prompt file relative to the base path."""
        path = self.base_path / name
        if not path.exists():
            msg = f"Prompt file not found: {path}"
            raise FileNotFoundError(msg)
        return path.read_text(encoding="utf-8")


def safe_json_parse(raw: str) -> Dict[str, Any]:
    """Parse JSON from raw text, handling fenced code blocks."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines()[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse JSON payload", extra={"payload": text})
        raise ExtractionError("LLM output is not valid JSON") from exc


def encode_image(image_bytes: bytes) -> str:
    """Return base64 representation suitable for OpenAI vision input."""
    return base64.b64encode(image_bytes).decode("ascii")


def build_json_schema() -> Dict[str, Any]:
    """Return JSON schema for the expected LLM response."""
    schema = ZoneAnalysis.model_json_schema()
    schema.setdefault("additionalProperties", False)
    return {
        "name": "zone_analysis",
        "schema": schema,
        "strict": True,
    }


def format_user_prompt(prompt_segments: Iterable[str]) -> str:
    """Join user prompt segments with separation."""
    return "\n\n".join(segment.strip() for segment in prompt_segments if segment.strip())


def parse_analysis_payload(raw_payload: str | Dict[str, Any]) -> ZoneAnalysis:
    """Return validated zone analysis from raw LLM output."""
    if isinstance(raw_payload, str):
        data = safe_json_parse(raw_payload)
    else:
        data = raw_payload
    try:
        return ZoneAnalysis.model_validate(data)
    except ValidationError as exc:
        LOGGER.error("LLM response failed schema validation", extra={"errors": exc.errors()})
        raise ExtractionError("LLM response failed validation") from exc


def build_system_prompt(core_rules: str, labeler: str) -> str:
    """Combine system prompt components."""
    return (
        "You are Dami's IOF Smart Money assistant.\n"
        "Follow the institutional order flow methodology precisely.\n"
        "Use the following trading doctrine and labeling instructions.\n\n"
        "-- CORE RULES --\n"
        f"{core_rules.strip()}\n\n"
        "-- LABELING GUIDELINES --\n"
        f"{labeler.strip()}"
    )
