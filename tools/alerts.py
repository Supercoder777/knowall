"""Price alerting utilities backed by APScheduler."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from telegram import Bot, InputFile

from config import settings
from modules.agent import TradingAgent
from tools.logger import get_active_zones, log_zone, update_zone_status
from tools.price_feed import AbstractPriceFeed

LOGGER = logging.getLogger(__name__)


def _pip_value(pair: str) -> float:
    suffix = pair[-3:]
    return 0.01 if suffix == "JPY" else 0.0001


class PriceAlertService:
    """Monitor tracked zones and dispatch Telegram alerts."""

    def __init__(
        self,
        price_feed: AbstractPriceFeed,
        agent: TradingAgent,
        bot: Bot,
        scheduler: Optional[AsyncIOScheduler] = None,
    ) -> None:
        self._price_feed = price_feed
        self._agent = agent
        self._bot = bot
        self._scheduler = scheduler or AsyncIOScheduler(timezone=settings.scheduler_timezone)
        self._alerted: Dict[int, float] = {}
        self._lock = asyncio.Lock()
        self._audio_dir = Path(settings.audio_path)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the scheduler if not already running."""
        if not self._scheduler.running:
            trigger = IntervalTrigger(minutes=1, timezone=settings.scheduler_timezone)
            self._scheduler.add_job(self.check_zones, trigger=trigger, id="price-check")
            self._scheduler.start()
            LOGGER.info("Price alert scheduler started")

    async def shutdown(self) -> None:
        """Shutdown scheduler gracefully."""
        if self._scheduler.running:
            await self._scheduler.shutdown(wait=False)
            LOGGER.info("Price alert scheduler stopped")

    async def check_zones(self) -> None:
        """Evaluate tracked zones and send alerts when thresholds are met."""
        async with self._lock:
            zones = await get_active_zones()
            if not zones:
                return
            for zone in zones:
                await self._evaluate_zone(zone)

    async def _evaluate_zone(self, zone: Dict[str, Any]) -> None:
        pair = zone["pair"].upper()
        price = await self._price_feed.get_price(pair)
        if price is None:
            LOGGER.warning("Price unavailable", extra={"pair": pair})
            return
        threshold_high = _pip_value(pair) * 80
        threshold_low = _pip_value(pair) * 50
        diff = abs(price - float(zone["entry"]))
        LOGGER.debug(
            "Zone evaluation",
            extra={"zone_id": zone["id"], "price": price, "diff": diff, "pair": pair},
        )
        if diff <= threshold_high:
            if diff >= threshold_low:
                await self._send_alert(zone, price)
            if diff <= _pip_value(pair):
                await self._handle_zone_touch(zone, price)

    async def _send_alert(self, zone: Dict[str, Any], price: float) -> None:
        zone_id = zone["id"]
        last_price = self._alerted.get(zone_id)
        if last_price and abs(last_price - price) < _pip_value(zone["pair"]):
            return
        chat_id = zone.get("chat_id")
        if not chat_id:
            return
        direction = zone["bias"].upper()
        message = (
            "⚠️ {pair} approaching {direction} zone {entry:.5f} (now {price:.5f})"
        ).format(pair=zone["pair"], direction=direction, entry=zone["entry"], price=price)
        await self._bot.send_message(chat_id=chat_id, text=message)
        self._alerted[zone_id] = price
        LOGGER.info("Alert dispatched", extra={"zone_id": zone_id, "price": price})
        if settings.audio_enabled:
            audio_bytes = await self._agent.synthesize_voice(message)
            if audio_bytes:
                audio_path = self._audio_dir / f"alert_{zone_id}_{int(time.time())}.mp3"
                audio_path.write_bytes(audio_bytes)
                with audio_path.open("rb") as audio_file:
                    await self._bot.send_voice(
                        chat_id=chat_id,
                        voice=InputFile(audio_file, filename=audio_path.name),
                    )

    async def _handle_zone_touch(self, zone: Dict[str, Any], price: float) -> None:
        chat_id = zone.get("chat_id")
        if not chat_id:
            return
        LOGGER.info("Zone touched; requesting re-evaluation", extra={"zone_id": zone["id"]})
        analysis = await self._agent.run_feedback_loop(zone, price)
        payload = analysis.to_dict()
        payload.update({
            "chat_id": chat_id,
            "user_id": zone.get("user_id"),
            "comment": payload["comment"],
            "status": "rechecked",
        })
        new_zone_id = await log_zone(payload)
        await self._agent.remember_zone(
            new_zone_id,
            analysis,
            {
                "chat_id": chat_id,
                "user_id": zone.get("user_id"),
                "trigger_price": price,
                "caption": "Zone revalidation",
            },
        )
        await update_zone_status(zone["id"], "rechecked")
        message = (
            "✅ {pair} zone hit at {price:.5f}. Recheck summary: {comment}"
        ).format(pair=zone["pair"], price=price, comment=payload["comment"])
        await self._bot.send_message(chat_id=chat_id, text=message)
        LOGGER.info(
            "Re-evaluation completed",
            extra={"zone_id": zone["id"], "status": "rechecked"},
        )

    async def manual_check(self) -> None:
        """Public method to trigger alerts on demand."""
        await self.check_zones()

    def running(self) -> bool:
        """Whether scheduler is active."""
        return self._scheduler.running
